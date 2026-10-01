"""Local POSIX run ownership and durable, conservative allocation accounting.

Every second in the session counts, including setup, validation and I/O. A lost
attempt is charged through the next recovery (including downtime); clock rollback
refuses recovery. Locks require a local filesystem with working flock/fsync.
"""

from __future__ import annotations

import copy
import ctypes
import fcntl
import hashlib
import json
import math
import os
import select
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import torch


def _sync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _sync_dir(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_checkpoint(
    path: str | Path, value: dict[str, Any], *, retain_previous: bool = False
) -> None:
    """Fully serialize/fsync/verify before replacing; keep one verified predecessor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        torch.load(name, map_location="cpu", weights_only=False)
        if retain_previous and path.exists():
            # Never rotate a corrupt current file over the known-good previous.
            torch.load(path, map_location="cpu", weights_only=False)
            previous = path.with_name(path.stem + ".previous" + path.suffix)
            link = path.parent / (".previous-" + uuid.uuid4().hex)
            try:
                os.link(path, link)
                os.replace(link, previous)
                _sync_dir(path.parent)
            finally:
                link.unlink(missing_ok=True)
        os.replace(name, path)
        _sync_dir(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def config_identity(config: dict[str, Any]) -> str:
    value = copy.deepcopy(config)
    runtime = value.pop("runtime", {})
    # Cadence changes selection semantics and therefore is part of the experiment.
    value["validation_every"] = runtime.get("validation_every", 1)
    for key in ("model_save_dir", "results_save_dir", "run_name"):
        value.get("pathing", {}).pop(key, None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _process_identity(pid: int) -> dict[str, Any]:
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "start_ticks": stat[19],
        "host": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def read_status(run_dir: str | Path) -> dict[str, Any]:
    directory = Path(run_dir)
    state: dict[str, Any] = json.loads((directory / "runtime.json").read_text())
    owner = state.get("owner", {})
    try:
        current = _process_identity(owner["pid"])
        verified = current == owner
    except (OSError, KeyError):
        verified = False
    state["ownership_verified"] = verified
    state["ownership"] = (
        "live"
        if verified and state["terminal_reason"] is None
        else ("released" if state["terminal_reason"] is not None else "stale")
    )
    if owner.get("host") != socket.gethostname():
        state["ownership"] = "unknown"
    if state["terminal_reason"] is None and not verified:
        state["terminal_reason"] = "unknown"
    state["heartbeat_age_seconds"] = max(0, time.time() - state.get("heartbeat_at", 0))
    state["progress_age_seconds"] = max(0, time.time() - state.get("progress_at", 0))
    threshold = state.get("resolved_config", {}).get("runtime", {}).get("stall_seconds")
    state["stalled"] = bool(
        state["ownership"] == "live"
        and threshold is not None
        and state["progress_age_seconds"] > threshold
    )
    state["storage_free_bytes"] = shutil.disk_usage(directory).free
    return state


def _libc_pidfd_call(name: str, argtypes: list[Any], *args: Any) -> int:
    """Use libc's ABI when this Python build omits Linux pidfd bindings."""
    try:
        function = getattr(ctypes.CDLL(None, use_errno=True), name)
    except (AttributeError, OSError) as exc:
        raise RuntimeError(
            f"Safe stop unavailable: neither Python nor libc exposes {name}; "
            "PID-based signaling is not a safe fallback"
        ) from exc
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    result = int(function(*args))
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, f"{name}: {os.strerror(error)}")
    return result


def _pidfd_open(pid: int) -> int:
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return int(native(pid, 0))
    return _libc_pidfd_call("pidfd_open", [ctypes.c_int, ctypes.c_uint], pid, 0)


def _pidfd_send_signal(fd: int, sig: int) -> None:
    native = getattr(signal, "pidfd_send_signal", None)
    if native is not None:
        native(fd, sig, None, 0)
        return
    _libc_pidfd_call(
        "pidfd_send_signal",
        [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint],
        fd,
        sig,
        None,
        0,
    )


def _bounded_status(
    run_dir: str | Path, seconds: float
) -> tuple[dict[str, Any] | None, str | None]:
    """Bound caller waiting even when run-filesystem I/O blocks indefinitely.

    A timed-out reader is daemonized, has no write/signal authority, and cannot
    prevent the stop command exiting. Never join it during timeout cleanup.
    """
    ready = threading.Event()
    result: list[dict[str, Any]] = []
    errors: list[str] = []

    def read() -> None:
        try:
            result.append(read_status(run_dir))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            ready.set()

    threading.Thread(target=read, daemon=True).start()
    if not ready.wait(seconds):
        return None, "status lookup timed out"
    return (result[0], None) if result else (None, errors[0])


def request_stop(
    run_dir: str | Path, *, wait_seconds: float | None = None, force: bool = False
) -> dict[str, Any]:
    """Request cooperation; optionally wait, then explicitly escalate to SIGKILL.

    The initial status lookup is unbounded. Later status reads each get at most
    min(wait_seconds, 1) seconds (one second without a wait). Missing fresh owner
    evidence refuses signaling, while reporting cached checkpoint information.
    """
    if wait_seconds is not None and (
        isinstance(wait_seconds, bool) or not math.isfinite(wait_seconds) or wait_seconds <= 0
    ):
        raise ValueError("wait_seconds must be positive finite seconds")
    if force and wait_seconds is None:
        raise ValueError("Force requires an explicit bounded wait")
    state = read_status(run_dir)
    if state["ownership"] != "live":
        raise RuntimeError("Refusing signal: run has no verified live owner")
    owner = state["owner"]
    cached = state
    status_error: str | None = None
    status_available = False
    requested = False
    forced = False
    status_seconds = min(wait_seconds, 1.0) if wait_seconds is not None else 1.0

    def same_attempt(current: dict[str, Any]) -> bool:
        return all(current.get(key) == state.get(key) for key in ("owner", "run_id", "attempt_id"))

    def refresh() -> dict[str, Any] | None:
        nonlocal cached, status_error, status_available
        current, status_error = _bounded_status(run_dir, status_seconds)
        status_available = current is not None and same_attempt(current)
        if status_available:
            assert current is not None
            cached = current
        elif current is not None:
            status_error = "run attempt changed"
        return current

    def report(outcome: str, exited: bool, *, refused: bool = False) -> dict[str, Any]:
        reason = cached.get("terminal_reason") if status_available else "unknown"
        if exited and reason is None:
            reason = "unknown"
        return {
            "outcome": outcome,
            "stop_requested": requested,
            "process_exited": exited,
            "force_requested": forced,
            "force_refused": refused,
            "clean_exit": exited
            and not forced
            and status_available
            and reason in ("user stopped", "completed", "budget exhausted"),
            "terminal_reason": reason,
            "status_available": status_available,
            "status_error": status_error,
            "last_committed_checkpoint": cached.get("last_committed_checkpoint"),
            "checkpoint_source": "refreshed_status" if status_available else "cached_status",
            "owner": owner,
            "attempt_id": state.get("attempt_id"),
        }

    try:
        fd = _pidfd_open(owner["pid"])
    except ProcessLookupError:
        # No process occupied this PID at acquisition; the initially verified
        # attempt exited. Never reopen or signal a replacement numeric PID.
        refresh()
        return report("exited", True)
    try:
        poller = select.poll()
        poller.register(fd, select.POLLIN)

        def exited_now() -> bool:
            return bool(poller.poll(0))

        def verify_target(current: dict[str, Any]) -> bool:
            if exited_now():
                return False
            # Terminal publication changes the ownership label before process exit.
            # It does not change the identity of the attempt pinned by the pidfd.
            if not same_attempt(current):
                raise RuntimeError("Owner changed before stop request")
            try:
                identity = _process_identity(owner["pid"])
            except (FileNotFoundError, ProcessLookupError):
                if exited_now():
                    return False
                raise
            if identity != owner:
                if exited_now():
                    return False
                raise RuntimeError("Owner changed before stop request")
            return True

        def send(sig: int) -> bool:
            try:
                _pidfd_send_signal(fd, sig)
                return True
            except ProcessLookupError:
                # ESRCH from a pinned pidfd means this target no longer exists.
                return False

        def wait_for_exit(seconds: float) -> bool:
            deadline = time.monotonic() + seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return exited_now()
                if poller.poll(math.ceil(min(remaining, 60) * 1000)):
                    return True

        current = refresh()
        if current is None:
            return report("stop_refused_status_unavailable", exited_now())
        if not verify_target(current):
            return report("exited", True)
        if current.get("terminal_reason") is None:
            if not send(signal.SIGTERM):
                refresh()
                return report("exited", True)
            requested = True
        exited = wait_for_exit(wait_seconds) if wait_seconds is not None else exited_now()
        if not exited and force and wait_seconds is not None:
            current = refresh()
            if current is None:
                return report("force_refused_status_unavailable", exited_now(), refused=True)
            if not verify_target(current):
                return report("exited", True)
            if not send(signal.SIGKILL):
                refresh()
                return report("exited", True)
            forced = True
            exited = wait_for_exit(wait_seconds)
        refresh()
        return report(
            "forced_exit"
            if exited and forced
            else "force_requested_still_alive"
            if forced
            else "exited"
            if exited
            else "timed_out_still_alive"
            if wait_seconds is not None
            else "already_terminal"
            if not requested
            else "stop_requested",
            exited,
        )
    finally:
        os.close(fd)


def describe_schedule(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    phases: dict[str, Any] = {}
    schedule_warnings = []
    scheduler_enabled = training.get("use_scheduler", False)
    if not isinstance(scheduler_enabled, bool):
        raise ValueError("use_scheduler must be a boolean")
    if scheduler_enabled:
        scheduler_name = training.get("scheduler")
        if scheduler_name != "StepLR":
            raise ValueError(f"Unknown scheduler: {scheduler_name!r}")
        step = training.get("scheduler_step")
        gamma = training.get("scheduler_gamma")
        initial_lr = training.get("learning_rate")
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError("scheduler_step must be a positive integer")

        def finite_nonnegative(value: Any) -> bool:
            try:
                return not isinstance(value, bool) and math.isfinite(value) and value >= 0
            except (TypeError, ValueError):
                return False

        if not finite_nonnegative(gamma):
            raise ValueError("scheduler_gamma must be finite and nonnegative")
        if not finite_nonnegative(initial_lr):
            raise ValueError("learning_rate must be finite and nonnegative")
    for phase, default in (("phase_a", 100), ("phase_b", 1000)):
        epochs = training.get(phase + "_epochs", default)
        steps = training.get(phase + "_steps_per_epoch", training.get("steps_per_epoch", 10))
        if type(epochs) is not int or type(steps) is not int or min(epochs, steps) < 1:
            raise ValueError("Phase epochs and steps must be positive integers")
        phases[phase] = {"epochs": epochs, "steps_per_epoch": steps, "updates": epochs * steps}
        if scheduler_enabled:
            try:
                final_factor = gamma ** ((epochs - 1) // step)
                final_epoch_lr = initial_lr * final_factor
                after_phase_lr = initial_lr * gamma ** (epochs // step)
            except OverflowError as exc:
                raise ValueError("StepLR schedule produces non-finite learning rates") from exc
            if not math.isfinite(final_epoch_lr) or not math.isfinite(after_phase_lr):
                raise ValueError("StepLR schedule produces non-finite learning rates")
            phases[phase]["learning_rate"] = {
                "scheduler": "StepLR",
                "step_unit": "completed phase epoch",
                "reset_at_phase_start": True,
                "initial": initial_lr,
                "updates_between_decays": step * steps,
                "final_epoch": final_epoch_lr,
                "after_phase": after_phase_lr,
            }
            if final_factor < 1e-6:
                schedule_warnings.append(
                    f"{phase}: final-epoch LR is {final_epoch_lr:.6g} "
                    f"({final_factor:.6g} of initial LR). StepLR decays every "
                    f"{step * steps} mini-batch updates; verify the intended training horizon."
                )
    cadence = config.get("runtime", {}).get("validation_every", 1)
    if type(cadence) is not int or cadence < 1:
        raise ValueError("runtime.validation_every must be a positive integer")
    return {
        "phases": phases,
        "validation_every": cadence,
        "total_updates": sum(p["updates"] for p in phases.values()),
        "recovery_every": 1,
        "warnings": schedule_warnings,
    }


def _clean_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    return value


class RunSession:
    """Single writer, append-preserved attempts, cooperative signals and budgets."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        resume: bool = False,
        wall_seconds: float | None = None,
        budget_seconds: float | None = None,
        recover: bool = False,
        max_retries: int | None = None,
    ) -> None:
        self.config = config
        self.resume = resume
        self.recover = recover
        runtime = config.get("runtime", {})
        self.wall_seconds = (
            wall_seconds if wall_seconds is not None else runtime.get("wall_seconds")
        )
        self.budget_seconds = (
            budget_seconds if budget_seconds is not None else runtime.get("budget_seconds")
        )
        self.max_retries = max_retries if max_retries is not None else runtime.get("max_retries", 0)
        from dense_unet_3d.training.experiment import validate_experiment

        effective_config = copy.deepcopy(config)
        effective_config.setdefault("runtime", {})["wall_seconds"] = self.wall_seconds
        validate_experiment(effective_config)
        for value in (self.wall_seconds, self.budget_seconds):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Runtime limits must be positive finite seconds")
        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        for key in ("heartbeat_seconds", "stall_seconds"):
            value = runtime.get(key)
            if value is not None and (
                not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"runtime.{key} must be positive finite seconds")
        minimum = runtime.get("min_free_bytes", 0)
        if not isinstance(minimum, int) or minimum < 0:
            raise ValueError("runtime.min_free_bytes must be a nonnegative integer")
        if runtime.get("recovery_every", 1) != 1:
            raise ValueError("Only recovery_every=1 is supported")
        self.run_dir = Path(config["pathing"]["model_save_dir"]) / config["pathing"]["run_name"]
        self.checkpoint_path = self.run_dir / "recovery.pt"
        self._mutex = threading.RLock()
        self._stop = threading.Event()
        self._signal_requested = False
        self._handlers: dict[int, Any] = {}
        self.state: dict[str, Any] = {}
        self._entered = False
        self._background_error: BaseException | None = None
        self._retry_charged = False

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    @property
    def cumulative_seconds(self) -> float:
        return float(self._base + self.elapsed)

    def __enter__(self) -> RunSession:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "RunSession must be entered on the main thread for cooperative signals"
            )
        self._start = time.monotonic()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = (self.run_dir / ".owner.lock").open("a+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            raise RuntimeError("Run already has an active owner") from exc
        try:
            # Install before any durable owner publication, including resumed attempts.
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_signal)
            path = self.run_dir / "runtime.json"
            identity = config_identity(self.config)
            # Temporary files are never continuation points; only remove our known
            # temporary patterns after obtaining exclusive local ownership.
            for pattern in ("**/.tmp-*", "**/.checkpoint-*", "**/.previous-*"):
                for abandoned in self.run_dir.glob(pattern):
                    abandoned.unlink()
            try:
                revision = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                revision = "unknown"
            now = time.time()
            if path.exists():
                if not self.resume:
                    raise ValueError("Run already exists; use explicit resume")
                self.state = json.loads(path.read_text())
                required = {
                    "schema_version",
                    "config_identity",
                    "budget_seconds",
                    "max_retries",
                    "retries_used",
                    "terminal_reason",
                    "attempts",
                    "cumulative_seconds",
                    "started_at",
                    "base_seconds",
                    "run_id",
                    "owner",
                }
                if (
                    not isinstance(self.state, dict)
                    or not required.issubset(self.state)
                    or self.state["schema_version"] != 1
                    or not self.state["attempts"]
                    or not isinstance(self.state["cumulative_seconds"], (int, float))
                    or not math.isfinite(self.state["cumulative_seconds"])
                    or self.state["cumulative_seconds"] < 0
                ):
                    raise ValueError("Invalid durable runtime schema; refusing unsafe recovery")
                if (
                    self.state["terminal_reason"] is None
                    and self.state["owner"].get("host") != socket.gethostname()
                ):
                    raise ValueError("Cannot verify lost owner on another host")
                if self.state["config_identity"] != identity:
                    raise ValueError("Incompatible experiment configuration")
                if (
                    self.budget_seconds is not None
                    and self.budget_seconds != self.state["budget_seconds"]
                ):
                    raise ValueError("Cumulative budget cannot be changed on resume")
                self.budget_seconds = self.state["budget_seconds"]
                self.max_retries = self.state["max_retries"]
                reason = self.state["terminal_reason"]
                if reason in (None, "failed", "unknown"):
                    if not self.recover:
                        raise ValueError("Failed/lost attempt requires explicit --recover")
                    if self.state["retries_used"] >= self.max_retries:
                        raise ValueError("Persistent retry allowance exhausted")
                    self.state["retries_used"] += 1
                    self._retry_charged = True
                    if reason is None:
                        delta = now - self.state["started_at"]
                        if delta < 0:
                            raise ValueError(
                                "Clock moved backwards; cannot safely account lost runtime"
                            )
                        self.state["cumulative_seconds"] = max(
                            self.state["cumulative_seconds"], self.state["base_seconds"] + delta
                        )
                        self.state["attempts"][-1]["terminal_reason"] = "unknown"
                self._base = self.state["cumulative_seconds"]
            else:
                if self.resume:
                    raise ValueError(
                        "No durable run state; legacy checkpoints cannot exactly resume"
                    )
                if any(self.run_dir.glob("**/*.pt")):
                    raise ValueError("Existing checkpoint artifacts; choose a new run name")
                try:
                    revision = subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
                    ).strip()
                except (OSError, subprocess.CalledProcessError):
                    revision = "unknown"
                self._base = 0.0
                self.state = {
                    "schema_version": 1,
                    "run_id": uuid.uuid4().hex,
                    "config_identity": identity,
                    "resolved_config": self.config,
                    "source_revision": revision,
                    "budget_seconds": self.budget_seconds,
                    "max_retries": self.max_retries,
                    "retries_used": 0,
                    "attempts": [],
                }
            self.attempt_id = uuid.uuid4().hex
            self.state.update(
                owner=_process_identity(os.getpid()),
                started_at=now,
                progress_at=now,
                base_seconds=self._base,
                terminal_reason=None,
                attempt_id=self.attempt_id,
            )
            self.state["attempts"].append(
                {
                    "attempt_id": self.attempt_id,
                    "started_at": now,
                    "terminal_reason": None,
                    "source_revision": revision,
                }
            )
            self._entered = True
            self.event("attempt_started", resume=self.resume)
            self._thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._thread.start()
            return self
        except BaseException:
            try:
                if self._entered:
                    self.state["terminal_reason"] = "failed"
                    self.state["attempts"][-1].update(
                        terminal_reason="failed", ended_at=time.time()
                    )
                    self._persist()
            finally:
                for saved_sig, handler in self._handlers.items():
                    signal.signal(saved_sig, handler)
                self._handlers.clear()
                self._entered = False
                self._lock.close()
            raise

    def _handle_signal(self, _signum: int, _frame: Any) -> None:
        self._signal_requested = True

    def request_stop(self) -> None:
        self._signal_requested = True

    def stop_reason(self) -> str | None:
        if self._background_error is not None:
            raise RuntimeError("Background runtime monitoring failed") from self._background_error
        if self._signal_requested:
            return "user stopped"
        if (self.wall_seconds is not None and self.elapsed >= self.wall_seconds) or (
            self.budget_seconds is not None and self.cumulative_seconds >= self.budget_seconds
        ):
            return "budget exhausted"
        return None

    def _persist(self) -> None:
        self.state["cumulative_seconds"] = self.cumulative_seconds
        self.state["heartbeat_at"] = time.time()
        self.state["storage_free_bytes"] = shutil.disk_usage(self.run_dir).free
        atomic_json(self.run_dir / "runtime.json", _clean_json(self.state))

    def _heartbeat(self) -> None:
        interval = max(0.1, float(self.config.get("runtime", {}).get("heartbeat_seconds", 5)))
        while not self._stop.wait(interval):
            try:
                with self._mutex:
                    self._persist()
            except OSError as exc:
                self._background_error = exc
                return

    def charge_recovery_retry(self) -> None:
        if self._retry_charged:
            return
        if self.state["retries_used"] >= self.max_retries:
            raise ValueError("Persistent retry allowance exhausted")
        self.state["retries_used"] += 1
        self._retry_charged = True
        self.event("recovery_retry_charged")

    def check_storage(self) -> None:
        minimum = self.config.get("runtime", {}).get("min_free_bytes", 0)
        if shutil.disk_usage(self.run_dir).free < minimum:
            raise OSError("Free storage below runtime.min_free_bytes")

    def event(self, kind: str, **fields: Any) -> None:
        with self._mutex:
            now = time.time()
            record = _clean_json(
                {
                    "event": kind,
                    "time": now,
                    "run_id": self.state["run_id"],
                    "attempt_id": self.attempt_id,
                    "cumulative_seconds": self.cumulative_seconds,
                    **fields,
                }
            )
            with (self.run_dir / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            if kind == "checkpoint":
                self.state["last_committed_checkpoint"] = {
                    "path": str(self.checkpoint_path),
                    "committed_at": now,
                    "attempt_id": self.attempt_id,
                    **fields,
                }
            self.state["progress_at"] = now
            self.state["progress"] = record
            self._persist()

    def finish(self, reason: str) -> None:
        with self._mutex:
            self.state["terminal_reason"] = reason
            self.state["attempts"][-1].update(terminal_reason=reason, ended_at=time.time())
            self.event("attempt_finished", terminal_reason=reason)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        try:
            if exc is not None:
                self.event("failure", error_type=type(exc).__name__, error=str(exc))
                self.finish("failed")
            elif self.state["terminal_reason"] is None:
                self.finish(self.stop_reason() or "completed")
        finally:
            for sig, handler in self._handlers.items():
                signal.signal(sig, handler)
            self._entered = False
            self._lock.close()
