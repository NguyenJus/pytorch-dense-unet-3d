"""Deterministic CPU subprocess coverage of startup and blocked cooperative stop."""

from __future__ import annotations

import fcntl
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from dense_unet_3d import cli
from dense_unet_3d.training import runtime


def config(tmp_path):
    return {"pathing": {"model_save_dir": str(tmp_path), "run_name": "run"}, "training": {}}


@pytest.fixture
def child_run(tmp_path):
    children = []

    def launch(body):
        source = f"""
import json, os, signal, sys, torch
from dense_unet_3d.training import runtime
config = {config(tmp_path)!r}
{body}
"""
        child = subprocess.Popen(
            [sys.executable, "-u", "-c", source],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
        )
        children.append(child)
        assert select.select([child.stdout], [], [], 20)[0], "child readiness timed out"
        assert child.stdout.readline().strip() == "ready", (
            child.stderr.read() if child.poll() is not None else ""
        )
        return child

    yield launch
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
        for stream in (child.stdin, child.stdout, child.stderr):
            stream.close()


def test_stop_at_owner_publication_is_cooperative(tmp_path, child_run):
    child = child_run("""
original = runtime.atomic_json
def publish(path, value):
    original(path, value)
    if value.get('progress', {}).get('event') == 'attempt_started':
        print('ready', flush=True)
        sys.stdin.readline()
runtime.atomic_json = publish
with runtime.RunSession(config) as session:
    assert session.stop_reason() == 'user stopped'
""")
    result = runtime.request_stop(tmp_path / "run")
    assert result["outcome"] == "stop_requested"
    child.stdin.write("continue\n")
    child.stdin.flush()
    assert child.wait(timeout=10) == 0
    assert runtime.read_status(tmp_path / "run")["terminal_reason"] == "user stopped"


def test_blocked_stop_wait_and_explicit_force(tmp_path, child_run):
    child = child_run("""
with runtime.RunSession(config) as session:
    runtime.atomic_checkpoint(session.checkpoint_path, {'committed': True})
    session.event('checkpoint', phase='phase_a', epoch=0, global_step=0)
    print('ready', flush=True)
    sys.stdin.readline()  # SIGTERM only sets a flag; this operation remains blocked.
""")
    start = time.monotonic()
    result = runtime.request_stop(tmp_path / "run", wait_seconds=0.05)
    assert time.monotonic() - start < 2
    assert result["outcome"] == "timed_out_still_alive"
    assert child.poll() is None
    assert result["clean_exit"] is False
    assert result["last_committed_checkpoint"]["global_step"] == 0
    result = runtime.request_stop(tmp_path / "run", wait_seconds=0.2, force=True)
    assert result["outcome"] == "forced_exit"
    assert result["clean_exit"] is False
    assert result["terminal_reason"] == "unknown"
    assert child.wait(timeout=10) == -signal.SIGKILL
    assert runtime.torch.load(tmp_path / "run/recovery.pt", weights_only=False)["committed"]


def test_wait_observes_clean_exit(tmp_path, child_run):
    child = child_run("""
with runtime.RunSession(config) as session:
    print('ready', flush=True)
    while session.stop_reason() is None:
        signal.pause()
""")
    result = runtime.request_stop(tmp_path / "run", wait_seconds=5, force=True)
    assert result["outcome"] == "exited"
    assert result["clean_exit"] is True
    assert result["force_requested"] is False
    assert child.wait(timeout=10) == 0


def test_force_rechecks_attempt_ownership(tmp_path, child_run, monkeypatch):
    child = child_run("""
with runtime.RunSession(config) as session:
    print('ready', flush=True)
    sys.stdin.readline()
""")
    original = runtime.read_status
    calls = 0

    def replaced(path):
        nonlocal calls
        calls += 1
        state = original(path)
        if calls >= 3:
            state["attempt_id"] = "replacement"
        return state

    monkeypatch.setattr(runtime, "read_status", replaced)
    with pytest.raises(RuntimeError, match="Owner changed"):
        runtime.request_stop(tmp_path / "run", wait_seconds=0.05, force=True)
    assert child.poll() is None


def test_non_main_thread_rejected_before_publication(tmp_path):
    errors = []

    def enter():
        try:
            with runtime.RunSession(config(tmp_path)):
                pytest.fail("must reject non-main-thread entry")
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=enter)
    thread.start()
    thread.join(timeout=5)
    assert errors and "main thread" in errors[0]
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "failure", ["validation", "publication", "thread_start", "handler_install"]
)
def test_entry_failure_restores_handlers_and_lock(tmp_path, monkeypatch, failure):
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    session = runtime.RunSession(config(tmp_path))
    if failure == "validation":
        session.resume = True
    elif failure == "publication":
        original = session.event

        def broken(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("publication failed")

        monkeypatch.setattr(session, "event", broken)
    elif failure == "thread_start":
        monkeypatch.setattr(
            threading.Thread,
            "start",
            lambda _self: (_ for _ in ()).throw(RuntimeError("thread failed")),
        )
    else:
        original_signal = signal.signal

        def broken_signal(sig, handler):
            if sig == signal.SIGTERM and handler == session._handle_signal:
                raise RuntimeError("handler failed")
            return original_signal(sig, handler)

        monkeypatch.setattr(signal, "signal", broken_signal)
    with pytest.raises((RuntimeError, ValueError)):
        session.__enter__()
    assert {sig: signal.getsignal(sig) for sig in before} == before
    with (tmp_path / "run/.owner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if failure in {"publication", "thread_start"}:
        assert (
            json.loads((tmp_path / "run/runtime.json").read_text())["terminal_reason"] == "failed"
        )


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan"), True])
def test_invalid_wait_refused_before_signaling(tmp_path, seconds):
    with pytest.raises(ValueError, match="positive finite"):
        runtime.request_stop(tmp_path, wait_seconds=seconds)


def test_force_requires_bounded_wait(tmp_path):
    with pytest.raises(ValueError, match="explicit bounded wait"):
        runtime.request_stop(tmp_path, force=True)


@pytest.mark.parametrize(
    "outcome,force,clean,exited,code",
    [
        ("stop_requested", False, False, False, 0),
        ("exited", False, True, True, 0),
        ("timed_out_still_alive", False, False, False, 2),
        ("exited", False, False, True, 2),
        ("forced_exit", True, False, True, 3),
        ("force_requested_still_alive", True, False, False, 3),
        ("stop_refused_status_unavailable", False, False, False, 2),
        ("force_refused_status_unavailable", False, False, False, 2),
    ],
)
def test_cli_stop_outcomes(monkeypatch, capsys, outcome, force, clean, exited, code):
    result = {
        "outcome": outcome,
        "force_requested": force,
        "clean_exit": clean,
        "process_exited": exited,
    }
    monkeypatch.setattr(runtime, "request_stop", lambda *a, **kw: result)
    args = cli._build_parser().parse_args(["stop", "--run-dir", "run", "--wait-seconds", "1"])
    if code:
        with pytest.raises(SystemExit) as error:
            cli._cmd_stop(args)
        assert error.value.code == code
    else:
        cli._cmd_stop(args)
    assert json.loads(capsys.readouterr().out) == result


def test_force_still_alive_is_reported_without_success(tmp_path, child_run, monkeypatch):
    child = child_run("""
with runtime.RunSession(config) as session:
    print('ready', flush=True)
    sys.stdin.readline()
""")
    sent = []
    original = runtime._pidfd_send_signal

    def signal_without_kill(fd, sig):
        sent.append(sig)
        if sig != signal.SIGKILL:
            original(fd, sig)

    monkeypatch.setattr(runtime, "_pidfd_send_signal", signal_without_kill)
    result = runtime.request_stop(tmp_path / "run", wait_seconds=0.05, force=True)
    assert sent == [signal.SIGTERM, signal.SIGKILL]
    assert result["outcome"] == "force_requested_still_alive"
    assert result["process_exited"] is False
    assert result["clean_exit"] is False
    assert child.poll() is None


@pytest.mark.parametrize("change", ["start_ticks", "boot_id", "host"])
def test_force_refuses_stale_process_identity(tmp_path, monkeypatch, change):
    owner = runtime._process_identity(os.getpid())
    owner[change] = "stale"
    runtime.atomic_json(tmp_path / "runtime.json", {"owner": owner, "terminal_reason": None})
    monkeypatch.setattr(runtime, "_pidfd_open", lambda _pid: pytest.fail("must not open stale PID"))
    with pytest.raises(RuntimeError, match="no verified live owner"):
        runtime.request_stop(tmp_path, wait_seconds=0.05, force=True)


@pytest.mark.parametrize("force", [False, True])
def test_post_request_blocked_status_is_bounded_and_uses_cached_checkpoint(
    tmp_path, child_run, monkeypatch, force
):
    child = child_run("""
with runtime.RunSession(config) as session:
    runtime.atomic_checkpoint(session.checkpoint_path, {'committed': True})
    session.event('checkpoint', phase='phase_a', epoch=0, global_step=0)
    print('ready', flush=True)
    sys.stdin.readline()
""")
    original = runtime.read_status
    blocked = threading.Event()
    release = threading.Event()
    calls = 0
    sent = []
    original_signal = runtime._pidfd_send_signal

    def block_status(path):
        nonlocal calls
        calls += 1
        if calls >= 3:
            blocked.set()
            release.wait(10)
        return original(path)

    def track_signal(fd, sig):
        sent.append(sig)
        original_signal(fd, sig)

    monkeypatch.setattr(runtime, "read_status", block_status)
    monkeypatch.setattr(runtime, "_pidfd_send_signal", track_signal)
    try:
        started = time.monotonic()
        result = runtime.request_stop(tmp_path / "run", wait_seconds=0.05, force=force)
        assert time.monotonic() - started < 1
        assert blocked.is_set()
        assert result["outcome"] == (
            "force_refused_status_unavailable" if force else "timed_out_still_alive"
        )
        assert result["status_available"] is False
        assert result["status_error"] == "status lookup timed out"
        assert result["checkpoint_source"] == "cached_status"
        assert result["last_committed_checkpoint"]["global_step"] == 0
        assert result["force_refused"] is force
        assert result["clean_exit"] is False
        assert sent == [signal.SIGTERM]
        assert child.poll() is None
    finally:
        release.set()


@pytest.mark.parametrize("stage", ["open", "identity", "term", "kill"])
def test_exit_races_report_observed_unclean_exit(tmp_path, child_run, monkeypatch, stage):
    child = child_run("""
with runtime.RunSession(config) as session:
    print('ready', flush=True)
    sys.stdin.readline()
""")
    if stage == "open":
        original_open = runtime._pidfd_open

        def open_after_exit(pid):
            child.kill()
            child.wait(timeout=10)
            return original_open(pid)

        monkeypatch.setattr(runtime, "_pidfd_open", open_after_exit)
    elif stage == "identity":
        original_identity = runtime._process_identity
        main_reads = 0

        def identity_after_exit(pid):
            nonlocal main_reads
            if threading.current_thread() is threading.main_thread():
                main_reads += 1
            if (
                pid == child.pid
                and main_reads > 1
                and threading.current_thread() is threading.main_thread()
            ):
                child.kill()
                child.wait(timeout=10)
            return original_identity(pid)

        monkeypatch.setattr(runtime, "_process_identity", identity_after_exit)
    else:
        original_send = runtime._pidfd_send_signal

        def send_after_exit(fd, sig):
            if sig == (signal.SIGTERM if stage == "term" else signal.SIGKILL):
                child.kill()
                child.wait(timeout=10)
            return original_send(fd, sig)

        monkeypatch.setattr(runtime, "_pidfd_send_signal", send_after_exit)
    result = runtime.request_stop(tmp_path / "run", wait_seconds=0.05, force=stage == "kill")
    assert result["outcome"] == "exited"
    assert result["process_exited"] is True
    assert result["force_requested"] is False
    assert result["clean_exit"] is False
    assert result["terminal_reason"] == "unknown"
    assert child.wait(timeout=10) == -signal.SIGKILL


def test_unavailable_status_before_signal_refuses_request(tmp_path, monkeypatch):
    runtime.atomic_json(
        tmp_path / "runtime.json",
        {"owner": runtime._process_identity(os.getpid()), "terminal_reason": None},
    )
    original = runtime.read_status
    release = threading.Event()
    calls = 0

    def block_recheck(path):
        nonlocal calls
        calls += 1
        if calls >= 2:
            release.wait(10)
        return original(path)

    monkeypatch.setattr(runtime, "read_status", block_recheck)
    monkeypatch.setattr(runtime, "_pidfd_send_signal", lambda *_: pytest.fail("must refuse signal"))
    try:
        result = runtime.request_stop(tmp_path, wait_seconds=0.05, force=True)
        assert result["outcome"] == "stop_refused_status_unavailable"
        assert result["stop_requested"] is False
        assert result["process_exited"] is False
        assert result["status_available"] is False
        assert result["status_error"] == "status lookup timed out"
    finally:
        release.set()
