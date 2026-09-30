"""Epoch-boundary continuation for standard single-process PyTorch loaders."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Subset,
    TensorDataset,
)
from tqdm import tqdm

from dense_unet_3d.training.runtime import (
    RunSession,
    atomic_checkpoint,
    config_identity,
    describe_schedule,
)

SCHEMA_VERSION = 1


def _transform_identity(transform: Any) -> Any:
    """Fingerprint supported preprocessing; unknown callables cannot exactly resume.

    These repository transforms use only the captured global NumPy RNG. Exact
    types and instance fields are checked to exclude custom subclasses, callbacks
    and hidden per-transform random generators from the continuation contract.
    """
    from torchvision.transforms import Compose

    from dense_unet_3d.dataset.transforms.ClampValues import ClampValues
    from dense_unet_3d.dataset.transforms.RandomHorizontalFlip import RandomHorizontalFlip
    from dense_unet_3d.dataset.transforms.ReshapeTensor import ReshapeTensor
    from dense_unet_3d.dataset.transforms.Resize import Resize
    from dense_unet_3d.dataset.transforms.ScaleAndPadOrCrop import ScaleAndPadOrCrop

    if transform is None:
        return None
    if type(transform) is Compose:
        if set(vars(transform)) != {"transforms"}:
            raise ValueError("Unsupported preprocessing state in Compose")
        return {
            "type": "Compose",
            "transforms": [_transform_identity(child) for child in transform.transforms],
        }
    fields = {
        ClampValues: {"voxel_range"},
        RandomHorizontalFlip: {"p"},
        ReshapeTensor: set(),
        Resize: {"size", "mode"},
        ScaleAndPadOrCrop: {"scale_lo", "scale_hi"},
    }
    expected = fields.get(type(transform))
    if expected is None or set(vars(transform)) != expected:
        raise ValueError(
            "Exact resume supports only repository preprocessing transforms "
            "and torchvision Compose; custom callables/state are unsupported"
        )
    parameters = {key: getattr(transform, key) for key in sorted(expected)}
    try:
        # Reject hidden generators and other non-serializable parameter objects.
        parameters = json.loads(json.dumps(parameters, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Unsupported preprocessing parameter state") from exc
    return {
        "type": type(transform).__module__ + "." + type(transform).__qualname__,
        "parameters": parameters,
    }


def _dataset_identity(dataset: Any) -> Any:
    if type(dataset) is Subset:
        return {
            "subset": list(map(int, dataset.indices)),
            "dataset": _dataset_identity(dataset.dataset),
        }
    if type(dataset) is TensorDataset:
        return {
            "tensors": [
                {
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "sha256": hashlib.sha256(
                        t.detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest(),
                }
                for t in dataset.tensors
            ]
        }
    from dense_unet_3d.dataset.LITSDataset import LITSDataset

    if type(dataset) is LITSDataset:
        preprocessing = {
            name: _transform_identity(getattr(dataset, name))
            for name in ("transform", "mask_transform", "paired_transform")
        }
        files = []
        for pair in zip(dataset.volume_img_paths, dataset.segmentation_img_paths, strict=True):
            record = []
            for name in pair:
                digest = hashlib.sha256()
                with open(name, "rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                record.append({"path": str(Path(name).resolve()), "sha256": digest.hexdigest()})
            files.append(record)
        return {
            "files": files,
            "detect_tumors": dataset.detect_tumors,
            "crop_to_liver": dataset.crop_to_liver,
            "preprocessing": preprocessing,
        }
    raise ValueError("Exact resume supports only TensorDataset, LITSDataset and Subset datasets")


def _loader_identity(loader: DataLoader | None) -> Any:
    if loader is None:
        return None
    if (
        type(loader) is not DataLoader
        or loader.num_workers != 0
        or type(loader.sampler) not in (RandomSampler, SequentialSampler)
        or type(loader.batch_sampler) is not BatchSampler
    ):
        raise ValueError(
            "Exact resume requires standard DataLoader, num_workers=0 and standard samplers"
        )
    from torch.utils.data._utils.collate import default_collate

    if loader.collate_fn is not default_collate:
        raise ValueError("Exact resume requires default_collate")
    return {
        "dataset": _dataset_identity(loader.dataset),
        "batch_size": loader.batch_size,
        "drop_last": loader.drop_last,
        "sampler": type(loader.sampler).__name__,
        "replacement": getattr(loader.sampler, "replacement", None),
        "num_samples": getattr(loader.sampler, "num_samples", None),
    }


def _generator_topology(loaders: list[DataLoader | None]) -> list[list[int | str | None]]:
    """Canonical alias graph, independent of process-local object addresses.

    Shared generators advance one stream; independent generators with identical
    initial state do not. Include all loader/sampler references across both phases
    and validation, plus explicit aliases of the global CPU generator.
    """
    aliases: dict[int, int] = {}
    topology: list[list[int | str | None]] = []
    for loader in loaders:
        references: list[int | str | None] = []
        for obj in (loader, getattr(loader, "sampler", None)):
            generator = getattr(obj, "generator", None)
            if generator is None:
                references.append(None)
            elif generator is torch.default_generator:
                references.append("global_cpu")
            else:
                references.append(aliases.setdefault(id(generator), len(aliases)))
        topology.append(references)
    return topology


def _generator_state(obj: Any) -> Any:
    generator = getattr(obj, "generator", None)
    return generator.get_state() if generator is not None else None


def _rng(loaders: list[DataLoader | None]) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        "loaders": [
            [_generator_state(obj) for obj in (loader, getattr(loader, "sampler", None))]
            for loader in loaders
        ],
    }


def _restore_rng(state: dict[str, Any], loaders: list[DataLoader | None]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology changed")
        torch.cuda.set_rng_state_all(state["cuda"])
    for loader, states in zip(loaders, state["loaders"], strict=True):
        for obj, saved in zip((loader, getattr(loader, "sampler", None)), states, strict=True):
            generator = getattr(obj, "generator", None)
            if (generator is None) != (saved is None):
                raise ValueError("DataLoader generator configuration changed")
            if generator is not None:
                generator.set_state(saved)


def _snapshot(
    model: Any, optimizer: Any, scheduler: Any, epoch: int, metrics: Any
) -> dict[str, Any]:
    return {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": copy.deepcopy(scheduler.state_dict())
        if scheduler is not None
        else None,
        "epoch": epoch,
        "metrics": metrics,
    }


def _validate_checkpoint(state: Any, identity: str, run_id: str, schedule: dict[str, Any]) -> None:
    keys = {
        "schema_version",
        "identity",
        "run_id",
        "phase",
        "epoch",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng",
        "history",
        "best",
        "cumulative_seconds",
    }
    if (
        not isinstance(state, dict)
        or not keys.issubset(state)
        or state["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("Legacy/incomplete recovery schema; exact resume refused")
    if state["identity"] != identity or state["run_id"] != run_id:
        raise ValueError("Incompatible architecture, config, dataset/split or run identity")
    if (
        not isinstance(state["rng"], dict)
        or set(state["rng"]) != {"python", "numpy", "torch", "cuda", "loaders"}
        or not isinstance(state["rng"]["loaders"], list)
        or len(state["rng"]["loaders"]) != 4
        or not isinstance(state["history"], dict)
        or set(state["history"]) != {"phase_a", "phase_b"}
        or not isinstance(state["best"], dict)
        or set(state["best"]) != {"phase_a", "phase_b"}
        or not isinstance(state["cumulative_seconds"], (int, float))
        or not np.isfinite(state["cumulative_seconds"])
        or state["cumulative_seconds"] < 0
    ):
        raise ValueError("Malformed continuation RNG/history/best/runtime state")
    try:
        random.Random().setstate(state["rng"]["python"])
        np.random.RandomState().set_state(state["rng"]["numpy"])
        torch.Generator().set_state(state["rng"]["torch"])
        for generators in state["rng"]["loaders"]:
            if not isinstance(generators, list) or len(generators) != 2:
                raise ValueError("Malformed loader generator state")
            for generator in generators:
                if generator is not None:
                    torch.Generator().set_state(generator)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError("Malformed continuation RNG state") from exc
    for phase_name in ("phase_a", "phase_b"):
        history = state["history"][phase_name]
        best = state["best"][phase_name]
        if (
            not isinstance(history, list)
            or not all(np.isfinite(v) for v in history)
            or len(history) > schedule["phases"][phase_name]["epochs"]
        ):
            raise ValueError("Malformed continuation history")
        if best is not None:
            if (
                not isinstance(best, dict)
                or not {"score", "epoch", "metrics", "checkpoint"}.issubset(best)
                or not np.isfinite(best["score"])
                or not isinstance(best["epoch"], int)
                or not 1 <= best["epoch"] <= len(history)
                or not isinstance(best["checkpoint"], dict)
                or not {
                    "model_state_dict",
                    "optimizer_state_dict",
                    "scheduler_state_dict",
                    "epoch",
                    "metrics",
                }.issubset(best["checkpoint"])
                or best["checkpoint"]["epoch"] != best["epoch"]
            ):
                raise ValueError("Malformed best-selection continuation state")
    phase = state["phase"]
    if phase not in ("phase_a", "phase_b", "completed"):
        raise ValueError("Invalid recovery phase")
    phase_key = "phase_b" if phase == "completed" else phase
    epochs = schedule["phases"][phase_key]["epochs"]
    if not isinstance(state["epoch"], int) or not 0 <= state["epoch"] <= epochs:
        raise ValueError("Invalid recovery epoch")
    expected = state["epoch"] * schedule["phases"][phase_key]["steps_per_epoch"]
    if phase != "phase_a":
        expected += schedule["phases"]["phase_a"]["updates"]
    if (
        len(state["history"][phase_key]) != state["epoch"]
        or (phase == "phase_a" and state["history"]["phase_b"])
        or (
            phase != "phase_a"
            and len(state["history"]["phase_a"]) != schedule["phases"]["phase_a"]["epochs"]
        )
        or (phase == "completed" and state["epoch"] != epochs)
    ):
        raise ValueError("Invalid recovery history/phase progress")
    if expected != state["global_step"]:
        raise ValueError("Invalid recovery global step")


def run_recoverable(
    config: dict[str, Any],
    model: Any,
    device: torch.device,
    loaders: list[DataLoader | None],
    session: RunSession,
) -> dict[str, Any]:
    from dense_unet_3d.training import cascaded_driver as driver

    if not session._entered or config_identity(config) != session.state["config_identity"]:
        raise ValueError("Training requires an entered session for this configuration")
    schedule = describe_schedule(config)
    identities = [_loader_identity(loader) for loader in loaders]
    architecture = {k: [list(v.shape), str(v.dtype)] for k, v in model.state_dict().items()}
    identity = hashlib.sha256(
        json.dumps(
            {
                "config": config_identity(config),
                "data": identities,
                "generator_topology": _generator_topology(loaders),
                "architecture": architecture,
                "model_class": type(model).__module__ + "." + type(model).__qualname__,
                "device_type": device.type,
                "torch_version": str(torch.__version__),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    model.to(device)
    optimizer = driver.get_optimizer(model, config)
    scheduler = driver.get_scheduler(optimizer, config)
    if session.resume:
        path = session.checkpoint_path
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            if not session.recover:
                raise ValueError(
                    "Recovery checkpoint unreadable; explicit recovery required"
                ) from exc
            session.charge_recovery_retry()
            previous = path.with_name("recovery.previous.pt")
            state = torch.load(previous, map_location="cpu", weights_only=False)
            _validate_checkpoint(state, identity, session.state["run_id"], schedule)
            if path.exists():
                path.rename(path.with_name("recovery.corrupt." + session.attempt_id + ".pt"))
            session.event("recovered_previous_checkpoint", path=str(previous))
        _validate_checkpoint(state, identity, session.state["run_id"], schedule)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if (scheduler is None) != (state["scheduler_state_dict"] is None):
            raise ValueError("Incompatible scheduler state")
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        _restore_rng(state["rng"], loaders)
    else:
        state = {
            "schema_version": SCHEMA_VERSION,
            "identity": identity,
            "run_id": session.state["run_id"],
            "phase": "phase_a",
            "epoch": 0,
            "global_step": 0,
            "history": {"phase_a": [], "phase_b": []},
            "best": {"phase_a": None, "phase_b": None},
            "phase_b_loaded_phase_a_state_dict": None,
        }

    def checkpoint() -> None:
        session.check_storage()
        state.update(
            _snapshot(model, optimizer, scheduler, state["epoch"], state.get("metrics", {}))
        )
        state["rng"] = _rng(loaders)
        state["cumulative_seconds"] = session.cumulative_seconds
        atomic_checkpoint(session.checkpoint_path, state, retain_previous=True)
        session.event(
            "checkpoint",
            phase=state["phase"],
            epoch=state["epoch"],
            global_step=state["global_step"],
        )

    def export_selected() -> None:
        # Recovery is the transaction authority. Reconcile exports after interrupted writes.
        for phase in ("phase_a", "phase_b"):
            best = state["best"][phase]
            if best is not None:
                atomic_checkpoint(session.run_dir / phase / "best.pt", best["checkpoint"])

    if not session.resume:
        checkpoint()
    else:
        export_selected()
    timings: dict[str, list[float]] = {"phase_a": [], "phase_b": []}
    reason = session.stop_reason()
    while state["phase"] != "completed" and reason is None:
        phase = state["phase"]
        phase_schedule = schedule["phases"][phase]
        if state["epoch"] == phase_schedule["epochs"]:
            if state["best"][phase] is None:
                raise ValueError(f"{phase} produced no finite validation selection score")
            atomic_checkpoint(
                session.run_dir / phase / "last.pt",
                _snapshot(model, optimizer, scheduler, state["epoch"], state["metrics"]),
            )
            if phase == "phase_a":
                weights = state["best"][phase]["checkpoint"]["model_state_dict"]
                model.load_state_dict(weights)
                state["phase_b_loaded_phase_a_state_dict"] = copy.deepcopy(weights)
                optimizer = driver.get_optimizer(model, config)
                scheduler = driver.get_scheduler(optimizer, config)
                state.update(phase="phase_b", epoch=0, metrics={})
            else:
                state["phase"] = "completed"
            checkpoint()
            session.event(
                "phase_transition", phase=state["phase"], global_step=state["global_step"]
            )
            reason = session.stop_reason()
            continue
        epoch = state["epoch"] + 1
        train_loader: Any
        val_loader: Any
        train_loader, val_loader = loaders[:2] if phase == "phase_a" else loaders[2:]
        if phase == "phase_a":
            train_loader = driver._LiverOnlyLoader(train_loader)
            val_loader = driver._LiverOnlyLoader(val_loader) if val_loader is not None else None
        start = time.monotonic()
        session.event("training", phase=phase, epoch=epoch, global_step=state["global_step"])
        loss = driver._run_epoch(
            config=config,
            model=model,
            device=device,
            loader=train_loader,
            optimizer=optimizer,
            steps_per_epoch=phase_schedule["steps_per_epoch"],
        )
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        training_seconds = time.monotonic() - start
        if scheduler is not None:
            scheduler.step()
        metrics = {"train_loss": loss}
        validation_seconds = 0.0
        selection = None
        if val_loader is None:
            selection = 0.0
        elif epoch % schedule["validation_every"] == 0 or epoch == phase_schedule["epochs"]:
            from dense_unet_3d.evaluation.evaluate import evaluate

            session.event("validation", phase=phase, epoch=epoch, global_step=state["global_step"])
            start_validation = time.monotonic()
            metrics.update(evaluate(model, device, val_loader))
            validation_seconds = time.monotonic() - start_validation
            components = [metrics.get("liver_per_case", float("nan"))]
            if phase == "phase_b":
                components.append(metrics.get("tumor_per_case", float("nan")))
            if any(np.isinf(value) for value in components):
                raise FloatingPointError("Infinite validation selection metric")
            finite = [value for value in components if np.isfinite(value)]
            selection = float(np.mean(finite)) if finite else None
            if selection is None:
                session.event("undefined_validation_selection", phase=phase, epoch=epoch)
        best = state["best"][phase]
        improved = selection is not None and (best is None or selection > best["score"])
        if improved:
            state["best"][phase] = {
                "score": selection,
                "epoch": epoch,
                "metrics": metrics,
                "checkpoint": _snapshot(model, optimizer, scheduler, epoch, metrics),
            }
        state["epoch"] = epoch
        state["global_step"] += phase_schedule["steps_per_epoch"]
        state["history"][phase].append(loss)
        state["metrics"] = metrics
        checkpoint()
        if improved:
            atomic_checkpoint(
                session.run_dir / phase / "best.pt", state["best"][phase]["checkpoint"]
            )
        elapsed = time.monotonic() - start
        timings[phase].append(elapsed)
        measured = timings[phase]
        remaining = phase_schedule["epochs"] - epoch
        eta = {
            "phase_seconds": float(np.mean(measured)) * remaining,
            "range_seconds": [min(measured) * remaining, max(measured) * remaining],
            "samples": len(measured),
            "other_phase": "unmeasured",
        }
        session.event(
            "epoch_completed",
            phase=phase,
            epoch=epoch,
            global_step=state["global_step"],
            metrics=metrics,
            training_seconds=training_seconds,
            validation_seconds=validation_seconds,
            epoch_seconds=elapsed,
            eta=eta,
        )
        tqdm.write(
            f"{phase} epoch {epoch}/{phase_schedule['epochs']} step={state['global_step']} "
            f"loss={loss:.5g} train={training_seconds:.2f}s val={validation_seconds:.2f}s "
            f"phase ETA={eta['phase_seconds']:.1f}s (n={len(measured)}; range={eta['range_seconds']})"
        )
        reason = session.stop_reason()
    terminal = reason or "completed"
    session.event(
        "training_finished",
        terminal_reason=terminal,
        phase=state["phase"],
        epoch=state["epoch"],
        global_step=state["global_step"],
    )
    result: dict[str, Any] = {
        "terminal_reason": terminal,
        "global_step": state["global_step"],
        "phase_b_loaded_phase_a_state_dict": state.get("phase_b_loaded_phase_a_state_dict"),
    }
    for phase in ("phase_a", "phase_b"):
        best = state["best"][phase]
        result[phase] = {
            "epoch_losses": state["history"][phase],
            "best_epoch": best["epoch"] if best else None,
            "best_metrics": best["metrics"] if best else {},
        }
    return result
