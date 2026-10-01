"""Console entry point for dense-unet-3d.

Subcommands
-----------
train
    Run the cascaded 2-phase training.  Accepts ``--config <path>``
    (required) and ``--dry-run`` (uses synthetic data, skips real NIfTI loading).

eval
    Evaluate a checkpoint on the validation split.  Prints Dice metrics.
    Requires ``--config <path>`` and ``--checkpoint <path>``.

predict
    Run inference on a NIfTI volume and write a NIfTI segmentation.
    Requires ``--config <path>``, ``--checkpoint <path>``, ``--input <path>``,
    and ``--output <path>``.

No hardcoded cwd dependence — all paths come from the CLI flags or the
config file supplied via ``--config``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML config from *config_path* (absolute or relative to cwd)."""
    abs_path = os.path.abspath(config_path)
    if not os.path.isfile(abs_path):
        sys.exit(f"Error: config file not found: {abs_path}")
    with open(abs_path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def _device_from_config(config: dict[str, Any]) -> torch.device:
    """Resolve the compute device from config, honouring the CPU-first rule."""
    gpu_cfg = config.get("gpu", {})
    use_gpu: bool = bool(gpu_cfg.get("use_gpu", False))
    gpu_name: str = str(gpu_cfg.get("gpu_name", "cpu"))
    if use_gpu and torch.cuda.is_available():
        return torch.device(gpu_name)
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Tiny synthetic loaders for --dry-run
# ---------------------------------------------------------------------------


def _make_dry_run_loader(
    batch_size: int = 2,
    d: int = 4,
    h: int = 8,
    w: int = 8,
    *,
    detect_tumors: bool = True,
) -> Any:
    """Return a DataLoader with synthetic tensors (CPU, no real NIfTI needed)."""
    from torch.utils.data import DataLoader, TensorDataset

    generator = torch.Generator().manual_seed(0)
    volumes = torch.randn(batch_size, 1, d, h, w, generator=generator)
    labels = torch.randint(0, 3, (batch_size, 1, d, h, w), generator=generator)
    if not detect_tumors:
        labels = labels.clamp(max=1)
    ds = TensorDataset(volumes, labels)
    return DataLoader(ds, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


class _TinyStub(nn.Module):
    """Test-stub model: single Conv3d(1,3,1) to match test checkpoint layout."""

    def __init__(self, conv: nn.Conv3d) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # type: ignore[no-any-return]


def _load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    config: dict[str, Any] | None = None,
    *,
    allow_legacy_preprocessing: bool = False,
) -> nn.Module:
    """Load the full model from a checkpoint.

    Tries to import ``DenseUNet3d``; falls back to a minimal Conv3d wrapper when
    the checkpoint was saved from a test-stub model (e.g. test helpers).  The
    fallback is transparent to callers.
    """
    ckpt: dict[str, Any] = torch.load(checkpoint_path, map_location=device, weights_only=False)
    from dense_unet_3d.training.experiment import configure_execution, preprocessing_identity

    if (
        config is not None
        and config.get("execution") is not None
        and ckpt.get("execution") is not None
    ):
        if config["execution"] != ckpt["execution"]:
            raise ValueError("Checkpoint execution configuration mismatch")
    configure_execution({"execution": ckpt.get("execution", (config or {}).get("execution"))})
    state = ckpt["model_state_dict"]

    def validate_preprocessing() -> None:
        if config is None:
            return
        expected_identity = preprocessing_identity(config)
        actual_identity = ckpt.get("preprocessing_identity")
        checkpoint_dataset = ckpt.get("dataset_config")
        if actual_identity is not None and not isinstance(checkpoint_dataset, dict):
            raise ValueError("Checkpoint has incomplete preprocessing metadata")
        if isinstance(checkpoint_dataset, dict):
            configured_dataset = config.get("dataset", {})
            for key, default in (
                ("sampling", "whole_volume"),
                ("resize_img", True),
                ("resize_dims", {"D": 12, "H": 224, "W": 224}),
                ("clamp_hu", True),
                ("clamp_hu_range", {"min": -200, "max": 250}),
            ):
                if checkpoint_dataset.get(key, default) != configured_dataset.get(key, default):
                    raise ValueError(f"Checkpoint preprocessing mismatch: {key}")
        if actual_identity is None:
            if not allow_legacy_preprocessing:
                raise ValueError(
                    "Checkpoint lacks preprocessing identity; pass "
                    "--allow-legacy-preprocessing only after verifying its historical pipeline"
                )
            warnings.warn(
                "LEGACY PREPROCESSING OVERRIDE: checkpoint sampling semantics are unknown; "
                "evaluation or prediction may not be comparable to the training pipeline.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif actual_identity != expected_identity:
            raise ValueError("Checkpoint preprocessing implementation mismatch")

    # Legitimate test-stub checkpoint: detect by its exact state_dict keys.
    if not {"model_config", "model_fingerprint"}.intersection(ckpt) and set(state.keys()) == {
        "conv.weight",
        "conv.bias",
    }:
        if (config or {}).get("model") is not None:
            raise ValueError(
                "Synthetic stub checkpoint cannot satisfy an explicit model configuration"
            )
        validate_preprocessing()
        conv = nn.Conv3d(1, 3, kernel_size=1)
        conv.load_state_dict({"weight": state["conv.weight"], "bias": state["conv.bias"]})
        stub: nn.Module = _TinyStub(conv)
        stub = stub.to(device)
        stub.eval()
        return stub

    # Otherwise this is a real DenseUNet3d checkpoint. Let a genuine key/shape
    # mismatch surface (re-raised with context) instead of being swallowed and
    # masked by a misleading 'Cannot reconstruct model' message.
    from dense_unet_3d.model.config import build_model, validate_model_metadata

    model_config: dict[str, Any] | str
    if "model_config" in ckpt or "model_fingerprint" in ckpt:
        model_config = validate_model_metadata(ckpt, (config or {}).get("model"))
    else:
        # The sole supported metadata-free production graph is the known
        # historical reduced model. Tensor count never selects a candidate.
        model_config = "historical_reduced"
        if (config or {}).get("model") is not None:
            from dense_unet_3d.model.config import canonical_model_config

            if (config or {})["model"] not in (
                "historical_reduced",
                canonical_model_config("historical_reduced"),
            ):
                raise ValueError(
                    "Metadata-free checkpoint supports only the historical reduced graph"
                )
    validate_preprocessing()
    model: nn.Module = build_model(model_config)
    try:
        model.load_state_dict(state)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load DenseUNet3d state_dict from checkpoint {checkpoint_path}: {exc}"
        ) from exc
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number of seconds")
    return seconds


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _print_training_plan(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Print before decoding data, creating a model or touching CUDA."""
    from dense_unet_3d.training.runtime import describe_schedule

    runtime = config.get("runtime", {})
    schedule = describe_schedule(config)
    wall = args.wall_seconds if args.wall_seconds is not None else runtime.get("wall_seconds")
    budget = (
        args.budget_seconds if args.budget_seconds is not None else runtime.get("budget_seconds")
    )
    prior_seconds = 0.0
    if args.command == "resume":
        run_dir = Path(config["pathing"]["model_save_dir"]) / config["pathing"]["run_name"]
        state_path = run_dir / "runtime.json"
        if state_path.exists():
            prior = json.loads(state_path.read_text())
            if budget is None:
                budget = prior["budget_seconds"]
            prior_seconds = prior.get("cumulative_seconds", 0.0)
    cadence = schedule["validation_every"]
    for phase in schedule["phases"].values():
        phase["validation_passes"] = (phase["epochs"] + cadence - 1) // cadence
    sys.stdout.write("Resolved training plan:\n" + json.dumps(schedule, sort_keys=True) + "\n")
    for warning in schedule["warnings"]:
        sys.stdout.write(f"Schedule warning: {warning}\n")
    sys.stdout.write(
        "Validation: full configured split on each scheduled validation; counts pending CPU preflight. "
        "Best selection uses those validations.\n"
        f"Invocation wall limit: {wall if wall is not None else 'unbounded'} seconds; "
        f"persistent cumulative limit: {budget if budget is not None else 'unbounded'} seconds.\n"
        "Accounting includes preflight, allocation, training, validation, checkpoint I/O "
        "and requested final evaluation. Stop at a consistent epoch boundary.\n"
        f"Previously charged runtime: {prior_seconds:.3f} seconds.\n"
        "Stops: completion, budget exhausted, SIGINT/SIGTERM/user stopped, or failure. "
        "No automatic restart. ETA is unknown until measured; the full schedule may exceed "
        "this invocation's allocation.\n"
    )
    if args.final_eval:
        sys.stdout.write(
            f"Final evaluation: at most {runtime.get('final_eval_max_batches', 100)} batches, "
            f"{runtime.get('final_eval_wall_seconds', 300)} seconds and remaining run budget.\n"
        )
    sys.stdout.flush()


def _cmd_train(args: argparse.Namespace) -> None:
    from dense_unet_3d.training.runtime import RunSession

    config = _load_config(args.config)
    _print_training_plan(config, args)
    three_class_a = config.get("training", {}).get("phase_a_targets") == "three_class"
    resume = args.command == "resume"
    with RunSession(
        config,
        resume=resume,
        wall_seconds=args.wall_seconds,
        budget_seconds=args.budget_seconds,
        recover=args.recover,
        max_retries=args.max_retries,
    ) as session:
        session.event("setup", dry_run=args.dry_run)
        remaining = (
            session.budget_seconds - session.cumulative_seconds
            if session.budget_seconds is not None
            else None
        )
        sys.stdout.write(
            f"Effective persistent budget: {session.budget_seconds}; "
            f"charged: {session.cumulative_seconds:.3f}; remaining: {remaining} seconds.\n"
        )
        sys.stdout.flush()
        reason = session.stop_reason()
        if reason:
            session.finish(reason)
            sys.stdout.write(f"Training {reason}.\n")
            return
        if args.dry_run:
            device = torch.device("cpu")
            model: nn.Module = _TinyStub(nn.Conv3d(1, 3, kernel_size=1))
            sys.stdout.write(
                "Validation workload: 2 synthetic cases, 1 CPU batch per validation.\n"
            )
            phase_a_train_loader = _make_dry_run_loader(detect_tumors=three_class_a)
            phase_a_val_loader = _make_dry_run_loader(detect_tumors=three_class_a)
            phase_b_train_loader = _make_dry_run_loader()
            phase_b_val_loader = _make_dry_run_loader()
        else:
            from dense_unet_3d.dataset.prepare_dataset import (
                discover_pairs,
                preflight_config,
                prepare_dataloader,
            )
            from dense_unet_3d.model.config import build_model
            from dense_unet_3d.training.experiment import configure_execution

            pairs = discover_pairs(config["pathing"]["test_img_dirs"])
            batch_size = config["dataset"]["batch_size"]
            sys.stdout.write(
                f"Validation workload before preflight: {len(pairs)} cases, "
                f"{math.ceil(len(pairs) / batch_size)} batches per validation.\n"
            )
            sys.stdout.flush()
            counts = preflight_config(config, full_decode=True)
            session.event("preflight_complete", **counts)
            sys.stdout.write(f"Validation workload: {counts['validation']} cases per validation.\n")
            reason = session.stop_reason()
            if reason:
                session.finish(reason)
                sys.stdout.write(f"Training {reason} during preflight.\n")
                return
            device = _device_from_config(config)
            configure_execution(config)
            model = build_model(config.get("model"))
            phase_a_train_loader = prepare_dataloader(
                config, train=True, detect_tumors=three_class_a
            )
            phase_a_val_loader = prepare_dataloader(
                config, train=False, detect_tumors=three_class_a
            )
            phase_b_train_loader = prepare_dataloader(config, train=True, detect_tumors=True)
            phase_b_val_loader = prepare_dataloader(config, train=False, detect_tumors=True)
            for name, loader in (
                ("train", phase_a_train_loader),
                ("validation", phase_a_val_loader),
            ):
                sys.stdout.write(
                    f"{name}: {len(cast(Sized, loader.dataset))} samples, {len(loader)} minibatches; "
                    f"microbatch={loader.batch_size}, accumulation=1.\n"
                )

        from dense_unet_3d.training.cascaded_driver import run_cascaded_training

        result = run_cascaded_training(
            config,
            model,
            device,
            phase_a_train_loader,
            val_loader=phase_a_val_loader,
            phase_b_train_loader=phase_b_train_loader,
            phase_b_val_loader=phase_b_val_loader,
            session=session,
            resume=resume,
        )
        reason = result["terminal_reason"]
        if reason == "completed" and args.final_eval:
            from dense_unet_3d.evaluation.evaluate import EvaluationInterrupted, evaluate

            runtime = config.get("runtime", {})
            session.event("training_completed", training_completed=True)
            evaluation_start = time.monotonic()
            reason = session.stop_reason() or reason
            if reason == "completed":
                best_path = Path(session.run_dir) / "phase_b" / "best.pt"
                model = _load_model_from_checkpoint(str(best_path), device)
                try:
                    metrics = evaluate(
                        model,
                        device,
                        phase_b_val_loader,
                        wall_seconds=max(
                            1e-9,
                            runtime.get("final_eval_wall_seconds", 300)
                            - (time.monotonic() - evaluation_start),
                        ),
                        max_batches=runtime.get("final_eval_max_batches", 100),
                        stop_requested=session.stop_reason,
                    )
                    session.event("final_evaluation", metrics=metrics)
                    sys.stdout.write(f"Final evaluation: {json.dumps(metrics, sort_keys=True)}\n")
                except EvaluationInterrupted as exc:
                    reason = session.stop_reason() or "budget exhausted"
                    session.event(
                        "final_evaluation_stopped", training_completed=True, error=str(exc)
                    )
        session.finish(reason)
        sys.stdout.write(
            "Training complete.\n" if reason == "completed" else f"Training {reason}.\n"
        )
        for phase in ("phase_a", "phase_b"):
            metadata = result.get(phase)
            if metadata:
                sys.stdout.write(f"  {phase} best epoch: {metadata['best_epoch']}\n")


def _cmd_status(args: argparse.Namespace) -> None:
    from dense_unet_3d.training.runtime import read_status

    started = time.monotonic()
    while True:
        sys.stdout.write(json.dumps(read_status(args.run_dir), sort_keys=True, default=str) + "\n")
        sys.stdout.flush()
        remaining = args.max_seconds - (time.monotonic() - started)
        if not args.watch or remaining <= 0:
            return
        time.sleep(min(args.interval, remaining))


def _cmd_stop(args: argparse.Namespace) -> None:
    from dense_unet_3d.training.runtime import request_stop

    request_stop(args.run_dir)
    sys.stdout.write("Stop requested; wait for terminal reason and ownership release.\n")


def _cmd_preflight(args: argparse.Namespace) -> None:
    """``dense-unet-3d preflight --config <path> [--full-decode]``."""
    from dense_unet_3d.dataset.prepare_dataset import preflight_config

    counts = preflight_config(_load_config(args.config), full_decode=args.full_decode)
    coverage = "full decode" if args.full_decode else "headers only"
    sys.stdout.write(
        f"Preflight passed ({coverage}): {counts['train']} training and "
        f"{counts['validation']} validation pairs.\n"
    )


# ---------------------------------------------------------------------------
# Subcommand: eval
# ---------------------------------------------------------------------------


def _cmd_eval(args: argparse.Namespace) -> None:
    """``dense-unet-3d eval --config <path> --checkpoint <path> [--dry-run]``."""
    config = _load_config(args.config)
    started = time.monotonic()
    device = torch.device("cpu") if args.dry_run else _device_from_config(config)

    model = _load_model_from_checkpoint(
        args.checkpoint,
        device,
        None if args.dry_run else config,
        allow_legacy_preprocessing=args.allow_legacy_preprocessing,
    )

    if args.dry_run:
        val_loader = _make_dry_run_loader()
    else:
        from dense_unet_3d.dataset.prepare_dataset import prepare_dataloader

        val_loader = prepare_dataloader(config, train=False)

    from dense_unet_3d.evaluation.evaluate import evaluate

    remaining = args.wall_seconds - (time.monotonic() - started)
    if remaining <= 0:
        raise RuntimeError("Evaluation budget exhausted during setup")
    metrics = evaluate(
        model, device, val_loader, wall_seconds=remaining, max_batches=args.max_batches
    )
    sys.stdout.write("Dice metrics:\n")
    for key, value in metrics.items():
        sys.stdout.write(f"  {key}: {value:.4f}\n")


# ---------------------------------------------------------------------------
# Subcommand: predict
# ---------------------------------------------------------------------------


def _cmd_predict(args: argparse.Namespace) -> None:
    """``dense-unet-3d predict --config --checkpoint --input --output``."""
    config = _load_config(args.config)
    device = _device_from_config(config)

    model = _load_model_from_checkpoint(
        args.checkpoint,
        device,
        config,
        allow_legacy_preprocessing=args.allow_legacy_preprocessing,
    )

    # Load the input NIfTI volume.
    input_img: nib.nifti1.Nifti1Image = nib.load(args.input)  # type: ignore[assignment]
    if len(input_img.shape) != 3:
        raise ValueError(f"predict expects a 3-D NIfTI volume, got shape {input_img.shape}")
    affine = input_img.affine
    pred_hwd: np.ndarray[Any, Any]

    from dense_unet_3d.dataset.prepare_dataset import sampling_mode

    mode = sampling_mode(config)
    if mode == "native_slabs":
        from dense_unet_3d.evaluation.predict import predict_volume

        pred_hwd = predict_volume(model, device, input_img, config["dataset"])
    else:
        data: np.ndarray[Any, Any] = input_img.get_fdata(dtype=np.float32)
        # Mirror the deterministic training preprocessing.  Defaults retain the
        # documented model contract for minimal inference-only config files.
        dataset_config = config.get("dataset", {})
        if dataset_config.get("clamp_hu", True):
            clamp_range = dataset_config.get("clamp_hu_range", {})
            data = np.clip(
                data,
                float(clamp_range.get("min", -200.0)),
                float(clamp_range.get("max", 250.0)),
            )

        # Convert to NCDHW tensor: (H, W, D) → (1, 1, D, H, W).
        volume = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()
        volume = volume.to(device)

        # Resize exactly as the deterministic image pipeline does, then map labels
        # back with nearest-neighbour interpolation to retain the input geometry.
        if not dataset_config.get("resize_img", True):
            raise ValueError(
                "predict requires dataset.resize_img=true because DenseUNet3d has a fixed spatial input contract"
            )
        resize_dims = dataset_config.get("resize_dims", {})
        model_dhw = (
            int(resize_dims.get("D", 12)),
            int(resize_dims.get("H", 224)),
            int(resize_dims.get("W", 224)),
        )
        orig_dhw = (volume.shape[2], volume.shape[3], volume.shape[4])
        volume = F.interpolate(volume, size=model_dhw, mode="trilinear", align_corners=False)

        # Run inference.
        with torch.no_grad():
            logits = model(volume)  # (1, C, D, H, W)

        # Argmax over channel dim → (1, 1, D, H, W) label volume at the model resolution.
        pred = logits.argmax(dim=1, keepdim=True).float()

        # Map labels back on the same half-pixel grid, preserving discrete classes.
        pred = F.interpolate(pred, size=orig_dhw, mode="nearest-exact")
        pred_np = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.int16)  # (D, H, W)

        # Back to NIfTI HWD order: (D, H, W) → (H, W, D).
        pred_hwd = np.transpose(pred_np, (1, 2, 0))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    header = input_img.header.copy()
    header.set_data_dtype(np.int16)
    out_img = nib.Nifti1Image(pred_hwd, affine, header=header)
    qform, qform_code = input_img.get_qform(coded=True)
    sform, sform_code = input_img.get_sform(coded=True)
    out_img.set_qform(qform, int(qform_code))
    out_img.set_sform(sform, int(sform_code))
    nib.save(out_img, args.output)
    sys.stdout.write(f"Segmentation saved to: {os.path.abspath(args.output)}\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dense-unet-3d",
        description=(
            "3D-DenseUNet-569 — reduced-depth 3-D medical image segmentation. "
            "Use train, resume, status, stop, preflight, eval, or predict."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    for command in ("train", "resume"):
        training_parser = subparsers.add_parser(
            command,
            help="Start training."
            if command == "train"
            else "Resume an exact recovery checkpoint.",
        )
        training_parser.add_argument("--config", required=True, metavar="PATH")
        training_parser.add_argument(
            "--dry-run", action="store_true", help="Synthetic CPU-only run."
        )
        training_parser.add_argument(
            "--wall-seconds",
            type=_positive_seconds,
            default=None,
            help="Invocation wall allocation, including setup.",
        )
        training_parser.add_argument(
            "--budget-seconds",
            type=_positive_seconds,
            default=None,
            help="Persistent allocation; resume cannot reset it.",
        )
        training_parser.add_argument(
            "--recover",
            action="store_true",
            help="Recover verified stale ownership after crash/reboot.",
        )
        training_parser.add_argument(
            "--max-retries",
            type=int,
            default=None,
            help="Persistent failed/unknown attempt recovery allowance.",
        )
        training_parser.add_argument(
            "--final-eval",
            action="store_true",
            help="Bounded Phase B best evaluation after completion.",
        )

    status_parser = subparsers.add_parser(
        "status", help="Print durable progress/ownership as JSON."
    )
    status_parser.add_argument("--run-dir", required=True)
    status_parser.add_argument("--watch", action="store_true")
    status_parser.add_argument("--interval", type=_positive_seconds, default=10.0)
    status_parser.add_argument("--max-seconds", type=_positive_seconds, default=3600.0)
    stop_parser = subparsers.add_parser("stop", help="Request safe stop from verified local owner.")
    stop_parser.add_argument("--run-dir", required=True)

    # -- preflight ------------------------------------------------------------
    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Validate configured training and validation NIfTI pairs.",
        description="Audit all configured labelled NIfTI pairs without creating a model or using CUDA.",
    )
    preflight_parser.add_argument(
        "--config", required=True, metavar="PATH", help="Path to YAML config file."
    )
    preflight_parser.add_argument(
        "--full-decode",
        action="store_true",
        default=False,
        help="Also decode CT and mask voxels; require finite CTs and labels in {0, 1, 2}.",
    )
    # -- eval -----------------------------------------------------------------
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a checkpoint and print Dice metrics.",
        description=(
            "Load a checkpoint and evaluate it on the validation split, "
            "printing liver + tumor Dice (per-case and global)."
        ),
    )
    eval_parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to YAML config file.",
    )
    eval_parser.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Path to .pt checkpoint file.",
    )
    eval_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Use synthetic validation data (no real NIfTI files needed).",
    )
    eval_parser.add_argument(
        "--allow-legacy-preprocessing",
        action="store_true",
        help="Allow a checkpoint without preprocessing identity and emit a prominent warning.",
    )

    eval_parser.add_argument(
        "--wall-seconds",
        type=_positive_seconds,
        default=300.0,
        help="Wall budget including setup (default 300 seconds).",
    )
    eval_parser.add_argument(
        "--max-batches",
        "--max-cases",
        type=_positive_int,
        default=100,
        help="Whole-case limit for native slabs; batch limit for historical input. Partial cohort scores are withheld.",
    )

    # -- predict --------------------------------------------------------------
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run inference on a NIfTI volume and write a segmentation.",
        description=(
            "Load a checkpoint, run inference on --input NIfTI volume, and "
            "write the predicted segmentation (labels 0/1/2) to --output."
        ),
    )
    predict_parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to YAML config file.",
    )
    predict_parser.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Path to .pt checkpoint file.",
    )
    predict_parser.add_argument(
        "--allow-legacy-preprocessing",
        action="store_true",
        help="Allow a checkpoint without preprocessing identity and emit a prominent warning.",
    )
    predict_parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Path to input NIfTI volume (.nii or .nii.gz).",
    )
    predict_parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output path for the NIfTI segmentation (.nii or .nii.gz).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ``dense-unet-3d`` console script."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command in {"train", "resume"}:
        _cmd_train(args)
    elif args.command == "status":
        _cmd_status(args)
    elif args.command == "stop":
        _cmd_stop(args)
    elif args.command == "preflight":
        _cmd_preflight(args)
    elif args.command == "eval":
        _cmd_eval(args)
    elif args.command == "predict":
        _cmd_predict(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
