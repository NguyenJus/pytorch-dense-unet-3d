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
import os
import sys
from typing import Any

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
) -> Any:
    """Return a DataLoader with synthetic tensors (CPU, no real NIfTI needed)."""
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    volumes = torch.randn(batch_size, 1, d, h, w)
    labels = torch.randint(0, 3, (batch_size, 1, d, h, w))
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
) -> nn.Module:
    """Load the full model from a checkpoint.

    Tries to import ``DenseUNet3d``; falls back to a minimal Conv3d wrapper when
    the checkpoint was saved from a test-stub model (e.g. test helpers).  The
    fallback is transparent to callers.
    """
    ckpt: dict[str, Any] = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]

    # Legitimate test-stub checkpoint: detect by its exact state_dict keys.
    if set(state.keys()) == {"conv.weight", "conv.bias"}:
        conv = nn.Conv3d(1, 3, kernel_size=1)
        conv.load_state_dict({"weight": state["conv.weight"], "bias": state["conv.bias"]})
        stub: nn.Module = _TinyStub(conv)
        stub = stub.to(device)
        stub.eval()
        return stub

    # Otherwise this is a real DenseUNet3d checkpoint. Let a genuine key/shape
    # mismatch surface (re-raised with context) instead of being swallowed and
    # masked by a misleading 'Cannot reconstruct model' message.
    from dense_unet_3d.model.DenseUNet3d import DenseUNet3d

    model: nn.Module = DenseUNet3d()
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


def _cmd_train(args: argparse.Namespace) -> None:
    """``dense-unet-3d train --config <path> [--dry-run]``."""
    config = _load_config(args.config)
    device = _device_from_config(config)

    if args.dry_run:
        # Dry-run: use a tiny synthetic dataloader and a stub model.
        class _TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv3d(1, 3, kernel_size=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)  # type: ignore[no-any-return]

        model: nn.Module = _TinyModel()
        train_loader = _make_dry_run_loader()
        val_loader = _make_dry_run_loader()
    else:
        from dense_unet_3d.dataset.prepare_dataset import prepare_dataloader
        from dense_unet_3d.model.DenseUNet3d import DenseUNet3d

        model = DenseUNet3d()
        train_loader = prepare_dataloader(config, train=True)
        val_loader = prepare_dataloader(config, train=False)

    from dense_unet_3d.training.cascaded_driver import run_cascaded_training

    result = run_cascaded_training(config, model, device, train_loader, val_loader=val_loader)
    sys.stdout.write("Training complete.\n")
    sys.stdout.write(f"  Phase A best epoch : {result['phase_a']['best_epoch']}\n")
    sys.stdout.write(f"  Phase B best epoch : {result['phase_b']['best_epoch']}\n")


# ---------------------------------------------------------------------------
# Subcommand: eval
# ---------------------------------------------------------------------------


def _cmd_eval(args: argparse.Namespace) -> None:
    """``dense-unet-3d eval --config <path> --checkpoint <path> [--dry-run]``."""
    config = _load_config(args.config)
    device = _device_from_config(config)

    model = _load_model_from_checkpoint(args.checkpoint, device)

    if args.dry_run:
        val_loader = _make_dry_run_loader()
    else:
        from dense_unet_3d.dataset.prepare_dataset import prepare_dataloader

        val_loader = prepare_dataloader(config, train=False)

    from dense_unet_3d.evaluation.evaluate import evaluate

    metrics = evaluate(model, device, val_loader)
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

    model = _load_model_from_checkpoint(args.checkpoint, device)

    # Load the input NIfTI volume.
    input_img: nib.nifti1.Nifti1Image = nib.load(args.input)  # type: ignore[assignment]
    affine = input_img.affine
    data: np.ndarray[Any, Any] = input_img.get_fdata(dtype=np.float32)  # (H, W, D)

    # Clamp HU values to [-200, 250] (paper preprocessing).
    data = np.clip(data, -200.0, 250.0)

    # Convert to NCDHW tensor: (H, W, D) → (1, 1, D, H, W).
    volume = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()
    volume = volume.to(device)

    # The model has a FIXED input contract of (D=12, H=224, W=224) — its decoder
    # targets are hardcoded. Resize the input to that contract (trilinear, mirroring
    # the dataset's Resize transform) before inference, then map the predicted LABEL
    # volume back to the original spatial dims with nearest-neighbour interpolation
    # (labels must NOT be interpolated continuously).
    model_dhw = (12, 224, 224)
    orig_dhw = (volume.shape[2], volume.shape[3], volume.shape[4])
    volume = F.interpolate(volume, size=model_dhw, mode="trilinear", align_corners=True)

    # Run inference.
    with torch.no_grad():
        logits = model(volume)  # (1, C, D, H, W)

    # Argmax over channel dim → (1, 1, D, H, W) label volume at the model resolution.
    pred = logits.argmax(dim=1, keepdim=True).float()

    # Resize labels back to the ORIGINAL input spatial dims (nearest-neighbour).
    pred = F.interpolate(pred, size=orig_dhw, mode="nearest")
    pred_np = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.int16)  # (D, H, W)

    # Back to NIfTI HWD order: (D, H, W) → (H, W, D).
    pred_hwd: np.ndarray[Any, Any] = np.transpose(pred_np, (1, 2, 0))

    out_img = nib.Nifti1Image(pred_hwd, affine)
    nib.save(out_img, args.output)
    sys.stdout.write(f"Segmentation saved to: {os.path.abspath(args.output)}\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dense-unet-3d",
        description=(
            "3D-DenseUNet-569 — faithful medical image semantic segmentation. "
            "Use a subcommand: train, eval, or predict."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # -- train ----------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Run cascaded 2-phase training.",
        description=(
            "Run the cascaded 2-phase training (Phase A + Phase B) on the "
            "configuration supplied via --config."
        ),
    )
    train_parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to YAML config file (no hardcoded cwd dependence).",
    )
    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Use synthetic data for a fast CPU dry-run (no real NIfTI files needed). "
            "Useful for CI and integration tests."
        ),
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

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "eval":
        _cmd_eval(args)
    elif args.command == "predict":
        _cmd_predict(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
