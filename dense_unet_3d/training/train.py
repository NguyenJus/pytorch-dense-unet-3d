"""Core training loop for 3D-DenseUNet-569.

Bug fixes (§4):
- Scheduler None guard: both scheduler.step() and scheduler.state_dict() are
  guarded by ``if scheduler is not None``.
- Criterion weights are created directly on the requested device by
  get_criterion(), avoiding an extra module transfer.
- Loss counters initialised BEFORE the loop: ``running_loss = 0.0`` and
  ``num_batches = 0`` are set before the for-loop so an empty dataloader never
  raises NameError.  An empty loader returns 0.0 for that epoch's loss.
- Per-epoch logging: train loss + val Dice (per-case + global) via evaluate().
- Optimizer: provisional SGD; paper-stated momentum=0.5 and lr=0.01.
- Scheduler: StepLR step_size=10, gamma=0.5 (paper values).
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dense_unet_3d.training.loss import IGNORE_INDEX, get_criterion


def get_optimizer(model: nn.Module, config: dict[str, Any]) -> optim.Optimizer:
    """Return the configured optimiser; SGD is a named reconstruction assumption.

    Parameters
    ----------
    model:
        The model whose parameters to optimise.
    config:
        Training config dict with keys under ``config["training"]``.

    Returns
    -------
    torch.optim.Optimizer
    """
    training = config["training"]
    optimizer_name: str = training["optimizer"]
    learning_rate = float(training["learning_rate"])
    if not math.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("learning_rate must be finite and nonnegative")
    semantics = training.get("optimizer_semantics", "pytorch_gradient_buffer")
    if semantics != "pytorch_gradient_buffer":
        raise ValueError(f"Unsupported optimizer_semantics: {semantics!r}")
    if optimizer_name == "SGD":
        momentum = float(training["momentum"])
        dampening = float(training.get("dampening", 0.0))
        weight_decay = float(training.get("weight_decay", 0.0))
        nesterov = training.get("nesterov", False)
        if not isinstance(nesterov, bool):
            raise ValueError("nesterov must be a boolean")
        for name, value in (
            ("momentum", momentum),
            ("dampening", dampening),
            ("weight_decay", weight_decay),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov SGD requires positive momentum and zero dampening")
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    raise ValueError(f"Unknown optimizer: {optimizer_name!r}")


def unpack_training_batch(
    batch: Any,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize tuples or native slab dictionaries and exclude padded targets.

    Dataset metadata remains in the original dictionary. This helper does not
    remap any valid class label; phase target mapping is a separate decision.
    """
    if isinstance(batch, dict):
        volume, target = batch["image"], batch["target"]
        valid_mask = batch.get("valid_mask")
    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
        volume, target = batch
        valid_mask = None
    else:
        raise ValueError("Training batch must be (image, target) or a slab dictionary")
    if not isinstance(volume, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("Training image and target must be tensors")
    volume = volume.to(device, dtype=torch.float32)
    if target.ndim == 5 and target.shape[1] == 1:
        target = target.squeeze(1)
    if (
        target.ndim != 4
        or volume.ndim != 5
        or volume.shape[0] != target.shape[0]
        or volume.shape[2:] != target.shape[1:]
    ):
        raise ValueError("Training image and target must share batch and spatial dimensions")
    target = target.to(device)
    if valid_mask is not None:
        if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be a boolean tensor")
        if valid_mask.ndim == 5 and valid_mask.shape[1] == 1:
            valid_mask = valid_mask.squeeze(1)
        if valid_mask.shape != target.shape:
            raise ValueError("valid_mask must match target dimensions")
        valid_mask = valid_mask.to(device)
    observed = target if valid_mask is None else target[valid_mask]
    if observed.is_floating_point() and not torch.equal(observed, observed.round()):
        raise ValueError("Target labels must be integer class indices")
    target = target.to(dtype=torch.long)
    if valid_mask is not None:
        target = target.masked_fill(~valid_mask, IGNORE_INDEX)
    valid = target != IGNORE_INDEX
    if not valid.any():
        raise ValueError("Training batch contains no valid target voxels")
    if ((target[valid] < 0) | (target[valid] > 2)).any():
        raise ValueError("Valid target labels must be 0, 1 or 2")
    return volume, target


def get_scheduler(
    optimizer: optim.Optimizer,
    config: dict[str, Any],
) -> optim.lr_scheduler.LRScheduler | None:
    """Return the LR scheduler, or *None* when ``use_scheduler`` is False.

    Parameters
    ----------
    optimizer:
        The optimiser to attach the scheduler to.
    config:
        Training config dict.

    Returns
    -------
    LR scheduler, or ``None`` if scheduling is disabled.
    """
    if not config["training"]["use_scheduler"]:
        return None

    scheduler_name: str = config["training"]["scheduler"]
    if scheduler_name == "StepLR":
        step_size: int = config["training"]["scheduler_step"]
        gamma: float = config["training"]["scheduler_gamma"]
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    raise ValueError(f"Unknown scheduler: {scheduler_name!r}")


def reject_unmanaged_reconstruction(config: dict[str, Any]) -> None:
    experiment = config.get("experiment")
    if isinstance(experiment, dict) and experiment.get("kind") == "reconstruction":
        raise ValueError("Reconstruction requires managed run_cascaded_training")


def train(
    config: dict[str, Any],
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    val_loader: DataLoader | None = None,
) -> list[float]:
    """Core single-phase training loop.

    Fixes applied vs. original train.py
    -------------------------------------
    1. ``scheduler.step()`` / ``scheduler.state_dict()`` are guarded with
       ``if scheduler is not None`` — prevents AttributeError when
       ``use_scheduler=False``.
    2. The criterion's class-weight tensor is created on ``device`` inside
       ``get_criterion`` (see ``dense_unet_3d.training.loss``).
    3. ``running_loss = 0.0`` and ``num_batches = 0`` are initialised BEFORE
       the inner for-loop; an empty DataLoader returns 0.0 for that epoch
       instead of raising ``NameError: name 'i' is not defined``.
    4. Per-epoch logging: train loss printed each epoch; val Dice logged when
       a ``val_loader`` is supplied.

    Parameters
    ----------
    config:
        Run configuration dict (see ``_base_config`` in tests for schema).
    model:
        The model to train (moved to *device* inside this function).
    device:
        CPU or CUDA device.
    dataloader:
        DataLoader for the training split.
    val_loader:
        Optional DataLoader for the validation split.  When supplied, Dice
        metrics are computed at the end of each epoch via ``evaluate()``.

    Returns
    -------
    list[float]
        Per-epoch average train loss (length == ``config["training"]["epochs"]``).
        Each entry is 0.0 for an empty loader (documented; NaN-safe).
    """
    reject_unmanaged_reconstruction(config)
    warnings.warn(
        "The single-phase train() API is unmanaged and non-resumable; use "
        "run_cascaded_training for ownership, recovery and budgets.",
        UserWarning,
        stacklevel=2,
    )
    run_name: str = config["pathing"]["run_name"]
    model_save_dir: str = config["pathing"]["model_save_dir"]
    ckpt_dir = os.path.join(model_save_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    model = model.to(device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Build the criterion ONCE (weights read from config, tensor on device) —
    # never rebuild it per batch inside the inner loop.
    criterion = get_criterion(config, device=device)

    total_epochs: int = config["training"]["epochs"]
    losses: list[float] = []

    for epoch in tqdm(range(1, total_epochs + 1), position=0, leave=True):
        model.train()

        # Counters BEFORE the loop — prevents NameError on empty dataloader.
        running_loss: float = 0.0
        num_batches: int = 0

        for batch in dataloader:
            volume, segmentation = unpack_training_batch(batch, device)

            optimizer.zero_grad()

            logits = model(volume)

            # Reuse the criterion built once before the loop (no per-batch rebuild).
            loss = criterion(logits, segmentation)

            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
            ):
                raise FloatingPointError("Nonfinite training gradients")
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        # For an empty loader, num_batches == 0; return 0.0 (documented).
        epoch_loss = running_loss / num_batches if num_batches > 0 else 0.0
        losses.append(epoch_loss)

        # Guard: only call scheduler.step() when a scheduler exists.
        if scheduler is not None:
            scheduler.step()

        # Optional val Dice logging.
        if val_loader is not None:
            from dense_unet_3d.evaluation.evaluate import evaluate  # lazy import

            model.eval()
            metrics = evaluate(model, device, val_loader)
            tqdm.write(
                f"Epoch {epoch}: loss={epoch_loss:.4f} "
                f"liver_dice_pc={metrics['liver_per_case']:.4f} "
                f"tumor_dice_pc={metrics['tumor_per_case']:.4f}"
            )
        else:
            tqdm.write(f"Epoch {epoch}: loss={epoch_loss:.4f}")

        # Checkpoint every 10 epochs.
        if epoch % 10 == 0:
            ckpt: dict[str, Any] = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                # Guard: None when scheduler is disabled.
                "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None),
                "loss": epoch_loss,
                "losses": losses,
            }
            from dense_unet_3d.training.experiment import (
                checkpoint_model_metadata,
                experiment_metadata,
            )
            from dense_unet_3d.training.runtime import atomic_checkpoint

            ckpt.update(checkpoint_model_metadata(model))
            ckpt.update(experiment_metadata(config))
            atomic_checkpoint(os.path.join(ckpt_dir, f"epoch{epoch}.pt"), ckpt)

    return losses
