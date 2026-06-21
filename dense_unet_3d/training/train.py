"""Core training loop for 3D-DenseUNet-569.

Bug fixes (§4):
- Scheduler None guard: both scheduler.step() and scheduler.state_dict() are
  guarded by ``if scheduler is not None``.
- criterion.to(device) dropped: the weight tensor is moved to device inside
  get_criterion() via loss.py (standard pattern); the criterion itself is never
  .to()-moved.
- Loss counters initialised BEFORE the loop: ``running_loss = 0.0`` and
  ``num_batches = 0`` are set before the for-loop so an empty dataloader never
  raises NameError.  An empty loader returns 0.0 for that epoch's loss.
- Per-epoch logging: train loss + val Dice (per-case + global) via evaluate().
- Optimizer: SGD momentum=0.5, lr=0.01 (paper values).
- Scheduler: StepLR step_size=10, gamma=0.5 (paper values).
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dense_unet_3d.training.loss import get_criterion


def get_optimizer(model: nn.Module, config: dict[str, Any]) -> optim.Optimizer:
    """Return the configured optimiser (SGD momentum=0.5 lr=0.01 per paper).

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
    optimizer_name: str = config["training"]["optimizer"]
    learning_rate: float = config["training"]["learning_rate"]

    if optimizer_name == "SGD":
        momentum: float = config["training"]["momentum"]
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
        )
    if optimizer_name == "Adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )
    raise ValueError(f"Unknown optimizer: {optimizer_name!r}")


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
    2. ``criterion.to(device)`` is gone; the weight tensor is moved to device
       inside ``get_criterion(logits)`` (see ``dense_unet_3d.training.loss``).
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
    run_name: str = config["pathing"]["run_name"]
    model_save_dir: str = config["pathing"]["model_save_dir"]
    ckpt_dir = os.path.join(model_save_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    model = model.to(device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    total_epochs: int = config["training"]["epochs"]
    losses: list[float] = []

    for epoch in tqdm(range(1, total_epochs + 1), position=0, leave=True):
        model.train()

        # Counters BEFORE the loop — prevents NameError on empty dataloader.
        running_loss: float = 0.0
        num_batches: int = 0

        for volume, segmentation in dataloader:
            volume = volume.to(device, dtype=torch.float32)
            segmentation = segmentation.to(device, dtype=torch.long)

            optimizer.zero_grad()

            logits = model(volume)

            # Move weight tensor to device inside get_criterion (no .to(criterion)).
            criterion = get_criterion(logits)
            loss = criterion(logits, segmentation.squeeze(1))

            loss.backward()
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
            torch.save(ckpt, os.path.join(ckpt_dir, f"epoch{epoch}.pt"))

    return losses
