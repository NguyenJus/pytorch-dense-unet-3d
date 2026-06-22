"""Cascaded two-phase training driver for 3D-DenseUNet-569.

Paper: Alalwan et al. (2021) §Training Scheme.

Design (§6 F2)
--------------
Phase A: ``phase_a_epochs`` epochs, each epoch runs ``phase_a_steps_per_epoch``
    steps over the dataloader (cycling as needed). At each epoch val Dice is
    computed; the epoch with the highest val Dice is saved as the ``best``
    checkpoint (plus a ``last`` checkpoint for the final epoch).

Phase B: Reload Phase A best weights into a fresh optimizer/scheduler, then
    run ``phase_b_epochs`` epochs × ``phase_b_steps_per_epoch`` steps.
    Again saves ``best`` (by val Dice) and ``last`` checkpoints.

"10 steps per epoch" interpretation
------------------------------------
The paper states "each epoch = 10 steps/sub-epochs" without further definition.
We treat one *step* as one full pass through the dataloader (i.e.
``steps_per_epoch`` passes per epoch).  This is the LITERAL reading of the
phrase "10 steps" — configurable via ``phase_a_steps_per_epoch`` and
``phase_b_steps_per_epoch`` (default 10 each).

See docs/research/2026-06-21-denseunet569-architecture-decisions.md for the
rationale and decision record.

Checkpoint schema
-----------------
Every checkpoint is a dict with these mandatory keys::

    {
      "model_state_dict":     ...,
      "optimizer_state_dict": ...,
      "scheduler_state_dict": ... or None,
      "epoch":                int,
      "metrics":              dict[str, float],
    }

The ``last`` and ``best`` checkpoint file names under each phase sub-dir::

    <model_save_dir>/<run_name>/phase_a/best.pt
    <model_save_dir>/<run_name>/phase_a/last.pt
    <model_save_dir>/<run_name>/phase_b/best.pt
    <model_save_dir>/<run_name>/phase_b/last.pt
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from itertools import islice
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dense_unet_3d.training.loss import get_criterion
from dense_unet_3d.training.train import get_optimizer, get_scheduler

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "run_phase_a",
    "run_phase_b",
    "run_cascaded_training",
]

# Default steps per epoch — literal paper reading (§6 F2 decision record).
DEFAULT_STEPS_PER_EPOCH: int = 10


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    *,
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    """Save a training checkpoint to *path*.

    Parameters
    ----------
    path:
        Full file path (e.g. ``".../phase_a/best.pt"``).
    model:
        The model whose ``state_dict`` to save.
    optimizer:
        The optimizer whose ``state_dict`` to save.
    scheduler:
        The LR scheduler, or ``None`` if not used.  Saves ``None`` for the
        ``scheduler_state_dict`` key when not present.
    epoch:
        Current epoch number (1-indexed).
    metrics:
        Dictionary of evaluation metrics (e.g. ``{"val_dice": 0.85}``).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(ckpt, path)


def load_checkpoint(
    *,
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    """Load a checkpoint from *path*, restoring model (and optionally optimizer/scheduler).

    Parameters
    ----------
    path:
        Full file path to the ``.pt`` checkpoint.
    model:
        Model to restore in-place.
    optimizer:
        Optional optimizer to restore in-place.
    scheduler:
        Optional LR scheduler to restore in-place.  Ignored when the
        checkpoint's ``scheduler_state_dict`` is ``None`` or when
        *scheduler* itself is ``None``.

    Returns
    -------
    dict
        The raw checkpoint dict (includes ``epoch``, ``metrics``, etc.).
    """
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Internal: one epoch of training (steps_per_epoch full loops over loader)
# ---------------------------------------------------------------------------


def _cycling_iter(loader: DataLoader) -> Iterator:
    """Yield batches from *loader* indefinitely (cycle)."""
    while True:
        yield from loader


def _run_epoch(
    *,
    config: dict[str, Any],
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    steps_per_epoch: int,
) -> float:
    """Run one training epoch (``steps_per_epoch`` mini-batch steps).

    A *step* here means one mini-batch gradient update (not a full pass over
    the dataset).  We cycle through the loader as needed.

    Returns
    -------
    float
        Mean loss over all steps in this epoch.  0.0 if steps_per_epoch == 0.
    """
    model.train()
    running_loss: float = 0.0
    count: int = 0

    # Build the criterion ONCE (weights from config, tensor on device) — never
    # rebuild it per step inside the loop.
    criterion = get_criterion(config, device=device)

    for volume, segmentation in islice(_cycling_iter(loader), steps_per_epoch):
        volume = volume.to(device, dtype=torch.float32)
        if segmentation.dim() == 5:
            segmentation = segmentation.squeeze(1)
        segmentation = segmentation.to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(volume)
        loss = criterion(logits, segmentation)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1

    return running_loss / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Phase A
# ---------------------------------------------------------------------------


def run_phase_a(
    config: dict[str, Any],
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Run Phase A of the cascaded training scheme.

    Phase A: ``phase_a_epochs`` epochs × ``phase_a_steps_per_epoch`` steps.
    Saves ``best`` (by val Dice) and ``last`` checkpoints to
    ``<model_save_dir>/<run_name>/phase_a/``.

    Parameters
    ----------
    config:
        Run configuration dict.
    model:
        Model to train (moved to *device* inside).
    device:
        CPU or CUDA device.
    train_loader:
        DataLoader for the training split.
    val_loader:
        Optional DataLoader for validation Dice computation each epoch.

    Returns
    -------
    dict with keys:
        ``epoch_losses`` (list[float]), ``best_epoch`` (int),
        ``best_metrics`` (dict).
    """
    run_name: str = config["pathing"]["run_name"]
    model_save_dir: str = config["pathing"]["model_save_dir"]
    phase_dir = os.path.join(model_save_dir, run_name, "phase_a")
    os.makedirs(phase_dir, exist_ok=True)

    training_cfg = config["training"]
    total_epochs: int = training_cfg.get("phase_a_epochs", 100)
    steps_per_epoch: int = training_cfg.get(
        "phase_a_steps_per_epoch",
        training_cfg.get("steps_per_epoch", DEFAULT_STEPS_PER_EPOCH),
    )

    model = model.to(device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    epoch_losses: list[float] = []
    best_val_dice: float = -1.0
    best_epoch: int = 1
    best_metrics: dict[str, float] = {}

    for epoch in tqdm(range(1, total_epochs + 1), desc="Phase A", position=0, leave=True):
        epoch_loss = _run_epoch(
            config=config,
            model=model,
            device=device,
            loader=train_loader,
            optimizer=optimizer,
            steps_per_epoch=steps_per_epoch,
        )
        epoch_losses.append(epoch_loss)

        if scheduler is not None:
            scheduler.step()

        # Compute val Dice for best-checkpoint tracking.
        metrics: dict[str, float] = {"train_loss": epoch_loss}
        if val_loader is not None:
            from dense_unet_3d.evaluation.evaluate import evaluate  # lazy import

            model.eval()
            val_metrics = evaluate(model, device, val_loader)
            metrics.update(val_metrics)
            val_dice = float(val_metrics.get("liver_per_case", 0.0))
        else:
            val_dice = 0.0

        tqdm.write(f"Phase A epoch {epoch}/{total_epochs}: loss={epoch_loss:.4f}")

        # Update best checkpoint (strict >: equal score keeps the earlier epoch).
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            best_metrics = metrics
            save_checkpoint(
                path=os.path.join(phase_dir, "best.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
            )

    # Save last checkpoint.
    save_checkpoint(
        path=os.path.join(phase_dir, "last.pt"),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=total_epochs,
        metrics=metrics,  # metrics from the final epoch
    )

    return {
        "epoch_losses": epoch_losses,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
    }


# ---------------------------------------------------------------------------
# Phase B
# ---------------------------------------------------------------------------


def run_phase_b(
    config: dict[str, Any],
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    phase_a_best_path: str,
) -> dict[str, Any]:
    """Run Phase B of the cascaded training scheme.

    Phase B: reload Phase A best weights into *model*, then run
    ``phase_b_epochs`` epochs × ``phase_b_steps_per_epoch`` steps.
    Saves ``best`` (by val Dice) and ``last`` checkpoints.

    Parameters
    ----------
    config:
        Run configuration dict.
    model:
        Fresh (or any) model — weights will be REPLACED by Phase A best.
    device:
        CPU or CUDA device.
    train_loader:
        DataLoader for the training split.
    val_loader:
        Optional DataLoader for validation Dice.
    phase_a_best_path:
        Absolute path to the Phase A ``best.pt`` checkpoint.

    Returns
    -------
    dict with keys:
        ``epoch_losses`` (list[float]), ``best_epoch`` (int),
        ``best_metrics`` (dict),
        ``loaded_phase_a_state_dict`` (dict — the state dict actually loaded,
        for test assertions).
    """
    run_name: str = config["pathing"]["run_name"]
    model_save_dir: str = config["pathing"]["model_save_dir"]
    phase_dir = os.path.join(model_save_dir, run_name, "phase_b")
    os.makedirs(phase_dir, exist_ok=True)

    training_cfg = config["training"]
    total_epochs: int = training_cfg.get("phase_b_epochs", 1000)
    steps_per_epoch: int = training_cfg.get(
        "phase_b_steps_per_epoch",
        training_cfg.get("steps_per_epoch", DEFAULT_STEPS_PER_EPOCH),
    )

    # Move model to device first so load_checkpoint maps to the right device.
    model = model.to(device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # RELOAD Phase A best weights — this is the core cascade step.
    phase_a_ckpt = torch.load(phase_a_best_path, map_location=device, weights_only=False)
    loaded_phase_a_state_dict: dict[str, torch.Tensor] = {
        k: v.clone() for k, v in phase_a_ckpt["model_state_dict"].items()
    }
    model.load_state_dict(phase_a_ckpt["model_state_dict"])

    epoch_losses: list[float] = []
    best_val_dice: float = -1.0
    best_epoch: int = 1
    best_metrics: dict[str, float] = {}

    for epoch in tqdm(range(1, total_epochs + 1), desc="Phase B", position=0, leave=True):
        epoch_loss = _run_epoch(
            config=config,
            model=model,
            device=device,
            loader=train_loader,
            optimizer=optimizer,
            steps_per_epoch=steps_per_epoch,
        )
        epoch_losses.append(epoch_loss)

        if scheduler is not None:
            scheduler.step()

        metrics: dict[str, float] = {"train_loss": epoch_loss}
        if val_loader is not None:
            from dense_unet_3d.evaluation.evaluate import evaluate  # lazy import

            model.eval()
            val_metrics = evaluate(model, device, val_loader)
            metrics.update(val_metrics)
            # Phase B: balanced selection on nanmean(liver_per_case, tumor_per_case).
            # If both are NaN the result is NaN; treat NaN as "never better" (nan > x is False).
            liver_pc = val_metrics.get("liver_per_case", float("nan"))
            tumor_pc = val_metrics.get("tumor_per_case", float("nan"))
            components = np.array([liver_pc, tumor_pc], dtype=float)
            if np.all(np.isnan(components)):
                val_dice = float("nan")
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    val_dice = float(np.nanmean(components))
        else:
            val_dice = 0.0

        tqdm.write(f"Phase B epoch {epoch}/{total_epochs}: loss={epoch_loss:.4f}")

        # Strict >: equal score keeps the earlier epoch; NaN is never better.
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            best_metrics = metrics
            save_checkpoint(
                path=os.path.join(phase_dir, "best.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
            )

    # Save last checkpoint.
    save_checkpoint(
        path=os.path.join(phase_dir, "last.pt"),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=total_epochs,
        metrics=metrics,
    )

    return {
        "epoch_losses": epoch_losses,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "loaded_phase_a_state_dict": loaded_phase_a_state_dict,
    }


# ---------------------------------------------------------------------------
# Full cascaded driver
# ---------------------------------------------------------------------------


def run_cascaded_training(
    config: dict[str, Any],
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Full cascaded two-phase training driver.

    Phase A then Phase B sequentially.  Phase B automatically reloads the
    Phase A best checkpoint before training begins.

    Parameters
    ----------
    config:
        Run configuration dict.
    model:
        Model to train.  Phase A trains from this state; Phase B creates a
        fresh copy of the same architecture and reloads Phase A best.
    device:
        CPU or CUDA device.
    train_loader:
        DataLoader for training split.
    val_loader:
        Optional validation DataLoader.

    Returns
    -------
    dict with keys:
        ``phase_a`` (phase A result dict),
        ``phase_b`` (phase B result dict),
        ``phase_b_loaded_phase_a_state_dict`` (the weights loaded into Phase B,
        for proof assertions in tests).
    """
    run_name: str = config["pathing"]["run_name"]
    model_save_dir: str = config["pathing"]["model_save_dir"]
    phase_a_best_path = os.path.join(model_save_dir, run_name, "phase_a", "best.pt")

    # --- Phase A ---
    phase_a_result = run_phase_a(
        config,
        model,
        device,
        train_loader,
        val_loader=val_loader,
    )

    # --- Phase B (fresh model reloads Phase A best) ---
    # We instantiate a same-class model to avoid modifying Phase A's final state.
    # For the driver we reuse the same model instance and reload weights.
    phase_b_result = run_phase_b(
        config,
        model,
        device,
        train_loader,
        val_loader=val_loader,
        phase_a_best_path=phase_a_best_path,
    )

    return {
        "phase_a": phase_a_result,
        "phase_b": phase_b_result,
        "phase_b_loaded_phase_a_state_dict": phase_b_result["loaded_phase_a_state_dict"],
    }
