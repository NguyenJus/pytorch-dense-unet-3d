"""Cascaded two-phase training driver for 3D-DenseUNet-569.

Paper: Alalwan et al. (2021) §Training Scheme.

Design (§6 F2)
--------------
Phase A: ``phase_a_epochs`` epochs, each epoch runs ``phase_a_steps_per_epoch``
    mini-batch updates (cycling over the dataloader as needed). At each epoch val Dice is
    computed; the epoch with the highest val Dice is saved as the ``best``
    checkpoint (plus a ``last`` checkpoint for the final epoch).

Phase B: Reload Phase A best weights into a fresh optimizer/scheduler, then
    run ``phase_b_epochs`` epochs × ``phase_b_steps_per_epoch`` steps.
    Again saves ``best`` (by val Dice) and ``last`` checkpoints.

"10 steps per epoch" interpretation
------------------------------------
The paper states "each epoch = 10 steps/sub-epochs" without further definition.
We treat one *step* as one mini-batch update, cycling through the dataloader
as needed. This is configurable via ``phase_a_steps_per_epoch`` and
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
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dense_unet_3d.training.experiment import checkpoint_model_metadata, experiment_metadata
from dense_unet_3d.training.loss import IGNORE_INDEX, get_criterion
from dense_unet_3d.training.train import (
    get_optimizer,
    get_scheduler,
    reject_unmanaged_reconstruction,
    unpack_training_batch,
)

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "run_phase_a",
    "run_phase_b",
    "run_cascaded_training",
]

# Default mini-batch updates per epoch; the paper leaves this interpretation underspecified.
DEFAULT_STEPS_PER_EPOCH: int = 10


class _LiverOnlyLoader:
    """Yield a loader's batches with tumour labels folded into liver labels."""

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self.dataset = getattr(loader, "dataset", None)

    def __iter__(self) -> Iterator[Any]:
        for batch in self._loader:
            if isinstance(batch, dict):
                yield {**batch, "target": batch["target"].clamp(max=1)}
            else:
                volume, segmentation = batch
                yield volume, segmentation.clamp(max=1)


def phase_a_target_mode(config: dict[str, Any]) -> str:
    mode = config.get("training", {}).get("phase_a_targets", "liver_only")
    if mode not in {"liver_only", "three_class"}:
        raise ValueError(f"Unsupported phase_a_targets: {mode!r}")
    if mode == "liver_only" and config.get("dataset", {}).get("sampling") == "native_slabs":
        raise ValueError("native_slabs requires three_class phase A targets")
    return str(mode)


def validate_phase_transfer_policy(config: dict[str, Any]) -> None:
    policy = config.get("training", {}).get("phase_transfer_policy", "best_weights_fresh_optimizer")
    if policy != "best_weights_fresh_optimizer":
        raise ValueError(f"Unsupported phase_transfer_policy: {policy!r}")


def foreground_selection_score(metrics: dict[str, float]) -> float:
    """Select on available foreground Dice; reject nonfinite aggregate scores."""
    values = np.array(
        [metrics.get("liver_per_case", float("nan")), metrics.get("tumor_per_case", float("nan"))]
    )
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def validate_checkpoint_graph(checkpoint: dict[str, Any], model: nn.Module) -> None:
    metadata = checkpoint_model_metadata(model)
    if not metadata:
        return
    from dense_unet_3d.model.config import HISTORICAL, validate_model_metadata

    if "model_config" not in checkpoint and metadata["model_config"]["name"] == HISTORICAL:
        if "model_fingerprint" in checkpoint:
            raise ValueError("Incomplete checkpoint model identity")
        return  # Known historical Python API contract, never inferred for the new graph.
    validate_model_metadata(checkpoint, expected_config=metadata["model_config"])


def save_checkpoint(
    *,
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any] | None = None,
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
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    ckpt: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "metrics": metrics,
    }
    ckpt.update(checkpoint_model_metadata(model))
    if config is not None:
        ckpt.update(experiment_metadata(config))
    from dense_unet_3d.training.runtime import atomic_checkpoint

    atomic_checkpoint(path, ckpt)


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
    validate_checkpoint_graph(ckpt, model)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Internal: one epoch of training (steps_per_epoch minibatch updates)
# ---------------------------------------------------------------------------


def _run_epoch(
    *,
    config: dict[str, Any],
    model: nn.Module,
    device: torch.device,
    loader: Iterable[Any],
    optimizer: optim.Optimizer,
    steps_per_epoch: int,
    diagnostics: dict[str, Any] | None = None,
) -> float:
    """Run one training epoch (``steps_per_epoch`` mini-batch steps).

    A *step* here means one mini-batch gradient update (not a full pass over
    the dataset).  We cycle through the loader as needed.

    Raises
    ------
    ValueError
        If a positive number of steps is requested from an empty loader.

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

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            class_voxel_counts=[0, 0, 0],
            predicted_class_voxel_counts=[0, 0, 0],
            class_weighted_ce_sums=[0.0, 0.0, 0.0],
            tumor_positive_samples=0,
            case_ids_available=False,
            case_ids_seen=[],
            tumor_positive_case_ids_seen=[],
            unique_cases_seen=0,
            tumor_positive_cases_seen=0,
            valid_voxels=0,
            samples=0,
            updates=0,
            gradient_l2_max=0.0,
            sampled_parameter_update_l2_max=0.0,
            sampled_parameter_count=0,
        )
        parameter_limit = config.get("training", {}).get("diagnostic_parameter_limit", 4096)
        if type(parameter_limit) is not int or parameter_limit < 1:
            raise ValueError("diagnostic_parameter_limit must be a positive integer")
    batches = iter(loader)
    seen_cases: set[str] = set()
    positive_cases: set[str] = set()
    for _ in range(steps_per_epoch):
        try:
            batch = next(batches)
        except StopIteration:
            # Start another pass when more updates than batches are requested.
            # A second immediate StopIteration identifies an empty loader;
            # unlike ``islice(_cycling_iter(...))``, it cannot loop forever.
            batches = iter(loader)
            try:
                batch = next(batches)
            except StopIteration as exc:
                raise ValueError(
                    "train_loader yielded no batches, so a training step cannot run."
                ) from exc
        volume, segmentation = unpack_training_batch(batch, device)

        optimizer.zero_grad()
        logits = model(volume)
        loss = criterion(logits, segmentation)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite training gradients")
        sampled: list[tuple[torch.Tensor, torch.Tensor]] = []
        if diagnostics is not None:
            grad_sq = sum(
                float(torch.linalg.vector_norm(p.grad.detach()).item()) ** 2
                for p in model.parameters()
                if p.grad is not None
            )
            diagnostics["gradient_l2_max"] = max(diagnostics["gradient_l2_max"], grad_sq**0.5)
            remaining = parameter_limit
            for p in model.parameters():
                if p.requires_grad and remaining > 0:
                    values = p.detach().reshape(-1)[:remaining]
                    sampled.append((p, values.clone()))
                    remaining -= values.numel()
            diagnostics["sampled_parameter_count"] = parameter_limit - remaining
            with torch.no_grad():
                per_voxel = nn.functional.cross_entropy(
                    logits.detach(),
                    segmentation,
                    weight=getattr(criterion, "weight", None),
                    ignore_index=IGNORE_INDEX,
                    reduction="none",
                )
                prediction = logits.detach().argmax(dim=1)
                valid = segmentation != IGNORE_INDEX
                diagnostics["samples"] += segmentation.shape[0]
                diagnostics["valid_voxels"] += int(valid.sum().item())
                diagnostics["tumor_positive_samples"] += int(
                    (segmentation == 2).flatten(1).any(1).sum().item()
                )
                if isinstance(batch, dict) and "case_id" in batch:
                    case_ids = batch["case_id"]
                    if (
                        not isinstance(case_ids, (tuple, list))
                        or len(case_ids) != segmentation.shape[0]
                    ):
                        raise ValueError("Collated case IDs must match the training batch")
                    positive = (segmentation == 2).flatten(1).any(1).tolist()
                    seen_cases.update(map(str, case_ids))
                    positive_cases.update(
                        str(case)
                        for case, has_tumor in zip(case_ids, positive, strict=True)
                        if has_tumor
                    )
                    diagnostics["case_ids_available"] = True
                    diagnostics["case_ids_seen"] = sorted(seen_cases)
                    diagnostics["tumor_positive_case_ids_seen"] = sorted(positive_cases)
                    diagnostics["unique_cases_seen"] = len(seen_cases)
                    diagnostics["tumor_positive_cases_seen"] = len(positive_cases)
                for c in range(3):
                    label_mask = segmentation == c
                    diagnostics["class_voxel_counts"][c] += int(label_mask.sum().item())
                    diagnostics["predicted_class_voxel_counts"][c] += int(
                        ((prediction == c) & valid).sum().item()
                    )
                    diagnostics["class_weighted_ce_sums"][c] += float(
                        per_voxel[label_mask].sum().item()
                    )
        optimizer.step()
        if diagnostics is not None:
            update_sq = sum(
                float(
                    torch.linalg.vector_norm(
                        p.detach().reshape(-1)[: before.numel()] - before
                    ).item()
                )
                ** 2
                for p, before in sampled
            )
            diagnostics["sampled_parameter_update_l2_max"] = max(
                diagnostics["sampled_parameter_update_l2_max"], update_sq**0.5
            )
            diagnostics["updates"] += 1
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite model parameters")

        running_loss += loss.item()
        count += 1

    return running_loss / count if count > 0 else 0.0


def _validate_phase_schedule(total_epochs: int, steps_per_epoch: int, phase_name: str) -> None:
    """Reject schedules that cannot satisfy the phase checkpoint contract."""
    if total_epochs < 1:
        raise ValueError(f"{phase_name}_epochs must be at least 1, got {total_epochs}.")
    if steps_per_epoch < 1:
        raise ValueError(f"{phase_name}_steps_per_epoch must be at least 1, got {steps_per_epoch}.")


def _is_better_score(candidate: float, best: float) -> bool:
    """Return whether a finite validation score strictly improves the best score."""
    return np.isfinite(candidate) and candidate > best


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
        DataLoader for the training split. Explicit three_class mode preserves
        tumour labels; historical liver_only mode folds them into liver.
    val_loader:
        Optional DataLoader for validation Dice computation each epoch.

    Returns
    -------
    dict with keys:
        ``epoch_losses`` (list[float]), ``best_epoch`` (int),
        ``best_metrics`` (dict).
    """
    reject_unmanaged_reconstruction(config)
    warnings.warn(
        "Standalone phase helpers do not provide exact resume, ownership or budgets; "
        "use run_cascaded_training for managed runs.",
        UserWarning,
        stacklevel=2,
    )
    run_name: str = config["pathing"]["run_name"]
    model_save_dir: str = config["pathing"]["model_save_dir"]
    phase_dir = os.path.join(model_save_dir, run_name, "phase_a")
    os.makedirs(phase_dir, exist_ok=True)
    best_path = os.path.join(phase_dir, "best.pt")

    training_cfg = config["training"]
    total_epochs: int = training_cfg.get("phase_a_epochs", 100)
    steps_per_epoch: int = training_cfg.get(
        "phase_a_steps_per_epoch",
        training_cfg.get("steps_per_epoch", DEFAULT_STEPS_PER_EPOCH),
    )
    _validate_phase_schedule(total_epochs, steps_per_epoch, "phase_a")

    model = model.to(device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    epoch_losses: list[float] = []
    best_val_dice: float = float("-inf")
    best_epoch: int = 1
    best_metrics: dict[str, float] = {}
    selected_best = False

    target_mode = phase_a_target_mode(config)
    phase_train_loader = (
        _LiverOnlyLoader(train_loader) if target_mode == "liver_only" else train_loader
    )
    phase_val_loader: Any = (
        _LiverOnlyLoader(val_loader)
        if target_mode == "liver_only" and val_loader is not None
        else val_loader
    )

    for epoch in tqdm(range(1, total_epochs + 1), desc="Phase A", position=0, leave=True):
        epoch_loss = _run_epoch(
            config=config,
            model=model,
            device=device,
            loader=phase_train_loader,
            optimizer=optimizer,
            steps_per_epoch=steps_per_epoch,
        )
        epoch_losses.append(epoch_loss)

        if scheduler is not None:
            scheduler.step()

        # Compute val Dice for best-checkpoint tracking.
        metrics: dict[str, float] = {"train_loss": epoch_loss}
        if phase_val_loader is not None:
            from dense_unet_3d.evaluation.evaluate import evaluate  # lazy import

            model.eval()
            val_metrics = evaluate(model, device, phase_val_loader)
            metrics.update(val_metrics)
            val_dice = (
                foreground_selection_score(val_metrics)
                if target_mode == "three_class"
                else float(val_metrics.get("liver_per_case", 0.0))
            )
        else:
            val_dice = 0.0

        tqdm.write(f"Phase A epoch {epoch}/{total_epochs}: loss={epoch_loss:.4f}")

        # Update best checkpoint (strict >: equal score keeps the earlier epoch).
        if _is_better_score(val_dice, best_val_dice):
            best_val_dice = val_dice
            best_epoch = epoch
            best_metrics = metrics
            selected_best = True
            save_checkpoint(
                path=best_path,
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
            )

    # Save last checkpoint.
    save_checkpoint(
        path=os.path.join(phase_dir, "last.pt"),
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=total_epochs,
        metrics=metrics,  # metrics from the final epoch
    )

    if not selected_best:
        raise ValueError(
            "Phase A produced no finite validation selection score; no best checkpoint was saved."
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
    reject_unmanaged_reconstruction(config)
    warnings.warn(
        "Standalone phase helpers do not provide exact resume, ownership or budgets; "
        "use run_cascaded_training for managed runs.",
        UserWarning,
        stacklevel=2,
    )
    run_name: str = config["pathing"]["run_name"]
    model_save_dir: str = config["pathing"]["model_save_dir"]
    phase_dir = os.path.join(model_save_dir, run_name, "phase_b")
    os.makedirs(phase_dir, exist_ok=True)
    best_path = os.path.join(phase_dir, "best.pt")

    training_cfg = config["training"]
    total_epochs: int = training_cfg.get("phase_b_epochs", 1000)
    steps_per_epoch: int = training_cfg.get(
        "phase_b_steps_per_epoch",
        training_cfg.get("steps_per_epoch", DEFAULT_STEPS_PER_EPOCH),
    )
    _validate_phase_schedule(total_epochs, steps_per_epoch, "phase_b")

    validate_phase_transfer_policy(config)
    # Move model to device first so load_checkpoint maps to the right device.
    model = model.to(device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # RELOAD Phase A best weights — this is the core cascade step.
    phase_a_ckpt = torch.load(phase_a_best_path, map_location=device, weights_only=False)
    loaded_phase_a_state_dict: dict[str, torch.Tensor] = {
        k: v.clone() for k, v in phase_a_ckpt["model_state_dict"].items()
    }
    validate_checkpoint_graph(phase_a_ckpt, model)
    model.load_state_dict(phase_a_ckpt["model_state_dict"])

    epoch_losses: list[float] = []
    best_val_dice: float = float("-inf")
    best_epoch: int = 1
    best_metrics: dict[str, float] = {}
    selected_best = False

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
            val_dice = foreground_selection_score(val_metrics)
        else:
            val_dice = 0.0

        tqdm.write(f"Phase B epoch {epoch}/{total_epochs}: loss={epoch_loss:.4f}")

        # Strict >: equal score keeps the earlier epoch; NaN is never better.
        if _is_better_score(val_dice, best_val_dice):
            best_val_dice = val_dice
            best_epoch = epoch
            best_metrics = metrics
            selected_best = True
            save_checkpoint(
                path=best_path,
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
            )

    # Save last checkpoint.
    save_checkpoint(
        path=os.path.join(phase_dir, "last.pt"),
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=total_epochs,
        metrics=metrics,
    )

    if not selected_best:
        raise ValueError(
            "Phase B produced no finite validation selection score; no best checkpoint was saved."
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
    *,
    phase_b_train_loader: DataLoader | None = None,
    phase_b_val_loader: DataLoader | None = None,
    session: Any = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Full cascaded two-phase training driver.

    Phase A then Phase B sequentially.  Phase B automatically reloads the
    Phase A best checkpoint before training begins.

    Parameters
    ----------
    config:
        Run configuration dict.
    model:
        Model to train. Phase A trains this instance; Phase B reloads Phase A
        best weights into the same instance before training.
    device:
        CPU or CUDA device.
    train_loader:
        DataLoader for training split.
    val_loader:
        Optional Phase A validation DataLoader. Tumour labels are folded into
        liver labels for Phase A training and validation.
    phase_b_train_loader:
        Optional Phase B training DataLoader. Defaults to ``train_loader``;
        its tumour labels are preserved.
    phase_b_val_loader:
        Optional Phase B validation DataLoader. Defaults to ``val_loader``;
        its tumour labels are preserved.

    Returns
    -------
    dict with keys:
        ``phase_a`` (phase A result dict),
        ``phase_b`` (phase B result dict),
        ``phase_b_loaded_phase_a_state_dict`` (the weights loaded into Phase B,
        for proof assertions in tests).
    """
    from dense_unet_3d.training.recovery import run_recoverable
    from dense_unet_3d.training.runtime import RunSession

    loaders = [
        train_loader,
        val_loader,
        phase_b_train_loader if phase_b_train_loader is not None else train_loader,
        phase_b_val_loader if phase_b_val_loader is not None else val_loader,
    ]
    if session is not None:
        return run_recoverable(config, model, device, loaders, session)
    with RunSession(config, resume=resume) as active:
        result = run_recoverable(config, model, device, loaders, active)
        active.finish(result["terminal_reason"])
        return result
