"""
Evaluation on the validation split.

Computes per-case and global Dice for liver (class 1) and tumor (class 2)
using the val_loader.  Returns a structured dict so callers can log or
display the metrics cleanly.

Usage
-----
    result = evaluate(model, device, val_loader)
    # result keys: liver_per_case, liver_global, tumor_per_case, tumor_global
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dense_unet_3d.evaluation.dice_score import dice_global, dice_per_case

NUM_CLASSES = 3  # 0=background, 1=liver, 2=tumor


def evaluate(
    model: nn.Module,
    device: torch.device,
    val_loader: DataLoader,
) -> dict[str, float]:
    """
    Evaluate *model* on the validation split and return Dice metrics.

    Computes two Dice aggregations for liver (class 1) and tumor (class 2):
    - **per-case**: mean of per-volume Dice over all cases in val_loader.
    - **global**: intersection / union aggregated across all voxels and all
      volumes (matches the global Dice reported in Alalwan et al. 2021).

    Parameters
    ----------
    model :
        A ``torch.nn.Module`` producing logits of shape ``(N, 3, D, H, W)``.
    device :
        The device on which to run inference (CPU or CUDA).
    val_loader :
        DataLoader over the *validation* split.  Each batch is a 2-tuple
        ``(volume, target)`` where ``volume`` is float ``(N, 1, D, H, W)``
        and ``target`` is integer ``(N, 1, D, H, W)`` with labels in {0,1,2}.

    Returns
    -------
    dict with keys:
        ``liver_per_case``, ``liver_global``, ``tumor_per_case``, ``tumor_global``
        — all ``float`` values in [0, 1].

    Raises
    ------
    ValueError
        If val_loader yields no batches (empty dataset).
    """
    model.eval()

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for volume, target in val_loader:
            volume = volume.to(device, dtype=torch.float32)
            # target shape: (N, 1, D, H, W) or (N, D, H, W)
            if target.dim() == 5:
                target = target.squeeze(1)  # → (N, D, H, W)
            target = target.to(device, dtype=torch.long)

            logits = model(volume)  # (N, 3, D, H, W)

            all_preds.append(logits.cpu())
            all_targets.append(target.cpu())

    if not all_preds:
        raise ValueError(
            "val_loader yielded no batches — cannot compute Dice metrics on an "
            "empty validation set.  Check that the val DataLoader is non-empty."
        )

    preds_cat = torch.cat(all_preds, dim=0)  # (N_total, 3, D, H, W)
    targets_cat = torch.cat(all_targets, dim=0)  # (N_total, D, H, W)

    pc = dice_per_case(preds_cat, targets_cat, num_classes=NUM_CLASSES)
    gl = dice_global(preds_cat, targets_cat, num_classes=NUM_CLASSES)

    return {
        "liver_per_case": float(pc[1].item()),
        "liver_global": float(gl[1].item()),
        "tumor_per_case": float(pc[2].item()),
        "tumor_global": float(gl[2].item()),
    }
