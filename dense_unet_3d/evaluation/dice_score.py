"""
Dice score metrics for 3D medical image segmentation.

Formula
-------
    Dice(A, B) = 2 |A ∩ B| / (|A| + |B|)

Two aggregation modes
---------------------
dice_per_case  (presence-aware)
    Compute Dice independently for every (volume, class) pair, but only
    include a volume in class c's average when class c is **present in GT
    OR prediction** for that volume (``gt_mask.any() or pred_mask.any()``).

    Rationale: a volume where a class is absent from both the ground-truth
    and the prediction carries no information about that class.  Including
    it and scoring it 1.0 (the global empty-class convention) would silently
    inflate the per-case metric — a model that never predicts tumor could
    appear perfect if most validation volumes contain no tumor.

    NaN-for-absent-class: if a class is absent from ALL volumes in the
    batch (neither GT nor prediction contains it anywhere), the denominator
    is zero and the per-case score is defined as ``float('nan')``.  This is
    an explicit "no evidence" signal; downstream callers must handle it
    deliberately (e.g. via ``numpy.nanmean``).

    Specifically: hallucination (empty-GT, non-empty-pred) and miss
    (non-empty-GT, empty-pred) both count and contribute Dice = 0.0.

dice_global  (unchanged — voxel-pooled, empty→1.0)
    Accumulate intersection and union counts across **all** voxels in the
    entire dataset, then compute the single ratio.  Volumes with many voxels
    dominate.  This matches the "global Dice" reported in Alalwan et al. (2021).
    The empty-class convention (returns 1.0 when global denom is 0) is retained
    for the global path.

The two measures can—and often will—differ on multi-case datasets.  See
tests/evaluation/test_dice_score.py for a constructed example that proves this.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _binary_dice(
    pred_mask: Tensor,
    gt_mask: Tensor,
) -> float:
    """
    Compute Dice for a single (class, volume) binary mask pair.

    pred_mask, gt_mask: bool or float tensors of the same shape.

    Returns 1.0 when both masks are empty (empty-class convention).
    """
    intersection = (pred_mask * gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return 1.0
    return (2.0 * intersection / denom).item()


def _to_binary_masks(preds: Tensor, targets: Tensor, num_classes: int) -> tuple[Tensor, Tensor]:
    """
    Convert raw predictions and integer targets to per-class binary masks.

    Parameters
    ----------
    preds : Tensor of shape (N, C, D, H, W) — soft or one-hot predictions.
             Argmax is used to obtain the hard predicted class.
    targets : Tensor of shape (N, D, H, W) — integer class labels.
    num_classes : int

    Returns
    -------
    pred_masks  : Tensor bool (N, C, D, H, W)
    gt_masks    : Tensor bool (N, C, D, H, W)
    """
    n = preds.shape[0]
    hard_pred = preds.argmax(dim=1)  # (N, D, H, W)
    pred_masks = torch.zeros(
        n, num_classes, *preds.shape[2:], dtype=torch.bool, device=preds.device
    )
    gt_masks = torch.zeros(n, num_classes, *preds.shape[2:], dtype=torch.bool, device=preds.device)
    for c in range(num_classes):
        pred_masks[:, c] = hard_pred == c
        gt_masks[:, c] = targets == c
    return pred_masks, gt_masks


def dice_per_case(preds: Tensor, targets: Tensor, num_classes: int) -> Tensor:
    """
    Presence-aware mean per-case Dice score.

    For each class c, only volumes where the class is present in GT OR
    prediction are included in the average.  Both-empty volumes are excluded.
    If a class is absent from every volume, the score is ``float('nan')``.

    Parameters
    ----------
    preds       : Tensor (N, C, D, H, W) — raw logits or probabilities.
    targets     : Tensor (N, D, H, W)    — integer class labels in [0, C-1].
    num_classes : int — number of classes C.

    Returns
    -------
    Tensor of shape (C,) — one mean Dice per class; ``nan`` when a class is
    absent from all volumes (GT and prediction both empty everywhere).
    """
    pred_masks, gt_masks = _to_binary_masks(preds, targets, num_classes)
    n = preds.shape[0]
    scores = torch.full((num_classes,), float("nan"), dtype=torch.float32)
    for c in range(num_classes):
        included_scores: list[float] = []
        for i in range(n):
            pm = pred_masks[i, c].float()
            gm = gt_masks[i, c].float()
            # Presence-aware: include volume only if class present in GT or pred
            if pm.any() or gm.any():
                included_scores.append(_binary_dice(pm, gm))
        if included_scores:
            scores[c] = sum(included_scores) / len(included_scores)
    return scores


def dice_global(preds: Tensor, targets: Tensor, num_classes: int) -> Tensor:
    """
    Global Dice score: intersection and union aggregated over the full dataset.

    Accumulate intersection and |A|+|B| across ALL voxels and ALL volumes for
    each class, then compute the single ratio.  Applies the empty-class
    convention (returns 1.0 when global denom is 0).

    Parameters
    ----------
    preds       : Tensor (N, C, D, H, W)
    targets     : Tensor (N, D, H, W)
    num_classes : int

    Returns
    -------
    Tensor of shape (C,) — one global Dice per class.
    """
    pred_masks, gt_masks = _to_binary_masks(preds, targets, num_classes)
    scores = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        intersection = (pred_masks[:, c].float() * gt_masks[:, c].float()).sum()
        denom = pred_masks[:, c].float().sum() + gt_masks[:, c].float().sum()
        if denom == 0:
            scores[c] = 1.0
        else:
            scores[c] = (2.0 * intersection / denom).item()
    return scores
