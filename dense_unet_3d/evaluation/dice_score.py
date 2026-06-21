"""
Dice score metrics for 3D medical image segmentation.

Formula
-------
    Dice(A, B) = 2 |A ∩ B| / (|A| + |B|)

Empty-class convention
----------------------
When BOTH the prediction mask (A) and the ground-truth mask (B) are empty for a
given class in a given volume, Dice is defined as **1.0** for that class/volume
pair.  This reflects the interpretation that a model that correctly predicts the
absence of a class should not be penalised.  The convention is applied
per-class, per-volume before any aggregation step.

Two aggregation modes
---------------------
dice_per_case
    Compute Dice independently for every (volume, class) pair, then average
    across volumes.  A class that is absent from a single volume but present in
    others still contributes its per-case score to the mean.

dice_global
    Accumulate intersection and union counts across **all** voxels in the
    entire dataset, then compute the single ratio.  Volumes with many voxels
    dominate.  This matches the "global Dice" reported in Alalwan et al. (2021).

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
    Mean per-case Dice score, averaged over volumes.

    For each volume n and class c, compute Dice(pred_n_c, gt_n_c) applying the
    empty-class convention (Dice=1.0 when both masks are empty).  Then average
    over the N volumes.

    Parameters
    ----------
    preds       : Tensor (N, C, D, H, W) — raw logits or probabilities.
    targets     : Tensor (N, D, H, W)    — integer class labels in [0, C-1].
    num_classes : int — number of classes C.

    Returns
    -------
    Tensor of shape (C,) — one mean Dice per class.
    """
    pred_masks, gt_masks = _to_binary_masks(preds, targets, num_classes)
    n = preds.shape[0]
    scores = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        class_scores = [
            _binary_dice(pred_masks[i, c].float(), gt_masks[i, c].float()) for i in range(n)
        ]
        scores[c] = sum(class_scores) / n
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
