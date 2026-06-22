"""
Evaluation on the validation split.

Computes per-case and global Dice for liver and tumor using the val_loader.
Returns a structured dict so callers can log or display the metrics cleanly.

Liver convention (folded, issue #6)
------------------------------------
``liver_*`` metrics use **folded** liver masks: liver = liver ∪ tumor.
- GT liver mask:   ``target >= 1``
- Pred liver mask: ``argmax(logits) >= 1``

This means both true liver (class 1) and tumor (class 2) voxels are counted
as "liver" for the purpose of the liver metric, which matches the clinical
notion that tumor lies inside the liver.  Reference: issue #6.

Tumor convention (strict)
--------------------------
``tumor_*`` metrics use strict class-2 masks: ``target == 2``,
``argmax(logits) == 2``.  The fold does not affect tumor Dice.

Per-case convention (presence-aware, issue #4)
----------------------------------------------
A volume counts toward a metric's per-case mean only when the relevant class
is present in GT **OR** prediction for that volume.  Both-empty volumes are
excluded.  If a class is absent from every volume (GT and pred both empty
everywhere), its per-case score is ``float('nan')`` — an explicit "no
evidence" signal.

Global convention (unchanged)
------------------------------
Intersection and union are accumulated across all voxels; the empty-class
convention (denom == 0 → 1.0) applies to the global path only.

Usage
-----
    result = evaluate(model, device, val_loader)
    # result keys: liver_per_case, liver_global, tumor_per_case, tumor_global
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dense_unet_3d.evaluation.dice_score import _binary_dice

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

    # Streaming aggregates — never hold all batches' full logits at once.
    #
    # Two separate binary metrics are accumulated:
    #
    # Liver (folded, issue #6):
    #   pred_liver = argmax(logits) >= 1  (liver ∪ tumor in prediction)
    #   gt_liver   = target >= 1          (liver ∪ tumor in ground truth)
    #   Presence-aware per-case (§2.4): include volume when gt_liver.any() OR
    #   pred_liver.any(). Finalise as liver_pc_sum / liver_pc_count if count>0,
    #   else nan (§2.5).
    #
    # Tumor (strict class 2):
    #   pred_tumor = argmax(logits) == 2
    #   gt_tumor   = target == 2
    #   Presence-aware per-case (§2.4/§2.5) — identical logic, strict class 2.
    #
    # Global path (unchanged): accumulate intersection and denom across all voxels;
    #   empty-class convention (denom == 0 → 1.0) applied at finalisation.
    liver_pc_sum: float = 0.0
    liver_pc_count: int = 0
    tumor_pc_sum: float = 0.0
    tumor_pc_count: int = 0

    liver_gl_inter: float = 0.0
    liver_gl_denom: float = 0.0
    tumor_gl_inter: float = 0.0
    tumor_gl_denom: float = 0.0

    n_volumes = 0

    with torch.no_grad():
        for volume, target in val_loader:
            volume = volume.to(device, dtype=torch.float32)
            # target shape: (N, 1, D, H, W) or (N, D, H, W)
            if target.dim() == 5:
                target = target.squeeze(1)  # → (N, D, H, W)
            target = target.to(device, dtype=torch.long)

            logits = model(volume).cpu()  # (N, 3, D, H, W)
            target = target.cpu()

            hard_pred = logits.argmax(dim=1)  # (N, D, H, W)

            # Folded liver masks: liver ∪ tumor (issue #6)
            pred_liver_batch = (hard_pred >= 1).float()  # (N, D, H, W)
            gt_liver_batch = (target >= 1).float()        # (N, D, H, W)

            # Strict tumor masks: class 2 only
            pred_tumor_batch = (hard_pred == 2).float()  # (N, D, H, W)
            gt_tumor_batch = (target == 2).float()        # (N, D, H, W)

            n = logits.shape[0]
            n_volumes += n

            for i in range(n):
                # Liver (folded) — per-volume presence-aware
                pl = pred_liver_batch[i]
                gl_ = gt_liver_batch[i]
                if pl.any() or gl_.any():
                    liver_pc_sum += _binary_dice(pl, gl_)
                    liver_pc_count += 1

                # Tumor (strict) — per-volume presence-aware
                pt = pred_tumor_batch[i]
                gt_ = gt_tumor_batch[i]
                if pt.any() or gt_.any():
                    tumor_pc_sum += _binary_dice(pt, gt_)
                    tumor_pc_count += 1

            # Global accumulation across all voxels in the batch
            liver_gl_inter += (pred_liver_batch * gt_liver_batch).sum().item()
            liver_gl_denom += pred_liver_batch.sum().item() + gt_liver_batch.sum().item()
            tumor_gl_inter += (pred_tumor_batch * gt_tumor_batch).sum().item()
            tumor_gl_denom += pred_tumor_batch.sum().item() + gt_tumor_batch.sum().item()

    if n_volumes == 0:
        raise ValueError(
            "val_loader yielded no batches — cannot compute Dice metrics on an "
            "empty validation set.  Check that the val DataLoader is non-empty."
        )

    # Finalise per-case: nan when a class is absent from every volume (§2.5)
    liver_pc = liver_pc_sum / liver_pc_count if liver_pc_count > 0 else float("nan")
    tumor_pc = tumor_pc_sum / tumor_pc_count if tumor_pc_count > 0 else float("nan")

    # Finalise global: empty-class convention (denom == 0 → 1.0)
    liver_gl = (2.0 * liver_gl_inter / liver_gl_denom) if liver_gl_denom > 0 else 1.0
    tumor_gl = (2.0 * tumor_gl_inter / tumor_gl_denom) if tumor_gl_denom > 0 else 1.0

    return {
        "liver_per_case": float(liver_pc),
        "liver_global": float(liver_gl),
        "tumor_per_case": float(tumor_pc),
        "tumor_global": float(tumor_gl),
    }
