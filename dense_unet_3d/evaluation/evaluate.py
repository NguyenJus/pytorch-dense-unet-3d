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

from dense_unet_3d.evaluation.dice_score import _binary_dice, _to_binary_masks

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
    # per_case: accumulate the sum of per-volume binary-Dice scalars and the
    #   volume count, then finalise as sum / n_volumes (mirrors dice_per_case).
    # global: accumulate per-class intersection and denominator (|A|+|B|) voxel
    #   counts across every volume, then finalise the single ratio at the end
    #   with the empty-class convention (denom == 0 → 1.0) (mirrors dice_global).
    pc_sum = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    gl_inter = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    gl_denom = torch.zeros(NUM_CLASSES, dtype=torch.float64)
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

            pred_masks, gt_masks = _to_binary_masks(logits, target, NUM_CLASSES)
            n = logits.shape[0]
            n_volumes += n

            for c in range(NUM_CLASSES):
                for i in range(n):
                    # per-case scalar (applies empty-class convention internally)
                    pc_sum[c] += _binary_dice(pred_masks[i, c].float(), gt_masks[i, c].float())
                # global counts accumulated across all voxels/volumes
                pred_c = pred_masks[:, c].float()
                gt_c = gt_masks[:, c].float()
                gl_inter[c] += (pred_c * gt_c).sum().item()
                gl_denom[c] += pred_c.sum().item() + gt_c.sum().item()

    if n_volumes == 0:
        raise ValueError(
            "val_loader yielded no batches — cannot compute Dice metrics on an "
            "empty validation set.  Check that the val DataLoader is non-empty."
        )

    pc = pc_sum / n_volumes
    gl = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    for c in range(NUM_CLASSES):
        if gl_denom[c] == 0:
            gl[c] = 1.0
        else:
            gl[c] = 2.0 * gl_inter[c] / gl_denom[c]

    return {
        "liver_per_case": float(pc[1].item()),
        "liver_global": float(gl[1].item()),
        "tumor_per_case": float(pc[2].item()),
        "tumor_global": float(gl[2].item()),
    }
