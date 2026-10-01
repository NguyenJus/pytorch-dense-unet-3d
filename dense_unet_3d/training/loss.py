"""Explicit weighted CE reductions for historical and Eq. (2) experiments.

``weighted_mean`` preserves the historical PyTorch target-weight denominator.
``valid_voxel_mean`` divides weighted CE by the count of non-padding voxels.
Padding is represented by ignore index -100 in both reductions.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CLASS_WEIGHTS", "IGNORE_INDEX", "WeightedVoxelCrossEntropy", "get_criterion"]

CLASS_WEIGHTS: torch.Tensor = torch.tensor([0.2, 1.2, 2.2], dtype=torch.float32)
IGNORE_INDEX = -100


class WeightedVoxelCrossEntropy(nn.CrossEntropyLoss):
    """Class-index CE with an explicit denominator and rejected empty targets."""

    def __init__(self, weight: torch.Tensor, loss_reduction: str) -> None:
        if loss_reduction not in {"weighted_mean", "valid_voxel_mean"}:
            raise ValueError(f"Unsupported loss_reduction: {loss_reduction!r}")
        super().__init__(weight=weight, ignore_index=IGNORE_INDEX)
        self.loss_reduction = loss_reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid_count = (target != self.ignore_index).sum()
        if valid_count.item() == 0:
            raise ValueError("Training batch contains no valid target voxels")
        if self.loss_reduction == "weighted_mean":
            return super().forward(input, target)
        numerator = F.cross_entropy(
            input, target, weight=self.weight, ignore_index=self.ignore_index, reduction="none"
        ).sum()
        return numerator / valid_count


def get_criterion(
    config: dict[str, Any],
    device: torch.device | str | None = None,
) -> WeightedVoxelCrossEntropy:
    """Build configured CE; omission retains the historical weighted mean."""
    training = config.get("training", {})
    class_weights = training.get("class_weights")
    if class_weights is None:
        weight = CLASS_WEIGHTS.clone()
    else:
        weight = torch.tensor(
            [class_weights["background"], class_weights["liver"], class_weights["lesion"]],
            dtype=torch.float32,
        )
    if not torch.isfinite(weight).all() or not (weight > 0).all():
        raise ValueError("Class weights must be finite and strictly positive")
    if device is not None:
        weight = weight.to(device)
    return WeightedVoxelCrossEntropy(weight, training.get("loss_reduction", "weighted_mean"))
