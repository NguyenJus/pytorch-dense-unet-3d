"""Weighted cross-entropy loss for 3D-DenseUNet-569.

Paper: Alalwan et al. (2021), §2.
Default class weights: background=0.2, liver=1.2, lesion=2.2.

Usage
-----
criterion = get_criterion(config, device=device)  # weight tensor on device
loss = criterion(logits, target)                   # target: (N,D,H,W) long

Design note
-----------
The criterion is a plain nn.CrossEntropyLoss with a weight= tensor.  Class
weights are read from ``config["training"]["class_weights"]`` (keys
``background`` / ``liver`` / ``lesion`` -> class indices 0 / 1 / 2), falling
back to the defaults below when the key is absent.  The weight tensor is moved
to *device* inside get_criterion() so the caller never needs to .to() the
criterion object itself.  This is the standard pattern; .to(criterion) is
non-standard and was a confirmed bug in the original training loop (§4).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

__all__ = ["CLASS_WEIGHTS", "get_criterion"]

# [background, liver, lesion] — order matches class indices 0 / 1 / 2
CLASS_WEIGHTS: torch.Tensor = torch.tensor([0.2, 1.2, 2.2], dtype=torch.float32)


def get_criterion(
    config: dict[str, Any],
    device: torch.device | str | None = None,
) -> nn.CrossEntropyLoss:
    """Return a CrossEntropyLoss with config-driven class weights on *device*.

    Parameters
    ----------
    config:
        Run configuration dict.  Class weights are read from
        ``config["training"]["class_weights"]`` with keys ``background`` /
        ``liver`` / ``lesion`` mapped to class indices 0 / 1 / 2.  When that
        key is absent, the module-level defaults ``[0.2, 1.2, 2.2]`` are used.
    device:
        Target device for the weight tensor (typically the model/logits
        device).  Defaults to CPU.  The criterion module itself is never
        .to()-moved — only the weight tensor.

    Returns
    -------
    nn.CrossEntropyLoss
        Criterion with weight tensor on *device*.  Do NOT call .to(device) on
        the returned criterion.
    """
    class_weights = config.get("training", {}).get("class_weights")
    if class_weights is None:
        weight = CLASS_WEIGHTS.clone()
    else:
        weight = torch.tensor(
            [
                class_weights["background"],
                class_weights["liver"],
                class_weights["lesion"],
            ],
            dtype=torch.float32,
        )

    if device is not None:
        weight = weight.to(device)

    return nn.CrossEntropyLoss(weight=weight)
