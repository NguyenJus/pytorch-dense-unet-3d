"""Weighted cross-entropy loss for 3D-DenseUNet-569.

Paper: Alalwan et al. (2021), §2.
Class weights: background=0.2, liver=1.2, lesion=2.2.

Usage
-----
criterion = get_criterion(logits)   # weight tensor moved to logits.device
loss = criterion(logits, target)    # target: (N,D,H,W) long tensor

Design note
-----------
The criterion is a plain nn.CrossEntropyLoss with a weight= tensor.
The weight tensor is moved to the input device in get_criterion() so
the caller never needs to .to() the criterion object itself.  This is
the standard pattern; .to(criterion) is non-standard and was a confirmed
bug in the original training loop (§4).
"""

import torch
import torch.nn as nn

__all__ = ["CLASS_WEIGHTS", "get_criterion"]

# [background, liver, lesion] — order matches class indices 0 / 1 / 2
CLASS_WEIGHTS: torch.Tensor = torch.tensor([0.2, 1.2, 2.2], dtype=torch.float32)


def get_criterion(input_tensor: torch.Tensor) -> nn.CrossEntropyLoss:
    """Return a CrossEntropyLoss with class weights on the input tensor's device.

    Parameters
    ----------
    input_tensor:
        Any tensor already on the target device (typically the model logits).
        Used only to infer the device; not consumed by this function.

    Returns
    -------
    nn.CrossEntropyLoss
        Criterion with weight tensor on the same device as *input_tensor*.
        Do NOT call .to(device) on the returned criterion.
    """
    weight = CLASS_WEIGHTS.to(input_tensor.device)
    return nn.CrossEntropyLoss(weight=weight)
