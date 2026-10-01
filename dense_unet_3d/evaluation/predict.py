"""One predictor shared by case evaluation and NIfTI inference."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from nibabel.spatialimages import SpatialImage
from torch import nn

from dense_unet_3d.dataset.slabs import image_slab, slab_starts, spatial_config


class PredictionInterrupted(RuntimeError):
    """A prediction did not cover the complete source grid."""


def predict_volume(
    model: nn.Module,
    device: torch.device,
    input_img: SpatialImage,
    dataset_config: dict,
    *,
    stop_requested: Callable[[], str | None] | None = None,
    deadline: float | None = None,
) -> np.ndarray:
    """Native HWD uint8 labels; at most one window of CPU probabilities.

    Accepts image geometry only. Probabilities are averaged over equal window
    coverage, restored to native in-plane grid, then converted to labels.
    Padded depth is never aggregated. Output affine is the input image affine.
    """
    if len(input_img.shape) != 3 or min(input_img.shape) < 1:
        raise ValueError("inference input must be a nonempty 3-D image")
    geometry = spatial_config(dataset_config)
    height, width, depth = input_img.shape
    output = np.empty((height, width, depth), dtype=np.uint8)
    active: dict[int, tuple[torch.Tensor, int]] = {}
    next_slice = 0

    def check_stop() -> None:
        reason = stop_requested() if stop_requested else None
        if reason or (deadline is not None and time.monotonic() >= deadline):
            raise PredictionInterrupted(reason or "prediction wall budget exhausted")

    def flush(stop: int) -> None:
        nonlocal next_slice
        for z in range(next_slice, stop):
            check_stop()
            if z not in active:
                raise ValueError(f"uncovered native slice {z}")
            probability, coverage = active.pop(z)
            native = F.interpolate(
                (probability / coverage)[None],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0]
            output[:, :, z] = native.argmax(dim=0).numpy().astype(np.uint8)
        next_slice = stop

    model.eval()
    with torch.no_grad():
        for start in slab_starts(depth, geometry["window"]):
            check_stop()
            flush(start)
            image, valid_depth = image_slab(input_img, start, geometry)
            logits = model(image[None].to(device, dtype=torch.float32))
            expected = (1, 3, geometry["window"], geometry["height"], geometry["width"])
            if tuple(logits.shape) != expected:
                raise ValueError(
                    f"prediction logits shape {tuple(logits.shape)} differs from {expected}"
                )
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite prediction logits")
            probabilities = logits.softmax(dim=1)[0, :, :valid_depth].cpu()
            for local_z in range(valid_depth):
                z = start + local_z
                plane = probabilities[:, local_z]
                if z in active:
                    total, count = active[z]
                    active[z] = (total + plane, count + 1)
                else:
                    active[z] = (plane.clone(), 1)
        flush(depth)
    if active or next_slice != depth:
        raise ValueError("case reconstruction did not finish all native slices")
    return output
