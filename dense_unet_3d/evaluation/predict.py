"""One predictor shared by case evaluation and NIfTI inference."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from nibabel.spatialimages import SpatialImage
from torch import nn

from dense_unet_3d.dataset.slabs import (
    image_slab,
    image_tile,
    slab_starts,
    spatial_config,
    tile_bounds,
)


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
    Native tiles use exact source indices and per-pixel weights without resize.
    Padding on every axis is excluded. Output affine is the input image affine.
    """
    if len(input_img.shape) != 3 or min(input_img.shape) < 1:
        raise ValueError("inference input must be a nonempty 3-D image")
    geometry = spatial_config(dataset_config)
    height, width, depth = input_img.shape
    output = np.empty((height, width, depth), dtype=np.uint8)
    native_tiles = geometry["representation"] == "native_tiles_v1"
    active: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
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
            if not torch.all(coverage > 0):
                raise ValueError(f"uncovered native pixels on slice {z}")
            native = probability / coverage
            if not native_tiles:
                native = F.interpolate(
                    native[None], size=(height, width), mode="bilinear", align_corners=False
                )[0]
            output[:, :, z] = native.argmax(dim=0).numpy().astype(np.uint8)
        next_slice = stop

    model.eval()
    with torch.no_grad():
        bounds = (
            tile_bounds(tuple(input_img.shape), geometry)
            if native_tiles
            else (
                (
                    (start, 0, 0),
                    (min(geometry["window"], depth - start), geometry["height"], geometry["width"]),
                )
                for start in slab_starts(depth, geometry["window"])
            )
        )
        previous_start = -1
        for (start, h, w), (valid_depth, valid_height, valid_width) in bounds:
            check_stop()
            if start != previous_start:
                flush(start)
                previous_start = start
            if native_tiles:
                image, _ = image_tile(input_img, (start, h, w), geometry)
            else:
                image, _ = image_slab(input_img, start, geometry)
            logits = model(image[None].to(device, dtype=torch.float32))
            expected = (1, 3, geometry["window"], geometry["height"], geometry["width"])
            if tuple(logits.shape) != expected:
                raise ValueError(
                    f"prediction logits shape {tuple(logits.shape)} differs from {expected}"
                )
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite prediction logits")
            probabilities = logits.softmax(dim=1)[
                0, :, :valid_depth, :valid_height, :valid_width
            ].cpu()
            for local_z in range(valid_depth):
                z = start + local_z
                if z not in active:
                    ph, pw = (
                        (height, width) if native_tiles else (geometry["height"], geometry["width"])
                    )
                    active[z] = (torch.zeros((3, ph, pw)), torch.zeros((ph, pw), dtype=torch.int32))
                total, coverage = active[z]
                total[:, h : h + valid_height, w : w + valid_width] += probabilities[:, local_z]
                coverage[h : h + valid_height, w : w + valid_width] += 1
        flush(depth)
    if active or next_slice != depth:
        raise ValueError("case reconstruction did not finish all native slices")
    return output
