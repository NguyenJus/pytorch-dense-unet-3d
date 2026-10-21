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
    pad_tile,
    slab_starts,
    spatial_config,
    tile_bounds,
    tile_slices,
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
    """Native HWD uint8 labels; at most one window of probabilities on ``device``.

    Accepts image geometry only. The image is read once; windows are sliced
    from memory and forwarded in batches of ``inference_batch``. Probabilities
    are averaged over equal window coverage, restored to native in-plane grid,
    then converted to labels. Native tiles use exact source indices and
    per-pixel weights without resize. Padding on every axis is excluded.
    Output affine is the input image affine.
    """
    if len(input_img.shape) != 3 or min(input_img.shape) < 1:
        raise ValueError("inference input must be a nonempty 3-D image")
    geometry = spatial_config(dataset_config)
    batch_size = dataset_config.get("inference_batch", 8)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("inference_batch must be a positive integer")
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
            output[:, :, z] = native.argmax(dim=0).to(torch.uint8).cpu().numpy()
        next_slice = stop

    check_stop()
    array = np.asarray(input_img.dataobj, dtype=np.float32)
    if not np.isfinite(array).all():
        kind = "tile" if native_tiles else "slab"
        raise ValueError(f"image {kind} contains nonfinite values")
    volume = torch.from_numpy(array).permute(2, 0, 1)
    if geometry["clamp_hu"]:
        volume = volume.clamp(geometry["hu_low"], geometry["hu_high"])

    def window(start: tuple[int, int, int], valid: tuple[int, int, int]) -> torch.Tensor:
        """Fixed CDHW model input, equal to image_tile / image_slab on the same source."""
        if native_tiles:
            rows, cols, slices = tile_slices(start, valid)
            tile: torch.Tensor = pad_tile(
                volume[None, slices, rows, cols], valid, geometry, geometry["hu_low"]
            )
            return tile
        slab = F.interpolate(
            volume[None, None, start[0] : start[0] + valid[0]],
            size=(valid[0], geometry["height"], geometry["width"]),
            mode="trilinear",
            align_corners=False,
        )[0]
        return F.pad(slab, (0, 0, 0, 0, 0, geometry["window"] - valid[0]), value=geometry["hu_low"])

    previous_start = -1

    def forward(pending: list[tuple[tuple[int, int, int], tuple[int, int, int], torch.Tensor]]):
        nonlocal previous_start
        logits = model(torch.stack([image for *_, image in pending]).to(device, torch.float32))
        expected = (len(pending), 3, geometry["window"], geometry["height"], geometry["width"])
        if tuple(logits.shape) != expected:
            raise ValueError(
                f"prediction logits shape {tuple(logits.shape)} differs from {expected}"
            )
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite prediction logits")
        probabilities = logits.softmax(dim=1)
        for index, ((start, h, w), (valid_depth, valid_height, valid_width), _) in enumerate(
            pending
        ):
            if start != previous_start:
                flush(start)
                previous_start = start
            tile = probabilities[index, :, :valid_depth, :valid_height, :valid_width]
            for local_z in range(valid_depth):
                z = start + local_z
                if z not in active:
                    ph, pw = (
                        (height, width) if native_tiles else (geometry["height"], geometry["width"])
                    )
                    active[z] = (
                        torch.zeros((3, ph, pw), device=device),
                        torch.zeros((ph, pw), dtype=torch.int32, device=device),
                    )
                total, coverage = active[z]
                total[:, h : h + valid_height, w : w + valid_width] += tile[:, local_z]
                coverage[h : h + valid_height, w : w + valid_width] += 1

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
        pending: list[tuple[tuple[int, int, int], tuple[int, int, int], torch.Tensor]] = []
        for start, valid in bounds:
            check_stop()
            pending.append((start, valid, window(start, valid)))
            if len(pending) == batch_size:
                forward(pending)
                pending = []
        if pending:
            forward(pending)
        flush(depth)
    if active or next_slice != depth:
        raise ValueError("case reconstruction did not finish all native slices")
    return output
