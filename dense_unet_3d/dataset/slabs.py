"""Explicit engineering slab candidate; native depth, full-FOV in-plane resize."""

from __future__ import annotations

import math
from typing import Any, cast

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.spatialimages import SpatialImage
from torch.utils.data import Dataset

from dense_unet_3d.dataset.LITSDataset import _case_id, discover_pairs, validate_pair


def slab_starts(depth: int, window: int = 12) -> list[int]:
    """Cover every depth slice, with one tail overlap and no label selection."""
    if depth < 1 or window < 1:
        raise ValueError("depth and window must be positive")
    if depth <= window:
        return [0]
    starts = list(range(0, depth - window + 1, window))
    if starts[-1] != depth - window:
        starts.append(depth - window)
    return starts


def spatial_config(config: dict) -> dict[str, Any]:
    dims = config.get("resize_dims", {"D": 12, "H": 224, "W": 224})
    if not isinstance(dims, dict) or any(type(dims.get(k)) is not int for k in ("D", "H", "W")):
        raise ValueError("native_slabs resize dimensions must be integers")
    window, height, width = (dims[k] for k in ("D", "H", "W"))
    if window < 1 or height < 1 or width < 1:
        raise ValueError("slab model dimensions must be positive")
    if config.get("resize_img", True) is not True:
        raise ValueError("native_slabs full_fov_resize requires resize_img=true")
    random_hflip = config.get("random_hflip", False)
    scale_img = config.get("scale_img", False)
    if type(random_hflip) is not bool or type(scale_img) is not bool:
        raise ValueError("native_slabs augmentation flags must be booleans")
    if random_hflip or scale_img:
        raise ValueError(
            "native_slabs augmentation is unresolved; explicitly disable random_hflip and scale_img"
        )
    if config.get("inplane_representation", "full_fov_resize") != "full_fov_resize":
        raise ValueError("only the gated full_fov_resize candidate is implemented")
    clamp_hu = config.get("clamp_hu", True)
    if type(clamp_hu) is not bool:
        raise ValueError("native_slabs clamp_hu must be a boolean")
    hu = config.get("clamp_hu_range", {"min": -200, "max": 250})
    if not isinstance(hu, dict) or any(
        isinstance(hu.get(k), bool) or not isinstance(hu.get(k), (int, float))
        for k in ("min", "max")
    ):
        raise ValueError("HU clipping bounds must be finite numbers")
    low, high = float(hu["min"]), float(hu["max"])
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError("HU clipping bounds must be finite and increasing")
    return {
        "window": window,
        "height": height,
        "width": width,
        "clamp_hu": clamp_hu,
        "hu_low": low,
        "hu_high": high,
        "representation": "full_fov_resize",
        "padding_target": -100,
        "metric_convention": "medpy022_empty_zero",
        "overlap": "uniform_probability_mean",
        "image_interpolation": "trilinear_half_pixel",
        "mask_interpolation": "nearest_exact_half_pixel",
        "sampling": "native_slabs",
        "augmentation": "disabled_unresolved",
    }


def model_to_source(shape: tuple[int, ...], geometry: dict, start: int) -> np.ndarray:
    """Map model (D,H,W) indices to original NIfTI array indices (H,W,D)."""
    qh, qw = shape[0] / geometry["height"], shape[1] / geometry["width"]
    return np.array(
        [[0, qh, 0, (qh - 1) / 2], [0, 0, qw, (qw - 1) / 2], [1, 0, 0, start], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def image_slab(image: SpatialImage, start: int, geometry: dict) -> tuple[torch.Tensor, int]:
    """Read a bounded proxy slice; return fixed CDHW tensor and valid depth."""
    depth = min(geometry["window"], image.shape[2] - start)
    array = np.asarray(image.dataobj[:, :, start : start + depth], dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("image slab contains nonfinite values")
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)[None, None]
    if geometry["clamp_hu"]:
        tensor = tensor.clamp(geometry["hu_low"], geometry["hu_high"])
    tensor = F.interpolate(
        tensor,
        size=(depth, geometry["height"], geometry["width"]),
        mode="trilinear",
        align_corners=False,
    )[0]
    if depth < geometry["window"]:
        tensor = F.pad(
            tensor, (0, 0, 0, 0, 0, geometry["window"] - depth), value=geometry["hu_low"]
        )
    return tensor, depth


class NativeSlabDataset(Dataset):
    """Stable sample index and explicit source/model physical coordinate maps.

    Full-FOV resize remains a gated candidate, not an established paper recipe.
    The source axes are NIfTI array axes; no anatomical orientation is assumed.
    """

    def __init__(self, img_dirs: list[str], config: dict, *, detect_tumors: bool = True):
        if not detect_tumors:
            raise ValueError("native_slabs requires the three-class target contract")
        self.geometry = spatial_config(config)
        self.detect_tumors = detect_tumors
        self.crop_to_liver = False
        self.cases = discover_pairs(img_dirs)
        self.volume_img_paths = [pair[0] for pair in self.cases]
        self.segmentation_img_paths = [pair[1] for pair in self.cases]
        self.case_geometry: list[dict] = []
        self.sample_index: list[tuple[int, int, int]] = []
        for case_index, (volume_path, mask_path) in enumerate(self.cases):
            validate_pair(volume_path, mask_path)
            image = cast(SpatialImage, nib.load(volume_path))
            shape = tuple(int(v) for v in image.shape)
            self.case_geometry.append({"shape": shape, "affine": image.affine.tolist()})
            for start in slab_starts(shape[2], self.geometry["window"]):
                self.sample_index.append(
                    (case_index, start, min(self.geometry["window"], shape[2] - start))
                )

    def manifest(self) -> dict:
        return {
            "version": 1,
            "geometry": self.geometry,
            "detect_tumors": self.detect_tumors,
            "cases": [
                {"image": image, "target": target, "case_id": _case_id(image, "volume"), **geometry}
                for (image, target), geometry in zip(self.cases, self.case_geometry, strict=True)
            ],
            "samples": self.sample_index,
        }

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case_index, start, valid_depth = self.sample_index[index]
        volume_path, mask_path = self.cases[case_index]
        validate_pair(volume_path, mask_path)
        image = cast(SpatialImage, nib.load(volume_path))
        target_img = cast(SpatialImage, nib.load(mask_path))
        tensor, depth = image_slab(image, start, self.geometry)
        if (
            depth != valid_depth
            or tuple(image.shape) != self.case_geometry[case_index]["shape"]
            or image.affine.tolist() != self.case_geometry[case_index]["affine"]
        ):
            raise ValueError("source geometry changed since sample indexing")
        array = np.asarray(target_img.dataobj[:, :, start : start + depth])
        if not np.isfinite(array).all() or not np.isin(array, [0, 1, 2]).all():
            raise ValueError("target slab labels must be finite integers in {0,1,2}")
        target = torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1)[None, None]
        target = F.interpolate(
            target,
            size=(depth, self.geometry["height"], self.geometry["width"]),
            mode="nearest-exact",
        )[0].long()
        if not self.detect_tumors:
            target = target.clamp_max(1)
        target = F.pad(target, (0, 0, 0, 0, 0, self.geometry["window"] - depth), value=-100)
        valid = target != -100
        mapping = model_to_source(tuple(image.shape), self.geometry, start)
        return {
            "image": tensor,
            "target": target,
            "valid_mask": valid,
            "case_id": _case_id(volume_path, "volume"),
            "case_index": case_index,
            "start": start,
            "valid_depth": depth,
            "source_shape": torch.tensor(image.shape, dtype=torch.int64),
            "source_affine": torch.tensor(image.affine, dtype=torch.float64),
            "model_to_source": torch.tensor(mapping, dtype=torch.float64),
            "model_to_world": torch.tensor(image.affine @ mapping, dtype=torch.float64),
        }
