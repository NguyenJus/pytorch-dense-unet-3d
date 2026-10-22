"""Foreground-centred random training crops (``foreground_crops_v1``).

Randomness is transported by ``CropSampler``: the main process draws the permutation
and one seed per item, and ``crop_parameters`` derives centre, scale and flip from that
seed alone, so worker count and scheduling never change the crop sequence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.spatialimages import SpatialImage
from torch.utils.data import Dataset, Sampler

from dense_unet_3d.dataset.LITSDataset import _case_id, discover_pairs, validate_pair
from dense_unet_3d.dataset.slabs import pad_tile, spatial_config, tile_slices


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a probability in [0, 1]")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def crop_config(dataset_config: dict) -> dict[str, Any]:
    """Validate and return the training-crop geometry recorded in the run identity."""
    if dataset_config.get("training_representation") != "foreground_crops_v1":
        raise ValueError("crop training requires training_representation=foreground_crops_v1")
    spatial = spatial_config(dataset_config)
    seed = dataset_config.get("sampling_seed")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("sampling_seed must be a non-negative 63-bit integer")
    scale_range = [1.0, 1.0]
    # spatial_config has already type-checked both augmentation flags.
    if dataset_config.get("scale_img", False):
        bounds = dataset_config.get("scale_img_range")
        if not isinstance(bounds, dict) or any(
            isinstance(bounds.get(k), bool) or not isinstance(bounds.get(k), (int, float))
            for k in ("min", "max")
        ):
            raise ValueError("scale_img_range must hold numeric min and max")
        low, high = float(bounds["min"]), float(bounds["max"])
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
            raise ValueError("scale_img_range must be finite, positive and ordered")
        scale_range = [low, high]
    hflip = 0.0
    if dataset_config.get("random_hflip", False):
        hflip = _probability(
            dataset_config.get("random_hflip_probability"), "random_hflip_probability"
        )
    return {
        "representation": "foreground_crops_v1",
        "window": spatial["window"],
        "height": spatial["height"],
        "width": spatial["width"],
        "clamp_hu": spatial["clamp_hu"],
        "hu_low": spatial["hu_low"],
        "hu_high": spatial["hu_high"],
        "padding_target": spatial["padding_target"],
        "passes_per_epoch": _positive_int(
            dataset_config.get("passes_per_epoch"), "passes_per_epoch"
        ),
        "tumor_centre_probability": _probability(
            dataset_config.get("tumor_centre_probability"), "tumor_centre_probability"
        ),
        "foreground_voxel_cache": _positive_int(
            dataset_config.get("foreground_voxel_cache"), "foreground_voxel_cache"
        ),
        "scale_range": scale_range,
        "hflip_probability": hflip,
        "sampling_seed": seed,
        "image_interpolation": "trilinear_half_pixel",
        "mask_interpolation": "nearest_exact_half_pixel",
    }


class CropSampler(Sampler[tuple[int, int]]):
    """Yield ``(index, item_seed)``; the whole epoch is drawn when iteration starts.

    The generator state at an epoch boundary therefore fixes the next epoch's order
    and crop parameters, independent of how many items workers have prefetched.
    """

    def __init__(self, length: int, generator: torch.Generator):
        if type(length) is not int or length < 1:
            raise ValueError("CropSampler length must be a positive integer")
        self.length = length
        self.generator = generator

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[tuple[int, int]]:
        order = torch.randperm(self.length, generator=self.generator).tolist()
        seeds = torch.randint(
            0, 2**63 - 1, (self.length,), generator=self.generator, dtype=torch.int64
        ).tolist()
        return iter(list(zip(order, seeds, strict=True)))


def _subsample(coordinates: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(coordinates) > limit:
        coordinates = coordinates[np.sort(rng.choice(len(coordinates), limit, replace=False))]
    return coordinates.astype(np.int32)


def _bounding_box(coordinates: np.ndarray) -> np.ndarray:
    if len(coordinates) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    return np.stack([coordinates.min(0), coordinates.max(0)]).astype(np.int32)


class ForegroundCropDataset(Dataset):
    """One foreground-centred crop per item; item ``index`` uses case ``index % n_cases``.

    Items are keyed by ``(index, item_seed)`` from ``CropSampler``. Liver means labels
    >= 1 (the evaluation convention); tumor means label 2. Source axes are NIfTI H,W,D.
    """

    def __init__(
        self,
        img_dirs: list[str],
        dataset_config: dict,
        *,
        cache_path: Path | None = None,
        detect_tumors: bool = True,
    ):
        if not detect_tumors:
            raise ValueError("foreground_crops_v1 requires the three-class target contract")
        self.geometry = crop_config(dataset_config)
        self.detect_tumors = detect_tumors
        self.cases = discover_pairs(img_dirs)
        if not self.cases:
            raise ValueError("foreground_crops_v1 found no training cases")
        self.case_geometry: list[dict] = []
        for volume_path, mask_path in self.cases:
            validate_pair(volume_path, mask_path)
            image = cast(SpatialImage, nib.load(volume_path))
            shape = tuple(int(v) for v in image.shape)
            self.case_geometry.append({"shape": shape, "affine": image.affine.tolist()})
        from dense_unet_3d.training.recovery import _source_file_identity

        file_cache: dict[str, Any] = {}
        self.source_identity = [
            [_source_file_identity(name, file_cache) for name in pair] for pair in self.cases
        ]
        self.foreground_index = self._load_or_build_index(cache_path)
        digest = hashlib.sha256()
        for name in sorted(self.foreground_index):
            array = np.ascontiguousarray(self.foreground_index[name])
            digest.update(f"{name}:{array.dtype.str}:{array.shape}".encode())
            digest.update(array.tobytes())
        self.foreground_index_sha256 = digest.hexdigest()

    def _index_key(self) -> str:
        return json.dumps(
            {
                "version": 1,
                "sampling_seed": self.geometry["sampling_seed"],
                "foreground_voxel_cache": self.geometry["foreground_voxel_cache"],
                "sources": [[record["sha256"] for record in pair] for pair in self.source_identity],
            },
            sort_keys=True,
        )

    def _load_or_build_index(self, cache_path: Path | None) -> dict[str, np.ndarray]:
        key = self._index_key()
        if cache_path is not None and Path(cache_path).exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                if "key" in cached.files and str(cached["key"]) == key:
                    return {name: cached[name] for name in cached.files if name != "key"}
        index = self._build_index()
        if cache_path is not None:
            temporary = Path(f"{cache_path}.tmp")
            with open(temporary, "wb") as stream:
                np.savez(stream, **cast(dict[str, Any], {"key": np.array(key), **index}))
            os.replace(temporary, cache_path)
        return index

    def _build_index(self) -> dict[str, np.ndarray]:
        index: dict[str, np.ndarray] = {}
        limit = self.geometry["foreground_voxel_cache"]
        for case_index, (volume_path, mask_path) in enumerate(self.cases):
            labels = np.asarray(cast(SpatialImage, nib.load(mask_path)).dataobj)
            if not np.isin(labels, [0, 1, 2]).all():
                raise ValueError(f"segmentation labels for {mask_path} must be in {{0,1,2}}")
            liver, tumor = np.argwhere(labels >= 1), np.argwhere(labels == 2)
            if len(liver) == 0:
                raise ValueError(
                    f"case {_case_id(volume_path, 'volume')} ({volume_path}) "
                    "has no liver or tumor voxels"
                )
            rng = np.random.default_rng([self.geometry["sampling_seed"], case_index])
            index[f"liver_{case_index}"] = _subsample(liver, limit, rng)
            index[f"tumor_{case_index}"] = _subsample(tumor, limit, rng)
            index[f"liver_bbox_{case_index}"] = _bounding_box(liver)
            index[f"tumor_bbox_{case_index}"] = _bounding_box(tumor)
        return index

    def manifest(self) -> dict:
        return {
            "version": 1,
            "geometry": self.geometry,
            "passes_per_epoch": self.geometry["passes_per_epoch"],
            "sampling_seed": self.geometry["sampling_seed"],
            "detect_tumors": self.detect_tumors,
            "cases": [
                {
                    "image": image,
                    "target": target,
                    "case_id": _case_id(image, "volume"),
                    "image_sha256": image_identity["sha256"],
                    "target_sha256": target_identity["sha256"],
                    **geometry,
                }
                for (image, target), (image_identity, target_identity), geometry in zip(
                    self.cases, self.source_identity, self.case_geometry, strict=True
                )
            ],
            "foreground_index_sha256": self.foreground_index_sha256,
        }

    def __len__(self) -> int:
        return int(self.geometry["passes_per_epoch"]) * len(self.cases)

    def crop_parameters(self, index: int, item_seed: int) -> dict[str, Any]:
        """Pure function of ``(index, item_seed)`` and the static foreground index."""
        if type(index) is not int or type(item_seed) is not int:
            raise TypeError("crop key must be (int index, int item_seed)")
        if not 0 <= index < len(self):
            raise IndexError("crop index out of range")
        if not 0 <= item_seed < 2**63:
            raise ValueError("item_seed must be a non-negative 63-bit integer")
        geometry = self.geometry
        case_index = index % len(self.cases)
        u_class, u_pick, u_scale, u_flip = np.random.default_rng(item_seed).random(4)
        tumor = self.foreground_index[f"tumor_{case_index}"]
        from_tumor = len(tumor) > 0 and u_class < geometry["tumor_centre_probability"]
        pool = tumor if from_tumor else self.foreground_index[f"liver_{case_index}"]
        centre_hwd = tuple(int(v) for v in pool[min(int(u_pick * len(pool)), len(pool) - 1)])
        low, high = geometry["scale_range"]
        scale = float(low + (high - low) * u_scale)
        extent = (
            geometry["window"],
            max(1, round(geometry["height"] * scale)),
            max(1, round(geometry["width"] * scale)),
        )
        height, width, depth = self.case_geometry[case_index]["shape"]
        centre_dhw = (centre_hwd[2], centre_hwd[0], centre_hwd[1])
        # Per dimension: clamp the crop inside the volume when it fits, else start at 0 and pad.
        start_dhw = tuple(
            min(max(c - e // 2, 0), n - e) if e <= n else 0
            for c, e, n in zip(centre_dhw, extent, (depth, height, width), strict=True)
        )
        return {
            "case_index": case_index,
            "centre_hwd": centre_hwd,
            "from_tumor": bool(from_tumor),
            "scale": scale,
            "extent": extent,
            "start_dhw": start_dhw,
            "flip": bool(u_flip < geometry["hflip_probability"]),
        }

    def __getitem__(self, key: tuple[int, int]) -> dict[str, Any]:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("ForegroundCropDataset items are keyed by (index, item_seed)")
        params = self.crop_parameters(*key)
        case_index, start, extent = params["case_index"], params["start_dhw"], params["extent"]
        volume_path, mask_path = self.cases[case_index]
        validate_pair(volume_path, mask_path)
        image = cast(SpatialImage, nib.load(volume_path))
        target_img = cast(SpatialImage, nib.load(mask_path))
        if (
            tuple(image.shape) != self.case_geometry[case_index]["shape"]
            or image.affine.tolist() != self.case_geometry[case_index]["affine"]
        ):
            raise ValueError("source geometry changed since foreground indexing")
        height, width, depth = image.shape
        valid = tuple(
            min(e, n - s) for s, e, n in zip(start, extent, (depth, height, width), strict=True)
        )
        crop = {"window": extent[0], "height": extent[1], "width": extent[2]}
        array = np.asarray(image.dataobj[tile_slices(start, valid)], dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("image crop contains nonfinite values")
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)[None]
        if self.geometry["clamp_hu"]:
            tensor = tensor.clamp(self.geometry["hu_low"], self.geometry["hu_high"])
        tensor = pad_tile(tensor, valid, crop, self.geometry["hu_low"])
        labels = np.asarray(target_img.dataobj[tile_slices(start, valid)])
        if not np.isfinite(labels).all() or not np.isin(labels, [0, 1, 2]).all():
            raise ValueError("target crop labels must be finite integers in {0,1,2}")
        target = torch.from_numpy(labels.astype(np.int64)).permute(2, 0, 1)[None]
        target = pad_tile(target, valid, crop, self.geometry["padding_target"])
        size = (self.geometry["window"], self.geometry["height"], self.geometry["width"])
        if tuple(extent) != size:
            # Depth is never resampled; in-plane uses the half-pixel grid for both.
            tensor = F.interpolate(tensor[None], size=size, mode="trilinear", align_corners=False)[
                0
            ]
            target = F.interpolate(target[None].float(), size=size, mode="nearest-exact")[0].long()
        if params["flip"]:
            tensor, target = tensor.flip(-1), target.flip(-1)
        mapping = crop_to_source(start, extent, self.geometry, params["flip"])
        return {
            "image": tensor,
            "target": target,
            "valid_mask": target != self.geometry["padding_target"],
            "case_id": _case_id(volume_path, "volume"),
            "case_index": case_index,
            "start": torch.tensor(start),
            "valid_extents": torch.tensor(valid),
            "valid_depth": valid[0],
            "source_shape": torch.tensor(image.shape),
            "source_affine": torch.tensor(image.affine, dtype=torch.float64),
            "model_to_source": torch.tensor(mapping, dtype=torch.float64),
            "model_to_world": torch.tensor(image.affine @ mapping, dtype=torch.float64),
        }


def crop_to_source(
    start: tuple[int, int, int], extent: tuple[int, int, int], geometry: dict, flip: bool
) -> np.ndarray:
    """Map model (D,H,W) indices to source (H,W,D) under half-pixel in-plane scaling.

    Identity-scaled (``tile_to_source``) when the extent equals the model size and no
    flip is applied; a W flip maps model column ``m`` to resized column ``width-1-m``.
    """
    d, h, w = start
    qh, qw = extent[1] / geometry["height"], extent[2] / geometry["width"]
    row_w = (
        [0, 0, -qw, w + qw * (geometry["width"] - 1) + (qw - 1) / 2]
        if flip
        else [0, 0, qw, w + (qw - 1) / 2]
    )
    return np.array(
        [[0, qh, 0, h + (qh - 1) / 2], row_w, [1, 0, 0, d], [0, 0, 0, 1]], dtype=np.float64
    )
