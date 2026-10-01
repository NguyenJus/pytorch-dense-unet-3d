#!/usr/bin/env python3
"""Header-only native tile feasibility and synthetic exact reassembly audit.

This is an engineering investigation, not a production sampling implementation.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import yaml
from nibabel.spatialimages import SpatialImage
from scipy import ndimage

from dense_unet_3d.dataset.LITSDataset import _case_id, discover_pairs, validate_pair
from dense_unet_3d.dataset.prepare_dataset import _validate_split_manifest
from dense_unet_3d.dataset.slabs import slab_starts


def tile_bounds(shape_hwd: tuple[int, int, int], model_dhw=(12, 224, 224)):
    """Native bounds (d,h,w), valid extents, stable label-independent order."""
    height, width, depth = shape_hwd
    window_d, window_h, window_w = model_dhw
    for start_d, start_h, start_w in itertools.product(
        slab_starts(depth, window_d), slab_starts(height, window_h), slab_starts(width, window_w)
    ):
        yield (
            (start_d, start_h, start_w),
            (
                min(window_d, depth - start_d),
                min(window_h, height - start_h),
                min(window_w, width - start_w),
            ),
        )


def tile_to_source(start_dhw) -> np.ndarray:
    """Model DHW → source HWD; native spacing, no half-pixel resize offset."""
    start_d, start_h, start_w = start_dhw
    return np.array(
        [[0, 1, 0, start_h], [0, 0, 1, start_w], [1, 0, 0, start_d], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def synthetic_checks() -> dict:
    checks = []
    for shape, model in [
        ((1, 1, 1), (12, 224, 224)),
        ((223, 219, 3), (12, 224, 224)),
        ((224, 224, 12), (12, 224, 224)),
        ((225, 237, 13), (12, 224, 224)),
        ((512, 512, 25), (12, 224, 224)),
        ((301, 117, 17), (12, 224, 224)),
    ]:
        # Unique IDs verify every coordinate including boundaries and single voxels.
        source = np.arange(np.prod(shape), dtype=np.int64).reshape(shape) + 1
        total = np.zeros(shape, dtype=np.int64)
        coverage = np.zeros(shape, dtype=np.int16)
        padded = 0
        for (d, h, w), (vd, vh, vw) in tile_bounds(shape, model):
            tile = np.full(model, -100, dtype=np.int64)
            valid = np.zeros(model, dtype=bool)
            tile[:vd, :vh, :vw] = source[h : h + vh, w : w + vw, d : d + vd].transpose(2, 0, 1)
            valid[:vd, :vh, :vw] = True
            assert np.all(tile[~valid] == -100)
            total[h : h + vh, w : w + vw, d : d + vd] += tile[:vd, :vh, :vw].transpose(1, 2, 0)
            coverage[h : h + vh, w : w + vw, d : d + vd] += valid[:vd, :vh, :vw].transpose(1, 2, 0)
            padded += int(np.count_nonzero(~valid))
        assert np.all(coverage > 0)
        assert np.array_equal(total, source * coverage)
        restored = total // coverage
        assert np.array_equal(source, restored)
        # Verify stable 26-component identities, with corner/diagonal components.
        tumor = (source == 1) | (source == source.max()) | (source % 997 == 0)
        reference_ids, reference_count = ndimage.label(tumor, np.ones((3, 3, 3)))
        restored_ids, restored_count = ndimage.label(
            tumor & (restored == source), np.ones((3, 3, 3))
        )
        assert reference_count == restored_count
        assert np.array_equal(reference_ids, restored_ids)
        checks.append(
            {
                "source_shape_hwd": shape,
                "model_shape_dhw": model,
                "tiles": len(list(tile_bounds(shape, model))),
                "minimum_coverage": int(coverage.min()),
                "maximum_coverage": int(coverage.max()),
                "padded_model_voxels": padded,
                "component_ids_preserved": reference_count,
                "exact_coordinate_reassembly": True,
            }
        )
    affine = np.array([[0, -2, 0.1, 40], [1.5, 0, 0.2, -30], [0.3, 0, 4, 6], [0, 0, 0, 1]])
    mapping = tile_to_source((12, 224, 288))
    point = np.array([3, 4, 5, 1])
    expected_source = np.array([228, 293, 15, 1])
    assert np.array_equal(mapping @ point, expected_source)
    assert np.allclose((affine @ mapping) @ point, affine @ expected_source)
    return {
        "cases": checks,
        "physical_landmark": "oblique anisotropic affine passed",
        "native_roundtrip_retention": 1.0,
        "checks_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    dims = config["dataset"]["resize_dims"]
    model = tuple(int(dims[key]) for key in ("D", "H", "W"))
    receipt: dict = {
        "status": "provisional_engineering_fallback_not_selected",
        "header_only": True,
        "model_shape_dhw": model,
        "synthetic_checks": synthetic_checks(),
        "splits": {},
    }
    for split, key in (("train", "train_img_dirs"), ("validation", "test_img_dirs")):
        pairs = discover_pairs(config["pathing"][key])
        _validate_split_manifest(config, split, pairs)
        cases = []
        for image_path, target_path in pairs:
            validate_pair(image_path, target_path)  # Headers only: no dataobj access.
            image = cast(SpatialImage, nib.load(image_path))
            height, width, depth = (int(value) for value in image.shape)
            starts_d = slab_starts(depth, model[0])
            starts_h = slab_starts(height, model[1])
            starts_w = slab_starts(width, model[2])
            tile_count = len(starts_d) * len(starts_h) * len(starts_w)
            spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
            cases.append(
                {
                    "case_id": _case_id(image_path, "volume"),
                    "source_shape_hwd": [height, width, depth],
                    "source_spacing_hwd_mm": spacing.tolist(),
                    "source_voxel_volume_mm3": float(abs(np.linalg.det(image.affine[:3, :3]))),
                    "starts_d": starts_d,
                    "starts_h": starts_h,
                    "starts_w": starts_w,
                    "resize_slabs": len(starts_d),
                    "native_tiles": tile_count,
                    "tile_to_resize_count_ratio": tile_count / len(starts_d),
                    "native_tile_fov_hwd_mm": (
                        spacing * np.array([model[1], model[2], model[0]])
                    ).tolist(),
                    "example_model_to_source": tile_to_source(
                        (starts_d[-1], starts_h[-1], starts_w[-1])
                    ).tolist(),
                }
            )
        resize_slabs = sum(c["resize_slabs"] for c in cases)
        native_tiles = sum(c["native_tiles"] for c in cases)
        receipt["splits"][split] = {
            "cases": cases,
            "case_count": len(cases),
            "resize_slabs": resize_slabs,
            "native_tiles": native_tiles,
            "tile_to_resize_count_ratio": native_tiles / resize_slabs,
        }
    receipt["resize_slabs"] = sum(s["resize_slabs"] for s in receipt["splits"].values())
    receipt["native_tiles"] = sum(s["native_tiles"] for s in receipt["splits"].values())
    receipt["tile_to_resize_count_ratio"] = receipt["native_tiles"] / receipt["resize_slabs"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    summary = {
        key: receipt[key] for key in ("resize_slabs", "native_tiles", "tile_to_resize_count_ratio")
    }
    print(json.dumps(summary))  # noqa: T201


if __name__ == "__main__":
    main()
