#!/usr/bin/env python3
"""Deterministic native-component retention audit; never selects model inputs.

Run only after the predeclared geometry tests. scipy is an audit dependency.
"""

from __future__ import annotations

import argparse
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
from dense_unet_3d.dataset.slabs import (
    categorical_tile,
    slab_starts,
    spatial_config,
    tile_bounds,
    tile_slices,
)

TOLERANCE_VERSION = "2026-09-30-exact-roundtrip-inner-shell-v1"


def nearest_indices(source: int, destination: int) -> np.ndarray:
    return np.minimum(
        source - 1, np.floor((np.arange(destination) + 0.5) * source / destination)
    ).astype(np.int64)


def roundtrip_radius(native: int, model: int) -> int:
    down = nearest_indices(native, model)
    up = nearest_indices(model, native)
    return int(np.max(np.abs(down[up] - np.arange(native))))


def audit_case(image_path: str, target_path: str, geometry: dict) -> dict:
    validate_pair(image_path, target_path)
    image = cast(SpatialImage, nib.load(image_path))
    target = cast(SpatialImage, nib.load(target_path))
    mask = np.asanyarray(target.dataobj)
    if not np.isfinite(mask).all() or not np.isin(mask, [0, 1, 2]).all():
        raise ValueError("census labels must be finite integers in {0,1,2}")
    ids, count = ndimage.label(mask == 2, structure=np.ones((3, 3, 3), dtype=np.uint8))
    del mask
    height, width, depth = ids.shape
    model_h, model_w = geometry["height"], geometry["width"]
    native_tiles = geometry["representation"] == "native_tiles_v1"
    if native_tiles:
        # Reassemble IDs with production extraction/indexing, then compare identities.
        # Overlap is never counted twice in native/model component volumes.
        reconstructed = np.zeros_like(ids)
        coverage_native = np.zeros(ids.shape, dtype=np.uint8)
        for start, valid in tile_bounds(ids.shape, geometry):
            slices = tile_slices(start, valid)
            tile, actual_valid = categorical_tile(ids, start, geometry)
            if actual_valid != valid:
                raise ValueError("native tile validity failure")
            vd, vh, vw = valid
            validity = np.zeros(tile.shape, dtype=bool)
            validity[:, :vd, :vh, :vw] = True
            if not np.all(tile.numpy()[~validity] == -100):
                raise ValueError("native tile padding validity failure")
            reconstructed[slices] = tile[0, :vd, :vh, :vw].numpy().transpose(1, 2, 0)
            coverage_native[slices] += 1
        if not np.all(coverage_native > 0) or not np.array_equal(reconstructed, ids):
            raise ValueError("native tile component identity or coverage failure")
        del reconstructed, coverage_native
        model_h, model_w = height, width
    rows, columns = nearest_indices(height, model_h), nearest_indices(width, model_w)
    restore_rows, restore_columns = (
        nearest_indices(model_h, height),
        nearest_indices(model_w, width),
    )
    rh, rw = roundtrip_radius(height, model_h), roundtrip_radius(width, model_w)
    native_counts = np.zeros(count + 1, dtype=np.int64)
    model_counts = native_counts.copy()
    retained_counts = native_counts.copy()
    restored_counts = native_counts.copy()
    shell_counts = native_counts.copy()
    core_lost_counts = native_counts.copy()
    for z in range(depth):
        plane = ids[:, :, z]
        sampled = plane[np.ix_(rows, columns)]
        restored = sampled[np.ix_(restore_rows, restore_columns)]
        size = (2 * rh + 1, 2 * rw + 1)
        minimum = (
            plane
            if native_tiles
            else ndimage.minimum_filter(plane, size=size, mode="constant", cval=0)
        )
        maximum = (
            plane
            if native_tiles
            else ndimage.maximum_filter(plane, size=size, mode="constant", cval=0)
        )
        core = (plane > 0) & (minimum == maximum)
        retained = (plane > 0) & (restored == plane)
        for accum, values in (
            (native_counts, plane.ravel()),
            (model_counts, sampled.ravel()),
            (restored_counts, restored.ravel()),
            (retained_counts, plane[retained]),
            (shell_counts, plane[(plane > 0) & ~core]),
            (core_lost_counts, plane[core & ~retained]),
        ):
            accum += np.bincount(values, minlength=count + 1)
    coverage = np.zeros(depth, dtype=np.int32)
    starts = slab_starts(depth, geometry["window"])
    for start in starts:
        coverage[start : min(depth, start + geometry["window"])] += 1
    spatial_units = cast(nib.Nifti1Header | nib.Nifti2Header, image.header).get_xyzt_units()[0]
    # LiTS reports millimeter spacing, but some distributed headers omit units.
    # Preserve that uncertainty rather than claiming header-verified mm values.
    mm_per_unit = {"mm": 1.0, "meter": 1000.0, "micron": 0.001, "unknown": 1.0}[spatial_units]
    native_voxel_volume = float(abs(np.linalg.det(image.affine[:3, :3]))) * mm_per_unit**3
    model_voxel_volume = (
        native_voxel_volume
        if native_tiles
        else native_voxel_volume * height / model_h * width / model_w
    )
    components = []
    for component in range(1, count + 1):
        original = int(native_counts[component])
        retained = int(retained_counts[component])
        model_count = int(model_counts[component])
        shell = int(shell_counts[component])
        core_loss = int(core_lost_counts[component])
        lost = original - retained
        components.append(
            {
                "native_component_id": component,
                "native_voxels": original,
                "model_voxels": model_count,
                "native_retained_voxels": retained,
                "native_roundtrip_voxels": int(restored_counts[component]),
                "native_lost_voxels": lost,
                "inner_shell_voxels": shell,
                "core_lost_voxels": core_loss,
                "erased": model_count == 0,
                "native_volume_mm3": original * native_voxel_volume,
                "model_volume_mm3": model_count * model_voxel_volume,
                "model_physical_volume_ratio": model_count
                * model_voxel_volume
                / (original * native_voxel_volume),
                "native_overlap_retention_ratio": retained / original,
                "native_lost_volume_mm3": lost * native_voxel_volume,
                "shell_tolerance_volume_mm3": shell * native_voxel_volume,
                "size_stratum_native_voxels": "1-9"
                if original < 10
                else "10-99"
                if original < 100
                else "100-999"
                if original < 1000
                else ">=1000",
                "boundary_gate_pass": core_loss == 0 and lost <= shell,
            }
        )
    return {
        "case_id": _case_id(image_path, "volume"),
        "image": image_path,
        "target": target_path,
        "source_shape_hwd": list(image.shape),
        "source_affine": image.affine.tolist(),
        "source_spatial_units": spatial_units,
        "physical_units_disposition": "assumed_mm_from_LiTS_provenance"
        if spatial_units == "unknown"
        else "header_declared_converted_to_mm",
        "native_voxel_volume_mm3": native_voxel_volume,
        "model_voxel_volume_mm3": model_voxel_volume,
        "geometry_radius_hw": [rh, rw],
        "slab_starts": starts,
        "native_tiles": sum(1 for _ in tile_bounds(ids.shape, geometry)) if native_tiles else None,
        "covered_slices": int(np.count_nonzero(coverage)),
        "native_slices": depth,
        "minimum_depth_coverage": int(coverage.min()),
        "maximum_depth_coverage": int(coverage.max()),
        "original_components": count,
        "retained_components": sum(not c["erased"] for c in components),
        "erased_components": sum(c["erased"] for c in components),
        "boundary_gate_failures": sum(not c["boundary_gate_pass"] for c in components),
        "native_tumor_voxels": int(native_counts[1:].sum()),
        "model_tumor_voxels": int(model_counts[1:].sum()),
        "native_retained_tumor_voxels": int(retained_counts[1:].sum()),
        "native_tumor_volume_mm3": float(native_counts[1:].sum() * native_voxel_volume),
        "model_tumor_volume_mm3": float(model_counts[1:].sum() * model_voxel_volume),
        "native_retained_tumor_volume_mm3": float(retained_counts[1:].sum() * native_voxel_volume),
        "gate_pass": bool(np.all(coverage > 0))
        and all(not c["erased"] and c["boundary_gate_pass"] for c in components),
        "components": components,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Omit per-case paths, grids and component data from saved receipt",
    )
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    geometry = spatial_config(config["dataset"])
    receipt: dict = {
        "tolerance_version": TOLERANCE_VERSION,
        "connectivity": 26,
        "geometry": geometry,
        "augmentation": "disabled",
        "splits": {},
    }
    for split, directory_key in (("train", "train_img_dirs"), ("validation", "test_img_dirs")):
        pairs = discover_pairs(config["pathing"][directory_key])
        _validate_split_manifest(config, split, pairs)
        cases = []
        for image, target in pairs:
            result = audit_case(image, target, geometry)
            cases.append(result)
            print(  # noqa: T201
                f"{split} {result['case_id']}: components={result['original_components']} erased={result['erased_components']} core_failures={result['boundary_gate_failures']}",
                flush=True,
            )
        components = [component for case in cases for component in case["components"]]
        strata = {}
        for stratum in ("1-9", "10-99", "100-999", ">=1000"):
            selected = [c for c in components if c["size_stratum_native_voxels"] == stratum]
            original = sum(c["native_volume_mm3"] for c in selected)
            retained = sum(c["native_volume_mm3"] - c["native_lost_volume_mm3"] for c in selected)
            strata[stratum] = {
                "components": len(selected),
                "erased": sum(c["erased"] for c in selected),
                "native_overlap_physical_retention_ratio": retained / original
                if original
                else None,
            }
        receipt["splits"][split] = {
            "cases": [] if args.aggregate_only else cases,
            "case_count": len(cases),
            "original_components": len(components),
            "erased_components": sum(c["erased"] for c in components),
            "boundary_gate_failures": sum(not c["boundary_gate_pass"] for c in components),
            "native_tumor_voxels": sum(c["native_tumor_voxels"] for c in cases),
            "covered_slices": sum(c["covered_slices"] for c in cases),
            "native_slices": sum(c["native_slices"] for c in cases),
            "native_tiles": sum(c["native_tiles"] or 0 for c in cases),
            "size_strata": strata,
            "native_tumor_volume_mm3": sum(c["native_tumor_volume_mm3"] for c in cases),
            "model_tumor_volume_mm3": sum(c["model_tumor_volume_mm3"] for c in cases),
            "native_retained_tumor_volume_mm3": sum(
                c["native_retained_tumor_volume_mm3"] for c in cases
            ),
            "gate_pass": bool(cases) and all(c["gate_pass"] for c in cases),
        }
    receipt["gate_pass"] = all(split["gate_pass"] for split in receipt["splits"].values())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(f"R2 gate_pass={receipt['gate_pass']}; receipt={output}", flush=True)  # noqa: T201
    if not receipt["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
