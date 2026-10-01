from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dense_unet_3d.dataset.slabs import (
    NativeSlabDataset,
    model_to_source,
    slab_starts,
    spatial_config,
)
from scripts.audit_native_tiling import physical_units
from scripts.census_reconstruction import audit_case, nearest_indices, roundtrip_radius


def config(height=5, width=7, depth=12):
    return {
        "resize_dims": {"D": depth, "H": height, "W": width},
        "clamp_hu": False,
        "random_hflip": False,
        "scale_img": False,
    }


def write_case(root: Path, name: str, labels: np.ndarray, affine=None):
    affine = np.eye(4) if affine is None else affine
    image = root / f"volume-{name}.nii"
    target = root / f"segmentation-{name}.nii"
    nib.save(nib.Nifti1Image(labels.astype(np.float32), affine), image)
    nib.save(nib.Nifti1Image(labels.astype(np.uint8), affine), target)
    return str(image), str(target)


def test_coverage_exhaustive():
    for depth in range(1, 150):
        starts = slab_starts(depth)
        assert starts == sorted(set(starts))
        coverage = np.zeros(depth, int)
        for start in starts:
            coverage[start : min(start + 12, depth)] += 1
        assert (coverage > 0).all()
        assert starts[-1] == max(0, depth - 12)
    with pytest.raises(ValueError):
        slab_starts(0)


@pytest.mark.parametrize(
    ("unit", "volume"), [("mm", 1.0), ("meter", 1e9), ("micron", 1e-9), ("unknown", 1.0)]
)
def test_census_volume_units_are_explicit(tmp_path, unit, volume):
    image_path, target_path = write_case(tmp_path, "units", np.full((2, 2, 2), 2, dtype=np.uint8))
    for path in (image_path, target_path):
        image = nib.load(path)
        data = np.asanyarray(image.dataobj).copy()
        image.header.set_xyzt_units(unit)
        nib.save(nib.Nifti1Image(data, image.affine, image.header), path)
    from dense_unet_3d.dataset.slabs import spatial_config

    receipt = audit_case(image_path, target_path, spatial_config(config(2, 2, 2)))
    assert receipt["source_spatial_units"] == unit
    assert receipt["native_voxel_volume_mm3"] == pytest.approx(volume)
    assert ("assumed" in receipt["physical_units_disposition"]) == (unit == "unknown")


def test_native_padding_collation_and_manifest(tmp_path):
    write_case(tmp_path, "a", np.full((5, 7, 3), 2))
    write_case(tmp_path, "b", np.full((5, 7, 13), 1))
    dataset = NativeSlabDataset([str(tmp_path)], config())
    assert dataset.sample_index == [(0, 0, 3), (1, 0, 12), (1, 1, 12)]
    sample = dataset[0]
    assert sample["image"].shape == (1, 12, 5, 7)
    assert torch.all(sample["target"][:, :3] == 2)
    assert torch.all(sample["target"][:, 3:] == -100)
    assert torch.equal(sample["valid_mask"], sample["target"] != -100)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["image"].shape == (2, 1, 12, 5, 7)
    assert len(dataset.manifest()["cases"]) == 2


def test_physical_landmarks_with_half_pixel_and_axis_map(tmp_path):
    affine = np.array([[0, -2, 0.1, 40], [1.5, 0, 0.2, -30], [0.3, 0, 4, 6], [0, 0, 0, 1]])
    labels = np.zeros((10, 14, 25), dtype=np.uint8)
    write_case(tmp_path, "a", labels, affine)
    dataset = NativeSlabDataset([str(tmp_path)], config())
    sample = dataset[1]
    point = np.array([3, 2, 4, 1])  # Model D,H,W at start 12.
    source = np.array([4.5, 8.5, 15, 1])
    np.testing.assert_allclose(sample["model_to_source"].numpy() @ point, source)
    np.testing.assert_allclose(sample["model_to_world"].numpy() @ point, affine @ source)
    np.testing.assert_allclose(
        model_to_source(labels.shape, dataset.geometry, 12), sample["model_to_source"]
    )


def test_geometry_tolerance_precedes_real_census():
    for native in range(1, 35):
        for model in (1, 2, 7, 12, 16, 31):
            ramp = torch.arange(native, dtype=torch.float64)[None, None]
            down = F.interpolate(ramp, size=model, mode="nearest-exact")
            restored = F.interpolate(down, size=native, mode="nearest-exact")[0, 0].numpy()
            np.testing.assert_array_equal(down[0, 0].numpy(), nearest_indices(native, model))
            assert int(np.max(np.abs(restored - np.arange(native)))) == roundtrip_radius(
                native, model
            )
            assert roundtrip_radius(native, model) <= np.ceil(native / (2 * model) + 0.5)


def test_census_tracks_native_ids_erasure_core_and_volume(tmp_path):
    labels = np.zeros((8, 8, 3), dtype=np.uint8)
    labels[0, 0, 0] = 2  # Completely missed by 8→2 center sampling.
    labels[2:7, 2:7, 1:] = 2
    image, target = write_case(tmp_path, "a", labels, np.diag([2, 3, 4, 1]))
    dataset = NativeSlabDataset([str(tmp_path)], config(2, 2))
    result = audit_case(image, target, dataset.geometry)
    assert result["original_components"] == 2
    assert result["erased_components"] == 1
    assert not result["gate_pass"]
    assert result["covered_slices"] == 3
    assert result["native_voxel_volume_mm3"] == pytest.approx(24)
    assert result["model_voxel_volume_mm3"] == pytest.approx(384)
    assert all(c["core_lost_voxels"] == 0 for c in result["components"])
    assert all(c["boundary_gate_pass"] for c in result["components"])


def test_census_26_connectivity_and_identity_native_grid(tmp_path):
    labels = np.zeros((5, 7, 3), dtype=np.uint8)
    labels[0, 0, 0] = labels[1, 1, 1] = 2
    labels[4, 6, 2] = 2
    image, target = write_case(tmp_path, "a", labels)
    dataset = NativeSlabDataset([str(tmp_path)], config())
    result = audit_case(image, target, dataset.geometry)
    assert result["original_components"] == 2
    assert result["gate_pass"]
    assert all(c["native_overlap_retention_ratio"] == 1 for c in result["components"])
    assert all(c["model_physical_volume_ratio"] == 1 for c in result["components"])


def test_unresolved_augmentation_requires_explicit_disable(tmp_path):
    with pytest.raises(ValueError, match="augmentation is unresolved"):
        NativeSlabDataset([str(tmp_path)], {**config(), "random_hflip": True})


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"resize_dims": {"D": 12.5, "H": 5, "W": 7}}, "dimensions must be integers"),
        ({"resize_dims": {"D": True, "H": 5, "W": 7}}, "dimensions must be integers"),
        ({"resize_img": False}, "requires resize_img=true"),
        ({"random_hflip": "false"}, "augmentation flags must be booleans"),
        ({"clamp_hu": 1}, "clamp_hu must be a boolean"),
        ({"clamp_hu_range": {"min": False, "max": 250}}, "finite numbers"),
    ],
)
def test_native_spatial_config_rejects_silent_coercions(update, message):
    with pytest.raises(ValueError, match=message):
        spatial_config({**config(), **update})


@pytest.mark.parametrize(
    ("unit", "scale", "disposition"),
    [
        ("mm", 1.0, "header_declared_converted_to_mm"),
        ("meter", 1000.0, "header_declared_converted_to_mm"),
        ("micron", 0.001, "header_declared_converted_to_mm"),
        ("unknown", 1.0, "assumed_mm_from_LiTS_provenance"),
    ],
)
def test_native_tile_audit_physical_units(unit, scale, disposition):
    image = nib.Nifti1Image(np.zeros((1, 1, 1)), np.eye(4))
    image.header.set_xyzt_units(unit)
    assert physical_units(image) == (unit, scale, disposition)
