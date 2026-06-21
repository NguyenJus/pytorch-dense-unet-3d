"""Tests for LITSDataset (Task D2).

TDD: written BEFORE the implementation rewrite.

Coverage
--------
find_liver
    - regression: returns a 2-tuple (no TypeError from ``tuple(vol, seg)``)
    - output shapes: depth axis is cropped to only slices with liver voxels

LITSDataset.__len__
    - reports the number of volumes discovered

LITSDataset.__getitem__
    - image shape: (1, 12, 224, 224)
    - mask labels: integer dtype, values in {0, 1, 2}
    - returns a (image, mask) 2-tuple
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from dense_unet_3d.dataset.LITSDataset import LITSDataset

# ---------------------------------------------------------------------------
# Helpers / local fixtures
# ---------------------------------------------------------------------------


def _write_nifti(path: Path, data: np.ndarray) -> None:
    """Write a NIfTI file at *path* with an identity affine."""
    img = nib.Nifti1Image(data, np.eye(4, dtype=np.float64))
    nib.save(img, str(path))


@pytest.fixture()
def nifti_dir(tmp_path: Path) -> Path:
    """Return a temp directory with one volume + one segmentation NIfTI pair.

    volume0.nii  — float32 (224, 224, 12) in HWD NIfTI order
    segmentation0.nii — int16  (224, 224, 12) labels {0, 1, 2}
    """
    rng = np.random.default_rng(0)
    vol_data = rng.uniform(-200.0, 250.0, size=(224, 224, 12)).astype(np.float32)
    seg_data = rng.choice(
        np.array([0, 1, 2], dtype=np.int16),
        size=(224, 224, 12),
        p=[0.7, 0.2, 0.1],
    ).astype(np.int16)

    _write_nifti(tmp_path / "volume0.nii", vol_data)
    _write_nifti(tmp_path / "segmentation0.nii", seg_data)
    return tmp_path


@pytest.fixture()
def dataset_no_transform(nifti_dir: Path) -> LITSDataset:
    """A LITSDataset with no transforms."""
    return LITSDataset(img_dirs=[str(nifti_dir)])


# ---------------------------------------------------------------------------
# find_liver regression: must NOT raise TypeError
# ---------------------------------------------------------------------------


class TestFindLiver:
    """Regression suite for find_liver."""

    def test_returns_2_tuple_not_type_error(self, dataset_no_transform: LITSDataset) -> None:
        """``find_liver`` must return a plain 2-tuple, not raise TypeError.

        The original bug was ``return tuple(vol, seg)`` — tuple() takes an
        iterable, not two positional args.  The fix is ``return vol, seg``.
        """
        rng = np.random.default_rng(1)
        vol = rng.random((12, 224, 224), dtype=np.float32)
        # At least one slice has liver voxels (value 1).
        seg = np.zeros((12, 224, 224), dtype=np.int16)
        seg[3, 50:60, 50:60] = 1  # inject a small liver region
        seg[7, 80:90, 80:90] = 1

        result = dataset_no_transform.find_liver((vol, seg))

        assert isinstance(result, tuple), "find_liver must return a tuple"
        assert len(result) == 2, "find_liver must return a 2-tuple"

    def test_return_value_is_not_type_error(self, dataset_no_transform: LITSDataset) -> None:
        """Calling find_liver must not raise TypeError at all."""
        rng = np.random.default_rng(2)
        vol = rng.random((12, 224, 224), dtype=np.float32)
        seg = np.zeros((12, 224, 224), dtype=np.int16)
        seg[5, 10:20, 10:20] = 1

        try:
            dataset_no_transform.find_liver((vol, seg))
        except TypeError as exc:
            pytest.fail(f"find_liver raised TypeError: {exc}")

    def test_crops_to_liver_slices(self, dataset_no_transform: LITSDataset) -> None:
        """Returned arrays only contain depth slices that have liver voxels."""
        rng = np.random.default_rng(3)
        vol = rng.random((12, 224, 224), dtype=np.float32)
        seg = np.zeros((12, 224, 224), dtype=np.int16)
        # Only slices 2 and 8 have liver.
        liver_slices = [2, 8]
        for s in liver_slices:
            seg[s, 0, 0] = 1

        result_vol, result_seg = dataset_no_transform.find_liver((vol, seg))

        # Depth dimension should equal the number of liver slices.
        # After the transpose in find_liver, shape is (H, W, n_slices).
        n_liver = len(liver_slices)
        assert result_vol.shape[2] == n_liver, (
            f"Expected depth={n_liver}, got {result_vol.shape[2]}"
        )
        assert result_seg.shape[2] == n_liver


# ---------------------------------------------------------------------------
# LITSDataset.__len__
# ---------------------------------------------------------------------------


class TestLen:
    def test_len_equals_number_of_volumes(self, nifti_dir: Path) -> None:
        ds = LITSDataset(img_dirs=[str(nifti_dir)])
        assert len(ds) == 1

    def test_len_two_volumes(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(10)
        for i in range(2):
            vol = rng.uniform(-200, 250, (224, 224, 12)).astype(np.float32)
            seg = np.zeros((224, 224, 12), dtype=np.int16)
            _write_nifti(tmp_path / f"volume{i}.nii", vol)
            _write_nifti(tmp_path / f"segmentation{i}.nii", seg)

        ds = LITSDataset(img_dirs=[str(tmp_path)])
        assert len(ds) == 2


# ---------------------------------------------------------------------------
# LITSDataset.__getitem__
# ---------------------------------------------------------------------------


class TestGetItem:
    def test_returns_2_tuple(self, dataset_no_transform: LITSDataset) -> None:
        item = dataset_no_transform[0]
        assert isinstance(item, tuple) and len(item) == 2

    def test_image_shape_is_1_12_224_224(self, dataset_no_transform: LITSDataset) -> None:
        """Image must have shape (1, 12, 224, 224) — channel-first CDHW."""
        image, _mask = dataset_no_transform[0]
        assert tuple(image.shape) == (1, 12, 224, 224), (
            f"Expected (1, 12, 224, 224), got {tuple(image.shape)}"
        )

    def test_mask_labels_in_0_1_2(self, dataset_no_transform: LITSDataset) -> None:
        """Mask labels must be integers in {0, 1, 2}."""
        _image, mask = dataset_no_transform[0]
        unique = torch.unique(mask).tolist()
        for v in unique:
            assert v in {0, 1, 2}, f"Unexpected label value {v}"

    def test_mask_is_integer_dtype(self, dataset_no_transform: LITSDataset) -> None:
        """Mask tensor must be an integer dtype (long)."""
        _image, mask = dataset_no_transform[0]
        assert mask.dtype in (torch.int32, torch.int64, torch.long), (
            f"Expected integer dtype, got {mask.dtype}"
        )

    def test_image_is_float_tensor(self, dataset_no_transform: LITSDataset) -> None:
        """Image tensor must be float."""
        image, _mask = dataset_no_transform[0]
        assert image.dtype == torch.float32, f"Expected float32, got {image.dtype}"

    def test_crop_to_liver_still_produces_correct_channel(self, nifti_dir: Path) -> None:
        """With crop_to_liver=True, image must still have shape (1, D, H, W)."""
        ds = LITSDataset(img_dirs=[str(nifti_dir)], crop_to_liver=True)
        image, mask = ds[0]
        # Channel dim must be 1; D may vary; H and W should be 224.
        assert image.shape[0] == 1
        assert image.shape[2] == 224
        assert image.shape[3] == 224
