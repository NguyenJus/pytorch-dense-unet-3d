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

from dense_unet_3d.dataset.LITSDataset import LITSDataset, preflight_pairs

# ---------------------------------------------------------------------------
# Helpers / local fixtures
# ---------------------------------------------------------------------------


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    """Write a NIfTI file at *path* with an identity affine by default."""
    img = nib.Nifti1Image(data, np.eye(4, dtype=np.float64) if affine is None else affine)
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

    def test_missing_segmentation_is_rejected_without_mispairing(self, tmp_path: Path) -> None:
        """A missing middle case must not shift later sorted paths into a wrong pair."""
        vol = np.zeros((8, 6, 4), dtype=np.float32)
        seg = np.zeros((8, 6, 4), dtype=np.int16)
        _write_nifti(tmp_path / "volume-0.nii", vol)
        _write_nifti(tmp_path / "volume-1.nii", vol)
        _write_nifti(tmp_path / "segmentation-0.nii", seg)
        with pytest.raises(ValueError, match="case IDs do not match|missing segmentations"):
            LITSDataset(img_dirs=[str(tmp_path)])

    def test_mismatched_image_mask_grids_are_rejected(self, tmp_path: Path) -> None:
        """Spatially different paired NIfTIs cannot be trained as aligned voxels."""
        _write_nifti(tmp_path / "volume0.nii", np.zeros((8, 6, 4), dtype=np.float32))
        _write_nifti(tmp_path / "segmentation0.nii", np.zeros((8, 6, 3), dtype=np.int16))
        with pytest.raises(ValueError, match="shape mismatch"):
            LITSDataset(img_dirs=[str(tmp_path)])[0]

    def test_four_dimensional_image_mask_pair_is_rejected(self, tmp_path: Path) -> None:
        """LiTS samples are 3-D volumes, not 4-D image sequences."""
        volume = np.zeros((8, 6, 4, 2), dtype=np.float32)
        segmentation = np.zeros((8, 6, 4, 2), dtype=np.int16)
        _write_nifti(tmp_path / "volume0.nii", volume)
        _write_nifti(tmp_path / "segmentation0.nii", segmentation)
        with pytest.raises(ValueError, match="must be 3-D"):
            LITSDataset(img_dirs=[str(tmp_path)])[0]

    def test_float_header_rounding_within_one_tenth_micrometre_is_accepted(
        self, tmp_path: Path
    ) -> None:
        """Equivalent float32 NIfTI grids must not fail on serialization noise."""
        volume = np.zeros((8, 6, 4), dtype=np.float32)
        segmentation = np.zeros((8, 6, 4), dtype=np.int16)
        affine = np.eye(4, dtype=np.float64)
        seg_affine = affine.copy()
        seg_affine[2, 3] += 5e-5
        _write_nifti(tmp_path / "volume0.nii", volume, affine)
        _write_nifti(tmp_path / "segmentation0.nii", segmentation, seg_affine)
        image, mask = LITSDataset(img_dirs=[str(tmp_path)])[0]
        assert image.shape == mask.shape

    def test_preflight_reports_every_invalid_pair(self, tmp_path: Path) -> None:
        """Preflight gives all bad cases before a run instead of failing late."""
        volume = np.zeros((8, 6, 4), dtype=np.float32)
        segmentation = np.zeros((8, 6, 4), dtype=np.int16)
        bad_affine = np.eye(4, dtype=np.float64)
        bad_affine[0, 3] = 1.0
        pairs: list[tuple[str, str]] = []
        for case in ("0", "1"):
            vol_path = tmp_path / f"volume{case}.nii"
            seg_path = tmp_path / f"segmentation{case}.nii"
            _write_nifti(vol_path, volume)
            _write_nifti(seg_path, segmentation, bad_affine)
            pairs.append((str(vol_path), str(seg_path)))
        with pytest.raises(ValueError, match="2 of 2 pairs") as exc_info:
            preflight_pairs(pairs, split_name="validation split")
        assert "volume0.nii" in str(exc_info.value)
        assert "volume1.nii" in str(exc_info.value)

    def test_near_integer_mask_label_is_rejected_exactly(self, tmp_path: Path) -> None:
        """A tolerance must not silently turn fractional labels into classes."""
        volume = np.zeros((8, 6, 4), dtype=np.float32)
        segmentation = np.zeros((8, 6, 4), dtype=np.float32)
        segmentation[0, 0, 0] = 1.00000894
        _write_nifti(tmp_path / "volume0.nii", volume)
        _write_nifti(tmp_path / "segmentation0.nii", segmentation)
        with pytest.raises(ValueError, match="finite integers"):
            LITSDataset(img_dirs=[str(tmp_path)])[0]

    def test_full_decode_rejects_nonfinite_ct_values(self, tmp_path: Path) -> None:
        volume = np.zeros((8, 6, 4), dtype=np.float32)
        volume[0, 0, 0] = np.nan
        segmentation = np.zeros((8, 6, 4), dtype=np.int16)
        vol_path = tmp_path / "volume0.nii"
        seg_path = tmp_path / "segmentation0.nii"
        _write_nifti(vol_path, volume)
        _write_nifti(seg_path, segmentation)
        with pytest.raises(ValueError, match="must all be finite"):
            preflight_pairs([(str(vol_path), str(seg_path))], full_decode=True)


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

    def test_full_pipeline_preserves_nifti_depth_axis(self, tmp_path: Path) -> None:
        """The full deterministic pipeline converts raw HWD once to CDHW."""
        from dense_unet_3d.dataset.prepare_dataset import compose_transforms

        h, w, d = 3, 4, 5
        vol = np.zeros((h, w, d), dtype=np.float32)
        seg = np.zeros((h, w, d), dtype=np.int16)
        for depth in range(d):
            vol[:, :, depth] = depth + 1
        _write_nifti(tmp_path / "volume0.nii", vol)
        _write_nifti(tmp_path / "segmentation0.nii", seg)
        transform = compose_transforms(
            {
                "dataset": {
                    "clamp_hu": False,
                    "resize_img": False,
                    "random_hflip": False,
                    "scale_img": False,
                }
            },
            train=False,
        )
        image, _mask = LITSDataset(
            img_dirs=[str(tmp_path)],
            transform=transform["all_transforms"],
            mask_transform=transform["mask_transforms"],
        )[0]
        assert image.shape == (1, d, h, w)
        for depth in range(d):
            assert torch.all(image[0, depth] == depth + 1)


# ---------------------------------------------------------------------------
# FIX 1: mask must be resized with NEAREST-neighbour, never trilinear
# ---------------------------------------------------------------------------


class TestMaskNearestResize:
    """The seg mask must pass through a nearest-neighbour resize path."""

    def test_omitted_mask_transform_does_not_reuse_image_resize(self, tmp_path: Path) -> None:
        """A geometry-changing image pipeline requires an explicit mask pipeline."""
        from torchvision import transforms

        from dense_unet_3d.dataset.transforms.ReshapeTensor import ReshapeTensor
        from dense_unet_3d.dataset.transforms.Resize import Resize

        vol = np.zeros((4, 4, 2), dtype=np.float32)
        seg = np.zeros((4, 4, 2), dtype=np.int16)
        seg[2:] = 2
        _write_nifti(tmp_path / "volume0.nii", vol)
        _write_nifti(tmp_path / "segmentation0.nii", seg)

        image_transform = transforms.Compose([ReshapeTensor(), Resize((3, 7, 7))])
        with pytest.raises(ValueError, match="matching mask_transform"):
            LITSDataset(img_dirs=[str(tmp_path)], transform=image_transform)[0]

    def test_image_only_intensity_transform_leaves_mask_untouched(self, tmp_path: Path) -> None:
        """An image-only intensity pipeline may omit ``mask_transform``."""
        from torchvision import transforms

        from dense_unet_3d.dataset.transforms.ClampValues import ClampValues
        from dense_unet_3d.dataset.transforms.ReshapeTensor import ReshapeTensor

        vol = np.full((4, 4, 2), 5.0, dtype=np.float32)
        seg = np.zeros((4, 4, 2), dtype=np.int16)
        seg[2:] = 2
        _write_nifti(tmp_path / "volume0.nii", vol)
        _write_nifti(tmp_path / "segmentation0.nii", seg)

        image_transform = transforms.Compose([ReshapeTensor(), ClampValues((0.0, 1.0))])
        _image, mask = LITSDataset(img_dirs=[str(tmp_path)], transform=image_transform)[0]

        expected_mask = torch.from_numpy(seg).permute(2, 0, 1).unsqueeze(0).long()
        assert torch.equal(mask, expected_mask)

    def test_no_interpolated_labels_at_boundaries(self, tmp_path: Path) -> None:
        """A mask with adjacent labels 1 and 2 resized through the dataset
        pipeline must contain ONLY values in {0, 1, 2} — no interpolated
        boundary classes (e.g. 1.5 -> rounds to 2 but the class is invented).
        """
        from dense_unet_3d.dataset.prepare_dataset import compose_transforms

        # Volume small in-plane so Resize up-samples 64->224, exercising interp.
        rng = np.random.default_rng(7)
        vol = rng.uniform(-200, 250, (64, 64, 6)).astype(np.float32)
        # Sharp adjacent label bands: half the image is liver(1), half tumour(2).
        seg = np.zeros((64, 64, 6), dtype=np.int16)
        seg[:32] = 1
        seg[32:] = 2
        _write_nifti(tmp_path / "volume0.nii", vol)
        _write_nifti(tmp_path / "segmentation0.nii", seg)

        config = {
            "dataset": {
                "clamp_hu": True,
                "clamp_hu_range": {"min": -200, "max": 250},
                "resize_img": True,
                "resize_dims": {"D": 12, "H": 224, "W": 224},
                "random_hflip": False,
                "scale_img": False,
            }
        }
        transform = compose_transforms(config, train=False)
        ds = LITSDataset(
            img_dirs=[str(tmp_path)],
            transform=transform["all_transforms"],
            mask_transform=transform["mask_transforms"],
            paired_transform=transform["paired_transforms"],
        )

        _image, mask = ds[0]
        unique = set(torch.unique(mask).tolist())
        assert unique <= {0, 1, 2}, f"interpolated/invented labels present: {unique}"


# ---------------------------------------------------------------------------
# FIX 3: all-background volume -> clear ValueError, not cryptic transpose crash
# ---------------------------------------------------------------------------


class TestEmptyForeground:
    def test_all_zero_seg_raises_value_error(self, dataset_no_transform: LITSDataset) -> None:
        """find_liver on an all-background seg must raise a descriptive ValueError."""
        vol = np.zeros((12, 224, 224), dtype=np.float32)
        seg = np.zeros((12, 224, 224), dtype=np.int16)
        with pytest.raises(ValueError, match="(?i)foreground|liver|background|empty"):
            dataset_no_transform.find_liver((vol, seg))
