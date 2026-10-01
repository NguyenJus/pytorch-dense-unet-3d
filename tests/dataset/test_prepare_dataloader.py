"""Tests for prepare_dataloader / compose_transforms (FIX 2).

TDD: written BEFORE the implementation change.

A validation loader (train=False) must:
  - NOT shuffle, regardless of config['dataset']['shuffle'].
  - NOT apply random augmentations (RandomHorizontalFlip / random ScaleAndPadOrCrop);
    only deterministic transforms (resize/clamp/reshape) survive.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import RandomSampler, SequentialSampler

from dense_unet_3d.dataset.prepare_dataset import (
    compose_transforms,
    preflight_config,
    prepare_dataloader,
    sampling_mode,
)
from dense_unet_3d.dataset.transforms.RandomHorizontalFlip import RandomHorizontalFlip
from dense_unet_3d.dataset.transforms.ScaleAndPadOrCrop import ScaleAndPadOrCrop


def _write_nifti(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data, np.eye(4, dtype=np.float64)), str(path))


def _config(data_dir: str) -> dict:
    return {
        "pathing": {"train_img_dirs": [data_dir], "test_img_dirs": [data_dir]},
        "dataset": {
            "batch_size": 1,
            "clamp_hu": True,
            "clamp_hu_range": {"min": -200, "max": 250},
            "resize_img": True,
            "resize_dims": {"D": 12, "H": 224, "W": 224},
            "random_hflip": True,
            "random_hflip_probability": 0.5,
            "scale_img": True,
            "scale_img_range": {"min": 0.8, "max": 1.2},
            "shuffle": True,
        },
    }


class TestComposeTransformsTrainFlag:
    def test_whole_ct_landmarks_align_in_image_and_mask(self) -> None:
        """Thin slabs sampled by the image must survive at the same mask depth."""
        config = _config("ignored")
        config["dataset"]["clamp_hu"] = False
        config["dataset"]["resize_dims"] = {"D": 12, "H": 3, "W": 4}
        # Half-pixel centers select slices 325 and 525. Legacy floor-nearest
        # selects 300 and 500 and drops both landmarks entirely.
        labels = np.zeros((3, 4, 600), dtype=np.float32)
        labels[:, :, 320:340] = 1
        labels[:, :, 520:540] = 2
        transforms = compose_transforms(config, train=False)
        image = transforms["all_transforms"](labels)
        mask = transforms["mask_transforms"](labels)
        expected = torch.zeros(1, 12, 3, 4)
        expected[:, 6] = 1
        expected[:, 10] = 2
        torch.testing.assert_close(image, expected)
        torch.testing.assert_close(mask, expected)
        assert set(mask.unique().tolist()) == {0.0, 1.0, 2.0}

    def test_val_drops_random_augmentations(self) -> None:
        """train=False -> paired_transforms contains no random augmentation."""
        config = _config("ignored")
        transform = compose_transforms(config, train=False)
        paired = transform["paired_transforms"].transforms
        for t in paired:
            assert not isinstance(t, (RandomHorizontalFlip, ScaleAndPadOrCrop)), (
                f"val pipeline leaked random augmentation: {type(t).__name__}"
            )

    def test_train_keeps_random_augmentations(self) -> None:
        """train=True -> random augmentations remain present."""
        config = _config("ignored")
        transform = compose_transforms(config, train=True)
        paired = transform["paired_transforms"].transforms
        types = {type(t) for t in paired}
        assert RandomHorizontalFlip in types
        assert ScaleAndPadOrCrop in types


class TestValLoaderDeterministic:
    def test_val_loader_does_not_shuffle(self, tmp_path: Path) -> None:
        """train=False loader must use a sequential (non-shuffling) sampler."""
        vol = np.zeros((224, 224, 12), dtype=np.float32)
        seg = np.zeros((224, 224, 12), dtype=np.int16)
        seg[:10, :10, :] = 1
        _write_nifti(tmp_path / "volume0.nii", vol)
        _write_nifti(tmp_path / "segmentation0.nii", seg)

        cfg = _config(str(tmp_path))
        cfg["pathing"]["train_img_dirs"] = []
        loader = prepare_dataloader(cfg, train=False)
        assert isinstance(loader.sampler, SequentialSampler), "val loader must not shuffle"

    def test_train_loader_shuffles(self, tmp_path: Path) -> None:
        """train=True loader honours config shuffle=True (RandomSampler)."""
        vol = np.zeros((224, 224, 12), dtype=np.float32)
        seg = np.zeros((224, 224, 12), dtype=np.int16)
        seg[:10, :10, :] = 1
        _write_nifti(tmp_path / "volume0.nii", vol)
        _write_nifti(tmp_path / "segmentation0.nii", seg)

        loader = prepare_dataloader(_config(str(tmp_path)), train=True)
        assert isinstance(loader.sampler, RandomSampler)

    def test_liver_only_loader_collapses_tumour_label(self, tmp_path: Path) -> None:
        """Phase A loaders fold class 2 into class 1; Phase B loaders retain it."""
        vol = np.zeros((16, 12, 4), dtype=np.float32)
        seg = np.zeros((16, 12, 4), dtype=np.int16)
        seg[0, 0, 0] = 1
        seg[1, 1, 1] = 2
        _write_nifti(tmp_path / "volume0.nii", vol)
        _write_nifti(tmp_path / "segmentation0.nii", seg)

        cfg = _config(str(tmp_path))
        cfg["dataset"]["resize_img"] = False
        cfg["dataset"]["random_hflip"] = False
        cfg["dataset"]["scale_img"] = False
        phase_a_labels = next(iter(prepare_dataloader(cfg, train=True, detect_tumors=False)))[1]
        phase_b_labels = next(iter(prepare_dataloader(cfg, train=True)))[1]

        assert 2 not in torch.unique(phase_a_labels).tolist()
        assert 1 in torch.unique(phase_a_labels).tolist()
        assert 2 in torch.unique(phase_b_labels).tolist()

    def test_shipped_historical_sampling_builds_whole_volume_loader(self, tmp_path: Path) -> None:
        vol = np.zeros((16, 12, 4), dtype=np.float32)
        seg = np.zeros((16, 12, 4), dtype=np.int16)
        _write_nifti(tmp_path / "volume0.nii", vol)
        _write_nifti(tmp_path / "segmentation0.nii", seg)
        config = yaml.safe_load(Path("configs/historical-reference.yaml").read_text())
        config["pathing"]["train_img_dirs"] = [str(tmp_path)]

        loader = prepare_dataloader(config, train=True)

        assert sampling_mode(config) == "whole_volume"
        assert len(loader.dataset) == 1

    @pytest.mark.parametrize("sampling", ["legacy_volume_resize", None, [], {}])
    def test_unknown_sampling_is_rejected(self, tmp_path: Path, sampling: object) -> None:
        config = _config(str(tmp_path))
        config["dataset"]["sampling"] = sampling
        with pytest.raises(ValueError, match="unknown dataset.sampling"):
            prepare_dataloader(config, train=True)
        with pytest.raises(ValueError, match="unknown dataset.sampling"):
            preflight_config(config)

    @pytest.mark.parametrize("dataset", [None, [], "whole_volume"])
    def test_non_mapping_dataset_config_is_rejected(self, dataset: object) -> None:
        with pytest.raises(ValueError, match="dataset must be a mapping"):
            sampling_mode({"dataset": dataset})


class TestValLoaderFromTestDirs:
    def test_val_loader_builds_from_test_dirs_and_yields_batch(self, tmp_path: Path) -> None:
        """Non-dry-run val loader built from test_img_dirs yields a (vol, seg) batch."""
        vol = np.zeros((224, 224, 12), dtype=np.float32)
        seg = np.zeros((224, 224, 12), dtype=np.int16)
        seg[:10, :10, :] = 1  # some foreground
        _write_nifti(tmp_path / "volume0.nii", vol)
        _write_nifti(tmp_path / "segmentation0.nii", seg)

        cfg = _config(str(tmp_path))
        cfg["pathing"]["test_img_dirs"] = [str(tmp_path)]
        # This test covers construction/yielding, not leakage detection.
        cfg["pathing"]["train_img_dirs"] = []
        loader = prepare_dataloader(cfg, train=False)

        batch = next(iter(loader))
        assert len(batch) == 2, "expected (volume, segmentation) pair"
        volume_batch, seg_batch = batch
        assert volume_batch.shape[0] == 1, "batch size should be 1"
        assert seg_batch.shape[0] == 1, "batch size should be 1"

    def test_unset_test_dirs_raises_clear_value_error(self, tmp_path: Path) -> None:
        """prepare_dataloader(train=False) raises ValueError when test_img_dirs is None."""
        cfg = _config(str(tmp_path))

        # Case 1: test_img_dirs is None
        cfg["pathing"]["test_img_dirs"] = None
        with pytest.raises(ValueError, match="pathing.test_img_dirs"):
            prepare_dataloader(cfg, train=False)

        # Case 2: test_img_dirs is [None]
        cfg["pathing"]["test_img_dirs"] = [None]
        with pytest.raises(ValueError, match="pathing.test_img_dirs"):
            prepare_dataloader(cfg, train=False)

        # Case 3: test_img_dirs mixes a null entry with a real dir.
        cfg["pathing"]["test_img_dirs"] = [None, str(tmp_path)]
        with pytest.raises(ValueError, match="pathing.test_img_dirs"):
            prepare_dataloader(cfg, train=False)

        # Case 4: empty list
        cfg["pathing"]["test_img_dirs"] = []
        with pytest.raises(ValueError, match="pathing.test_img_dirs"):
            prepare_dataloader(cfg, train=False)

    def test_overlapping_train_and_val_dirs_raise(self, tmp_path: Path) -> None:
        """The same labeled case cannot be used for both training and validation."""
        vol = np.zeros((16, 12, 4), dtype=np.float32)
        seg = np.zeros((16, 12, 4), dtype=np.int16)
        _write_nifti(tmp_path / "volume0.nii", vol)
        _write_nifti(tmp_path / "segmentation0.nii", seg)
        with pytest.raises(ValueError, match="overlap|leak"):
            prepare_dataloader(_config(str(tmp_path)), train=False)


class TestPreflightConfig:
    def test_checks_both_splits_before_loader_or_model_creation(self, tmp_path: Path) -> None:
        train_dir = tmp_path / "train"
        validation_dir = tmp_path / "validation"
        train_dir.mkdir()
        validation_dir.mkdir()
        volume = np.zeros((8, 6, 4), dtype=np.float32)
        segmentation = np.zeros((8, 6, 4), dtype=np.int16)
        _write_nifti(train_dir / "volume0.nii", volume)
        _write_nifti(train_dir / "segmentation0.nii", segmentation)
        _write_nifti(validation_dir / "volume1.nii", volume)
        _write_nifti(validation_dir / "segmentation1.nii", segmentation)

        cfg = _config(str(train_dir))
        cfg["pathing"]["test_img_dirs"] = [str(validation_dir)]
        assert preflight_config(cfg) == {"train": 1, "validation": 1}
