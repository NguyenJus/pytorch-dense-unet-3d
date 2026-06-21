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
from torch.utils.data import RandomSampler, SequentialSampler

from dense_unet_3d.dataset.prepare_dataset import compose_transforms, prepare_dataloader
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

        loader = prepare_dataloader(_config(str(tmp_path)), train=False)
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
