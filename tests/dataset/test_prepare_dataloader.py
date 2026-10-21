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

import dense_unet_3d.dataset.prepare_dataset as prepare_dataset_module
from dense_unet_3d.dataset.foreground_crops import CropSampler, ForegroundCropDataset
from dense_unet_3d.dataset.prepare_dataset import (
    compose_transforms,
    preflight_config,
    prepare_dataloader,
    prepare_dataset,
    sampling_mode,
)
from dense_unet_3d.dataset.slabs import NativeSlabDataset
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
    @pytest.mark.parametrize("representation", ["full_fov_resize", "native_tiles_v1"])
    @pytest.mark.parametrize(
        ("update", "message"),
        [
            ({"random_hflip": True}, "augmentation is unresolved"),
            ({"scale_img": True}, "augmentation is unresolved"),
            ({"scale_img": "false"}, "augmentation flags must be booleans"),
            ({"resize_dims": {"D": 0, "H": 3, "W": 4}}, "dimensions must be positive"),
            ({"resize_dims": {"D": 2, "H": 3.5, "W": 4}}, "dimensions must be integers"),
            ({"resize_dims": {"D": 2, "H": 3}}, "dimensions must be integers"),
        ],
    )
    def test_native_spatial_errors_are_rejected_before_discovery(
        self, monkeypatch, representation, update, message
    ):
        cfg = _config("unused")
        cfg["dataset"].update(
            sampling="native_slabs",
            inplane_representation=representation,
            resize_img=representation == "full_fov_resize",
            random_hflip=False,
            scale_img=False,
        )
        cfg["dataset"].update(update)

        def unexpected_discovery(_directories):
            pytest.fail("invalid spatial settings must fail before scanning source data")

        monkeypatch.setattr(
            "dense_unet_3d.dataset.prepare_dataset.discover_pairs", unexpected_discovery
        )
        with pytest.raises(ValueError, match=message):
            preflight_config(cfg)

    @pytest.mark.parametrize("representation", ["full_fov_resize", "native_tiles_v1"])
    def test_native_resize_contract_is_checked(self, representation):
        cfg = _config("unused")
        cfg["dataset"].update(
            sampling="native_slabs",
            inplane_representation=representation,
            resize_img=representation != "full_fov_resize",
            random_hflip=False,
            scale_img=False,
        )
        with pytest.raises(ValueError, match="requires resize_img="):
            preflight_config(cfg)

    @pytest.mark.parametrize("representation", [None, "full_fov_resize", "native_tiles_v1"])
    def test_checks_both_splits_before_loader_or_model_creation(
        self, tmp_path: Path, representation
    ) -> None:
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
        if representation is not None:
            cfg["dataset"].update(
                sampling="native_slabs",
                inplane_representation=representation,
                resize_dims={"D": 2, "H": 3, "W": 4},
                random_hflip=False,
                scale_img=False,
            )
            # Exercise the representation-aware omitted resize default as well
            # as preserving arbitrary geometry for standalone raw-data audits.
            cfg["dataset"].pop("resize_img")
        assert preflight_config(cfg) == {"train": 1, "validation": 1}


def _crop_config(train_dir: str, **dataset_update) -> dict:
    cfg = _config(train_dir)
    cfg["dataset"].update(
        sampling="native_slabs",
        inplane_representation="native_tiles_v1",
        training_representation="foreground_crops_v1",
        resize_img=False,
        clamp_hu=False,
        resize_dims={"D": 4, "H": 8, "W": 8},
        passes_per_epoch=3,
        tumor_centre_probability=0.5,
        foreground_voxel_cache=50,
        sampling_seed=7,
        batch_size=2,
        **dataset_update,
    )
    return cfg


def _write_crop_cases(root: Path, ids: tuple[int, ...] = (0, 1, 2)) -> None:
    root.mkdir(exist_ok=True)
    labels = np.zeros((20, 18, 10), dtype=np.uint8)
    labels[4:16, 3:15, 2:9] = 1
    labels[6:8, 5:7, 4:6] = 2
    for index in ids:
        _write_nifti(root / f"volume-{index}.nii", labels.astype(np.float32))
        _write_nifti(root / f"segmentation-{index}.nii", labels)


def _crop_split(tmp_path: Path, **dataset_update) -> dict:
    _write_crop_cases(tmp_path / "train")
    _write_crop_cases(tmp_path / "val", (7,))
    cfg = _crop_config(str(tmp_path / "train"), **dataset_update)
    cfg["pathing"]["test_img_dirs"] = [str(tmp_path / "val")]
    return cfg


class TestForegroundCropWiring:
    def test_train_returns_crop_dataset_and_validation_native_tiles(self, tmp_path: Path) -> None:
        cfg = _crop_split(tmp_path)
        assert isinstance(prepare_dataset(cfg, True), ForegroundCropDataset)
        assert type(prepare_dataset(cfg, False)) is NativeSlabDataset

    def test_cache_path_uses_run_directory(self, tmp_path: Path) -> None:
        cfg = _crop_split(tmp_path)
        cfg["pathing"].update(model_save_dir=str(tmp_path / "models"), run_name="run")
        prepare_dataloader(cfg, train=True)
        assert (tmp_path / "models" / "run" / "foreground-index.npz").is_file()

    def test_loader_length_sampler_and_worker_settings(self, tmp_path: Path) -> None:
        # 3 cases, 3 passes, batch 2 -> ceil(9 / 2) = 5
        cfg = _crop_split(tmp_path, num_workers=2, pin_memory=True, persistent_workers=True)
        loader = prepare_dataloader(cfg, train=True)
        assert isinstance(loader.dataset, ForegroundCropDataset)
        assert isinstance(loader.sampler, CropSampler)
        assert len(loader) == 5
        assert (loader.num_workers, loader.pin_memory, loader.persistent_workers) == (2, True, True)
        expected = torch.Generator().manual_seed(7)
        assert torch.equal(loader.sampler.generator.get_state(), expected.get_state())

    def test_worker_settings_do_not_reach_validation_loader(self, tmp_path: Path) -> None:
        cfg = _crop_split(tmp_path, num_workers=2, pin_memory=True, persistent_workers=True)
        loader = prepare_dataloader(cfg, train=False)
        assert isinstance(loader.dataset, NativeSlabDataset)
        assert (loader.num_workers, loader.pin_memory, loader.persistent_workers) == (
            0,
            False,
            False,
        )

    def test_defaults_and_validation_of_loader_keys(self, tmp_path: Path) -> None:
        loader = prepare_dataloader(_crop_split(tmp_path), train=True)
        assert (loader.num_workers, loader.pin_memory, loader.persistent_workers) == (
            0,
            False,
            False,
        )
        with pytest.raises(ValueError, match="persistent_workers must be a boolean"):
            prepare_dataloader(_crop_split(tmp_path, persistent_workers="yes"), train=True)
        with pytest.raises(ValueError, match="shuffle"):
            prepare_dataloader(_crop_split(tmp_path, shuffle=False), train=True)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"num_workers": -1}, "num_workers must be a non-negative integer"),
            ({"num_workers": True}, "num_workers must be a non-negative integer"),
            ({"num_workers": "4"}, "num_workers must be a non-negative integer"),
            ({"pin_memory": "yes"}, "pin_memory must be a boolean"),
            ({"persistent_workers": True}, "persistent_workers requires"),
            ({"shuffle": False}, "shuffle"),
        ],
    )
    def test_loader_keys_rejected_before_dataset_is_built(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict, match: str
    ) -> None:
        cfg = _crop_split(tmp_path, **overrides)
        monkeypatch.setattr(
            prepare_dataset_module, "prepare_dataset", lambda *a, **k: pytest.fail("built")
        )
        with pytest.raises(ValueError, match=match):
            prepare_dataloader(cfg, train=True)

    def test_augmentation_flags_still_rejected_for_plain_native_tiles(self) -> None:
        cfg = _config("unused")
        cfg["dataset"].update(
            sampling="native_slabs",
            inplane_representation="native_tiles_v1",
            resize_img=False,
        )
        with pytest.raises(ValueError, match="augmentation is unresolved"):
            preflight_config(cfg)

    def test_preflight_accepts_crop_config(self, tmp_path: Path) -> None:
        assert preflight_config(_crop_split(tmp_path)) == {"train": 3, "validation": 1}
