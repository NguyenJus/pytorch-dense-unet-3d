from copy import deepcopy

import pytest
import torch
from torch import nn

from dense_unet_3d.cli import _load_model_from_checkpoint, _TinyStub
from dense_unet_3d.model.config import canonical_model_config
from dense_unet_3d.training.experiment import (
    configure_execution,
    experiment_metadata,
    preprocessing_identity,
)


def test_synthetic_checkpoint_cannot_impersonate_named_graph(tmp_path):
    path = tmp_path / "synthetic.pt"
    model = _TinyStub(nn.Conv3d(1, 3, 1))
    torch.save({"model_state_dict": model.state_dict()}, path)
    config = {"model": canonical_model_config("figure_skip_reconstruction_v1")}
    with pytest.raises(ValueError, match="Synthetic stub"):
        _load_model_from_checkpoint(str(path), torch.device("cpu"), config)
    assert isinstance(_load_model_from_checkpoint(str(path), torch.device("cpu")), _TinyStub)


def test_malformed_graph_metadata_is_not_bypassed_by_stub_keys(tmp_path):
    path = tmp_path / "malformed.pt"
    model = _TinyStub(nn.Conv3d(1, 3, 1))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"name": "bogus"},
            "model_fingerprint": "bogus",
        },
        path,
    )
    with pytest.raises(ValueError, match="Unsupported model"):
        _load_model_from_checkpoint(str(path), torch.device("cpu"))


def test_inference_restores_execution_and_rejects_preprocessing_mismatch(tmp_path):
    path = tmp_path / "reference.pt"
    model = _TinyStub(nn.Conv3d(1, 3, 1))
    execution = {
        "precision": "fp32",
        "tf32": False,
        "deterministic": True,
        "cudnn_benchmark": False,
        "autocast": "none",
    }
    dataset = {"sampling": "native_slabs", "resize_dims": {"D": 12, "H": 224, "W": 224}}
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "execution": execution,
            "dataset_config": dataset,
            "preprocessing_identity": preprocessing_identity({"dataset": dataset}),
        },
        path,
    )
    prior = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
    )
    try:
        torch.backends.cudnn.allow_tf32 = True
        _load_model_from_checkpoint(str(path), torch.device("cpu"))
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
        assert torch.are_deterministic_algorithms_enabled()
        wrong = deepcopy(dataset)
        wrong["sampling"] = "whole_volume"
        with pytest.raises(ValueError, match="preprocessing mismatch"):
            _load_model_from_checkpoint(str(path), torch.device("cpu"), {"dataset": wrong})
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior[0]
        torch.backends.cudnn.allow_tf32 = prior[1]
        torch.backends.cudnn.benchmark = prior[2]
        torch.backends.cudnn.deterministic = prior[3]
        torch.use_deterministic_algorithms(prior[4])


def test_training_run_execution_and_crop_identity_round_trip(tmp_path):
    path = tmp_path / "training.pt"
    model = _TinyStub(nn.Conv3d(1, 3, 1))
    config = {
        "dataset": {
            "sampling": "native_slabs",
            "inplane_representation": "native_tiles_v1",
            "resize_img": False,
            "training_representation": "foreground_crops_v1",
            "passes_per_epoch": 10,
            "tumor_centre_probability": 0.5,
            "foreground_voxel_cache": 20000,
            "sampling_seed": 20261021,
            "random_hflip": True,
            "random_hflip_probability": 0.5,
            "scale_img": True,
            "scale_img_range": {"min": 0.8, "max": 1.2},
        },
        "execution": {
            "precision": "fp32",
            "tf32": True,
            "deterministic": False,
            "cudnn_benchmark": True,
            "autocast": "bf16",
        },
    }
    torch.save({"model_state_dict": model.state_dict(), **experiment_metadata(config)}, path)
    prior = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True)
        _load_model_from_checkpoint(str(path), torch.device("cpu"), config)
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
        assert torch.backends.cudnn.benchmark
        assert not torch.backends.cudnn.deterministic
        assert not torch.are_deterministic_algorithms_enabled()
        wrong = deepcopy(config)
        wrong["dataset"]["sampling_seed"] = 1
        with pytest.raises(ValueError, match="preprocessing implementation mismatch"):
            _load_model_from_checkpoint(str(path), torch.device("cpu"), wrong)
        strict = deepcopy(config)
        strict["execution"].update({"deterministic": True})
        with pytest.raises(ValueError, match="deterministic excludes cudnn_benchmark"):
            configure_execution(strict)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior[0]
        torch.backends.cudnn.allow_tf32 = prior[1]
        torch.backends.cudnn.benchmark = prior[2]
        torch.backends.cudnn.deterministic = prior[3]
        torch.use_deterministic_algorithms(prior[4])


def test_current_preprocessing_metadata_round_trips(tmp_path):
    path = tmp_path / "current.pt"
    model = _TinyStub(nn.Conv3d(1, 3, 1))
    config = {"dataset": {"sampling": "whole_volume"}}
    torch.save({"model_state_dict": model.state_dict(), **experiment_metadata(config)}, path)

    loaded = _load_model_from_checkpoint(str(path), torch.device("cpu"), config)

    assert isinstance(loaded, _TinyStub)


def test_legacy_preprocessing_requires_explicit_opt_in_and_warns(tmp_path):
    path = tmp_path / "legacy.pt"
    model = _TinyStub(nn.Conv3d(1, 3, 1))
    torch.save({"model_state_dict": model.state_dict()}, path)
    config = {"dataset": {"sampling": "whole_volume"}}

    with pytest.raises(ValueError, match="--allow-legacy-preprocessing"):
        _load_model_from_checkpoint(str(path), torch.device("cpu"), config)
    with pytest.warns(RuntimeWarning, match="LEGACY PREPROCESSING OVERRIDE"):
        loaded = _load_model_from_checkpoint(
            str(path),
            torch.device("cpu"),
            config,
            allow_legacy_preprocessing=True,
        )

    assert isinstance(loaded, _TinyStub)


def test_legacy_opt_in_does_not_bypass_known_mismatch(tmp_path):
    path = tmp_path / "known-mismatch.pt"
    model = _TinyStub(nn.Conv3d(1, 3, 1))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset_config": {"sampling": "native_slabs"},
        },
        path,
    )

    with pytest.raises(ValueError, match="preprocessing mismatch"):
        _load_model_from_checkpoint(
            str(path),
            torch.device("cpu"),
            {"dataset": {"sampling": "whole_volume"}},
            allow_legacy_preprocessing=True,
        )


def test_resize_behavior_is_part_of_checkpoint_preprocessing_contract(tmp_path):
    path = tmp_path / "resize-mismatch.pt"
    model = _TinyStub(nn.Conv3d(1, 3, 1))
    checkpoint_config = {"dataset": {"sampling": "whole_volume", "resize_img": True}}
    torch.save(
        {"model_state_dict": model.state_dict(), **experiment_metadata(checkpoint_config)},
        path,
    )

    with pytest.raises(ValueError, match="preprocessing mismatch: resize_img"):
        _load_model_from_checkpoint(
            str(path),
            torch.device("cpu"),
            {"dataset": {"sampling": "whole_volume", "resize_img": False}},
            allow_legacy_preprocessing=True,
        )


def test_legacy_opt_in_does_not_bypass_version_mismatch(tmp_path):
    path = tmp_path / "version-mismatch.pt"
    model = _TinyStub(nn.Conv3d(1, 3, 1))
    config = {"dataset": {"sampling": "whole_volume"}}
    identity = preprocessing_identity(config)
    identity["coordinate_grid"] = "legacy_floor_v0"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset_config": config["dataset"],
            "preprocessing_identity": identity,
        },
        path,
    )

    with pytest.raises(ValueError, match="implementation mismatch"):
        _load_model_from_checkpoint(
            str(path),
            torch.device("cpu"),
            config,
            allow_legacy_preprocessing=True,
        )
