"""Reconstruction gates prevent an unresolved reference from becoming a long run."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from dense_unet_3d.training.experiment import validate_experiment
from dense_unet_3d.training.runtime import RunSession, config_identity


def diagnostic():
    return yaml.safe_load(Path("configs/reconstruction-diagnostic.yaml").read_text())


@pytest.mark.parametrize(
    "path", ["configs/reconstruction-reference.yaml", "dense_unet_3d/config.yaml"]
)
def test_reference_is_not_launchable(path):
    config = yaml.safe_load(Path(path).read_text())
    with pytest.raises(ValueError, match="not launchable"):
        validate_experiment(config)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("training", "phase_a_targets", "liver_only", "three_class"),
        ("training", "loss_reduction", "weighted_mean", "valid_voxel_mean"),
        ("training", "phase_b_epochs", 1000, "max_updates"),
        ("runtime", "wall_seconds", None, "finite"),
        ("execution", "tf32", True, "FP32"),
        ("experiment", "split_manifest", None, "split_manifest"),
        ("experiment", "mode", "training", "diagnostic mode"),
    ],
)
def test_diagnostic_contract_rejects_unsupported_semantics(section, key, value, message):
    config = diagnostic()
    config[section][key] = value
    with pytest.raises(ValueError, match=message):
        validate_experiment(config)


def test_diagnostic_contract_and_runtime_override(tmp_path):
    config = diagnostic()
    validate_experiment(config)
    config["runtime"]["wall_seconds"] = None
    config["pathing"]["model_save_dir"] = str(tmp_path)
    RunSession(config, wall_seconds=10)
    with pytest.raises(ValueError, match="finite"):
        RunSession(config)


@pytest.mark.parametrize("representation", ["full_fov_resize", "native_tiles_v1"])
@pytest.mark.parametrize("model_as_name", [False, True])
def test_reconstruction_geometry_matches_named_model_before_launch(
    tmp_path, representation, model_as_name
):
    config = diagnostic()
    config["dataset"]["inplane_representation"] = representation
    config["dataset"]["resize_img"] = representation == "full_fov_resize"
    if model_as_name:
        config["model"] = config["model"]["name"]
    validate_experiment(config)
    config["dataset"]["resize_dims"] = {"D": 2, "H": 3, "W": 4}
    config["pathing"]["model_save_dir"] = str(tmp_path)
    for launch in (validate_experiment, RunSession):
        with pytest.raises(ValueError, match=r"dataset.resize_dims must match model.input_shape"):
            launch(config)
    assert list(tmp_path.iterdir()) == []


def test_generic_native_grid_is_not_restricted_by_named_launch_contract():
    config = {
        "dataset": {
            "sampling": "native_slabs",
            "inplane_representation": "native_tiles_v1",
            "resize_dims": {"D": 2, "H": 3, "W": 4},
        }
    }
    validate_experiment(config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("training", "loss_reduction", "weighted_mean"),
        ("training", "phase_a_targets", "liver_only"),
        ("execution", "tf32", True),
        ("dataset", "sampling", "whole_volume"),
        ("experiment", "split_manifest", "another.json"),
    ],
)
def test_experiment_identity_changes_with_semantics(section, key, value):
    config = diagnostic()
    changed = deepcopy(config)
    changed[section][key] = value
    assert config_identity(config) != config_identity(changed)
