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
