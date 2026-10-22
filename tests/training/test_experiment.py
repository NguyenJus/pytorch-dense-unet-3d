"""Reconstruction gates prevent an unresolved reference from becoming a long run."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from dense_unet_3d.dataset.foreground_crops import crop_config
from dense_unet_3d.dataset.slabs import spatial_config
from dense_unet_3d.training.experiment import (
    EXECUTION_KEYS,
    REQUIRED_FIDELITY_KEYS,
    configure_execution,
    experiment_metadata,
    preprocessing_identity,
    validate_experiment,
)
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
        ("experiment", "mode", "training", "training_pass"),
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


FIDELITY = {
    key: {"label": "INFERRED", "evidence": f"spec section 2 row for {key}"}
    for key in REQUIRED_FIDELITY_KEYS
}


def training_run():
    """A `mode: training` config mirroring the launchable run config."""
    config = diagnostic()
    config["dataset"].update(
        {
            "batch_size": 2,
            "resize_img": False,
            "inplane_representation": "native_tiles_v1",
            "training_representation": "foreground_crops_v1",
            "passes_per_epoch": 10,
            "tumor_centre_probability": 0.5,
            "foreground_voxel_cache": 20000,
            "sampling_seed": 20261021,
            "random_hflip": True,
            "scale_img": True,
        }
    )
    config["training"].update(
        {"phase_a_epochs": 100, "phase_b_epochs": 100, "step_unit": "training_pass"}
    )
    config["runtime"].update({"wall_seconds": 3600, "budget_seconds": 172800})
    config["experiment"] = {
        "kind": "reconstruction",
        "name": "figure_skip_paper_faithful_training",
        "mode": "training",
        "split_manifest": "docs/research/2026-10-21-split-manifest.json",
        "fidelity": deepcopy(FIDELITY),
    }
    config["execution"] = {
        "precision": "fp32",
        "tf32": True,
        "deterministic": False,
        "cudnn_benchmark": True,
        "autocast": "none",
    }
    return config


def test_crop_training_identity_records_both_representations():
    config = training_run()
    identity = preprocessing_identity(config)
    assert identity == {
        "schema_version": 1,
        "sampling": "native_slabs",
        "coordinate_grid": "native_tiles_v1",
        "geometry": spatial_config(config["dataset"]),
        "training": "foreground_crops_v1",
        "inference": "native_tiles_v1",
        "training_geometry": crop_config(config["dataset"]),
    }
    plain = deepcopy(config)
    for key in ("training_representation", "random_hflip", "scale_img"):
        del plain["dataset"][key]
    assert preprocessing_identity(plain) == {
        key: identity[key] for key in ("schema_version", "sampling", "coordinate_grid", "geometry")
    }


def test_training_mode_accepts_the_full_explicit_contract():
    config = training_run()
    validate_experiment(config)
    config["execution"]["autocast"] = "bf16"
    validate_experiment(config)
    # The ledger labels travel into every checkpoint with the experiment block.
    assert experiment_metadata(config)["experiment"]["fidelity"] == FIDELITY


@pytest.mark.parametrize(
    ("section", "key", "message"),
    [
        ("training", "phase_a_targets", "three_class"),
        ("training", "loss_reduction", "valid_voxel_mean"),
        ("training", "step_unit", "training_pass"),
        ("training", "phase_transfer_policy", "phase_transfer_policy"),
        ("training", "optimizer_semantics", "optimizer_semantics"),
        # The shared spatial contract rejects these before the training gate.
        ("dataset", "training_representation", "augmentation is unresolved"),
        ("dataset", "inplane_representation", "full_fov_resize requires resize_img"),
        ("experiment", "split_manifest", "split_manifest"),
        ("experiment", "fidelity", "fidelity"),
        ("runtime", "wall_seconds", "wall_seconds"),
        ("runtime", "budget_seconds", "budget_seconds"),
        *(("execution", key, "execution requires exactly") for key in EXECUTION_KEYS),
    ],
)
def test_training_mode_rejects_each_missing_interpretation(section, key, message):
    config = training_run()
    del config[section][key]
    with pytest.raises(ValueError, match=message):
        validate_experiment(config)


def test_training_mode_requires_crop_training_without_augmentation():
    config = training_run()
    for key in ("training_representation", "random_hflip", "scale_img"):
        del config["dataset"][key]
    with pytest.raises(ValueError, match="dataset.training_representation=foreground_crops_v1"):
        validate_experiment(config)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("training", "step_unit", "minibatch_update", "training_pass"),
        ("training", "phase_transfer_policy", "best_weights_keep_optimizer", "phase_transfer"),
        ("dataset", "inplane_representation", "full_fov_resize", "full_fov_resize requires"),
        ("runtime", "budget_seconds", float("inf"), "budget_seconds"),
        ("runtime", "budget_seconds", 0, "budget_seconds"),
        ("runtime", "budget_seconds", True, "budget_seconds"),
        ("runtime", "wall_seconds", None, "wall_seconds"),
        ("execution", "autocast", "fp16", "autocast"),
        ("execution", "autocast", None, "autocast"),
        ("execution", "cudnn_benchmark", "yes", "booleans"),
        ("execution", "precision", "bf16", "fp32"),
        ("execution", "deterministic", True, "excludes cudnn_benchmark"),
        ("execution", "extra", 1, "execution requires exactly"),
    ],
)
def test_training_mode_rejects_malformed_settings(section, key, value, message):
    config = training_run()
    config[section][key] = value
    with pytest.raises(ValueError, match=message):
        validate_experiment(config)


@pytest.mark.parametrize(
    "entry",
    [
        None,
        "INFERRED",
        {"label": "INFERRED"},
        {"evidence": "Fig. 1"},
        {"label": "GUESSED", "evidence": "Fig. 1"},
        {"label": "INFERRED", "evidence": "  "},
        {"label": "INFERRED", "evidence": 3},
        {"label": "INFERRED", "evidence": "Fig. 1", "note": "extra"},
    ],
)
@pytest.mark.parametrize("key", [REQUIRED_FIDELITY_KEYS[0], "extra_recorded_piece"])
def test_training_mode_rejects_malformed_fidelity_entries(key, entry):
    config = training_run()
    config["experiment"]["fidelity"][key] = entry
    with pytest.raises(ValueError, match=f"fidelity.{key}"):
        validate_experiment(config)


@pytest.mark.parametrize("key", REQUIRED_FIDELITY_KEYS)
def test_training_mode_requires_every_fidelity_key(key):
    config = training_run()
    del config["experiment"]["fidelity"][key]
    with pytest.raises(ValueError, match=f"fidelity is missing {key}"):
        validate_experiment(config)


def test_fidelity_gate_requires_a_mapping_of_unique_keys():
    assert len(REQUIRED_FIDELITY_KEYS) == len(set(REQUIRED_FIDELITY_KEYS)) == 17
    config = training_run()
    config["experiment"]["fidelity"] = []
    with pytest.raises(ValueError, match="fidelity mapping"):
        validate_experiment(config)


def test_reference_still_refuses_and_diagnostic_stays_strict():
    config = training_run()
    config["experiment"]["mode"] = "reference"
    config["training"]["step_unit"] = "minibatch_update"
    with pytest.raises(ValueError, match="FP32"):
        validate_experiment(config)
    config["execution"].update({"tf32": False, "cudnn_benchmark": False})
    with pytest.raises(ValueError, match="not launchable"):
        validate_experiment(config)
    strict = diagnostic()
    validate_experiment(strict)
    for key, value in (("cudnn_benchmark", True), ("autocast", "bf16")):
        loose = deepcopy(strict)
        loose["execution"][key] = value
        with pytest.raises(ValueError, match="FP32"):
            validate_experiment(loose)


def test_execution_block_requires_exactly_the_five_keys():
    three_key = {"precision": "fp32", "tf32": False, "deterministic": False}
    with pytest.raises(ValueError, match="requires exactly precision, tf32, deterministic"):
        configure_execution({"execution": three_key})
    configure_execution({})
