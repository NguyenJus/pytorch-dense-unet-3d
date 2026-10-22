"""Explicit reconstruction contracts and diagnostic-only launch gates.

These gates describe local evidence, not permission to call an experiment an
exact reproduction. Historical configurations retain their known API contract.
"""

from __future__ import annotations

import math
from typing import Any

import torch

PREPROCESSING_SCHEMA_VERSION = 1

# One entry per INFERRED or DEVIATION row of the paper-faithful fidelity ledger
# (docs/specs/2026-10-21-paper-faithful-training-run-design.md, section 2).
REQUIRED_FIDELITY_KEYS = (
    "stem_pool_padding",
    "transition_stride",
    "decoder_interpolation",
    "batch_norm_defaults",
    "initialization",
    "split",
    "training_sample",
    "step_unit",
    "mirror",
    "scaling",
    "inference_tiling",
    "batch_size",
    "optimizer_semantics",
    "phase_transfer_reset",
    "phase_b_epochs",
    "best_selection",
    "execution",
)
FIDELITY_LABELS = ("LITERAL", "INFERRED", "DEVIATION")
EXECUTION_KEYS = ("precision", "tf32", "deterministic", "cudnn_benchmark", "autocast")


def preprocessing_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Identify preprocessing code semantics that configuration alone cannot express."""
    from dense_unet_3d.dataset.prepare_dataset import sampling_mode

    dataset = config.get("dataset")
    if (
        isinstance(dataset, dict)
        and dataset.get("training_representation") == "foreground_crops_v1"
    ):
        from dense_unet_3d.dataset.foreground_crops import crop_config
        from dense_unet_3d.dataset.slabs import spatial_config

        # Training crops and inference tiles share one dataset block; record both.
        return {
            "schema_version": PREPROCESSING_SCHEMA_VERSION,
            "sampling": sampling_mode(config),
            "coordinate_grid": "native_tiles_v1",
            "geometry": spatial_config(config["dataset"]),
            "training": "foreground_crops_v1",
            "inference": "native_tiles_v1",
            "training_geometry": crop_config(config["dataset"]),
        }
    if (
        sampling_mode(config) == "native_slabs"
        and config.get("dataset", {}).get("inplane_representation") == "native_tiles_v1"
    ):
        from dense_unet_3d.dataset.slabs import spatial_config

        return {
            "schema_version": PREPROCESSING_SCHEMA_VERSION,
            "sampling": sampling_mode(config),
            "coordinate_grid": "native_tiles_v1",
            "geometry": spatial_config(config["dataset"]),
        }
    return {
        "schema_version": PREPROCESSING_SCHEMA_VERSION,
        "sampling": sampling_mode(config),
        "coordinate_grid": "half_pixel_v1",
    }


def checkpoint_model_metadata(model: torch.nn.Module) -> dict[str, Any]:
    """Owned graphs carry explicit metadata; generic Python API models do not."""
    from dense_unet_3d.model.config import model_metadata
    from dense_unet_3d.model.DenseUNet3d import DenseUNet3d
    from dense_unet_3d.model.reconstruction import FigureSkipReconstruction

    if type(model) in (DenseUNet3d, FigureSkipReconstruction):
        return model_metadata(model)
    return {}


def validate_experiment(config: dict[str, Any]) -> None:
    # All launch paths share these two canonical data representations. Validate
    # before preflight so a misspelled mode cannot trigger an expensive scan.
    preprocessing_identity(config)
    # Checked before the early returns so a bad block fails before the expensive preflight.
    if config.get("execution") is not None:
        _execution_settings(config["execution"])
    experiment = config.get("experiment")
    if experiment is None:
        return
    if not isinstance(experiment, dict):
        raise ValueError("experiment must be a mapping")
    kind = experiment.get("kind")
    if kind == "historical_reference":
        return
    if kind != "reconstruction":
        raise ValueError("Unknown experiment.kind")
    if not experiment.get("name"):
        raise ValueError("Reconstruction requires an explicit experiment name")
    training = config["training"]
    if training.get("phase_a_targets") != "three_class":
        raise ValueError("Reconstruction requires explicit three_class phase A targets")
    if training.get("loss_reduction") != "valid_voxel_mean":
        raise ValueError("Reconstruction requires explicit valid_voxel_mean loss")
    mode = experiment.get("mode")
    # A training run counts passes over the cases; diagnostics count updates.
    step_unit = "training_pass" if mode == "training" else "minibatch_update"
    if training.get("step_unit") != step_unit:
        raise ValueError(f"Reconstruction requires explicit {step_unit} interpretation")
    if config.get("dataset", {}).get("sampling") != "native_slabs":
        raise ValueError("Reconstruction requires explicit native_slabs sampling")
    if not config.get("model"):
        raise ValueError("Reconstruction requires explicit model configuration")
    from dense_unet_3d.dataset.slabs import spatial_config
    from dense_unet_3d.model.config import canonical_model_config, model_fingerprint

    model_config = config["model"]
    # Validate the complete named identity without allocating the model.
    model_fingerprint(model_config)
    named_model = canonical_model_config(
        model_config if isinstance(model_config, str) else model_config["name"]
    )
    geometry = spatial_config(config["dataset"])
    dataset_shape = [1, geometry["window"], geometry["height"], geometry["width"]]
    if dataset_shape != named_model["input_shape"]:
        raise ValueError(
            "Reconstruction dataset.resize_dims must match model.input_shape: "
            f"dataset {dataset_shape}, model {named_model['input_shape']}"
        )
    if not experiment.get("split_manifest"):
        raise ValueError("Reconstruction requires a recorded split_manifest")
    execution = config.get("execution", {})
    if mode == "training":
        _validate_training_run(config, execution)
        return
    if not _strict_execution(execution):
        raise ValueError("Only the explicit FP32 execution reference is supported")
    if mode == "reference":
        raise ValueError("Reference reconstruction is not launchable: unresolved research gates")
    if mode != "diagnostic":
        raise ValueError("Reconstruction supports training or bounded diagnostic mode only")
    max_updates = experiment.get("max_updates")
    if type(max_updates) is not int or max_updates < 1:
        raise ValueError("Diagnostic max_updates must be a positive integer")
    updates = sum(
        training.get(f"{phase}_epochs", default)
        * training.get(f"{phase}_steps_per_epoch", training.get("steps_per_epoch", 10))
        for phase, default in (("phase_a", 100), ("phase_b", 1000))
    )
    if updates > max_updates:
        raise ValueError("Configured schedule exceeds the declared diagnostic max_updates")
    wall = config.get("runtime", {}).get("wall_seconds")
    if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
        raise ValueError("Reconstruction diagnostics require finite runtime.wall_seconds")


def _validate_training_run(config: dict[str, Any], execution: Any) -> None:
    """Every interpretation of a training run is explicit; nothing is defaulted."""
    training, dataset = config["training"], config["dataset"]
    for key, value in (
        ("phase_transfer_policy", "best_weights_fresh_optimizer"),
        ("optimizer_semantics", "pytorch_gradient_buffer"),
    ):
        if training.get(key) != value:
            raise ValueError(f"Training mode requires explicit training.{key}={value}")
    for key, value in (
        ("training_representation", "foreground_crops_v1"),
        ("inplane_representation", "native_tiles_v1"),
    ):
        if dataset.get(key) != value:
            raise ValueError(f"Training mode requires explicit dataset.{key}={value}")
    _execution_settings(execution)
    runtime = config.get("runtime", {})
    for key in ("wall_seconds", "budget_seconds"):
        value = runtime.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Training mode requires finite positive runtime.{key}")
    fidelity = config["experiment"].get("fidelity")
    if not isinstance(fidelity, dict):
        raise ValueError("Training mode requires an experiment.fidelity mapping")
    missing = [key for key in REQUIRED_FIDELITY_KEYS if key not in fidelity]
    if missing:
        raise ValueError(f"experiment.fidelity is missing {', '.join(missing)}")
    for key, entry in fidelity.items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"label", "evidence"}
            or entry["label"] not in FIDELITY_LABELS
            or not isinstance(entry["evidence"], str)
            or not entry["evidence"].strip()
        ):
            raise ValueError(
                f"experiment.fidelity.{key} must be {{label: LITERAL|INFERRED|DEVIATION, "
                "evidence: non-empty string}"
            )


def _execution_settings(execution: Any) -> dict[str, Any]:
    """Validate an execution block holding exactly the EXECUTION_KEYS."""
    if not isinstance(execution, dict) or set(execution) != set(EXECUTION_KEYS):
        raise ValueError(
            f"Unsupported execution settings: execution requires exactly {', '.join(EXECUTION_KEYS)}"
        )
    if execution["precision"] != "fp32":
        raise ValueError("Unsupported execution settings: precision must be fp32")
    if any(
        type(execution[key]) is not bool for key in ("tf32", "deterministic", "cudnn_benchmark")
    ):
        raise ValueError(
            "Unsupported execution settings: tf32, deterministic and cudnn_benchmark are booleans"
        )
    if execution["autocast"] not in ("none", "bf16"):
        raise ValueError("Unsupported execution settings: autocast must be none or bf16")
    if execution["deterministic"] and execution["cudnn_benchmark"]:
        raise ValueError("Unsupported execution settings: deterministic excludes cudnn_benchmark")
    return execution


def _strict_execution(execution: Any) -> bool:
    """Reference and diagnostic runs keep the strict FP32 path."""
    settings = _execution_settings(execution)
    return (
        settings["tf32"] is False
        and settings["cudnn_benchmark"] is False
        and settings["autocast"] == "none"
    )


def configure_execution(config: dict[str, Any]) -> None:
    """Apply only execution modes implemented and recorded in this experiment."""
    execution = config.get("execution")
    if execution is None:
        return
    settings = _execution_settings(execution)
    torch.backends.cuda.matmul.allow_tf32 = settings["tf32"]
    torch.backends.cudnn.allow_tf32 = settings["tf32"]
    torch.backends.cudnn.benchmark = settings["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = not settings["cudnn_benchmark"]
    # Without benchmark, cuDNN algorithm choice remains deterministic. CUDA
    # MaxPool3d backward uses atomic accumulation, so strict global determinism
    # is a separate explicit setting, with replay tolerances recorded by GPU
    # diagnostics. Autocast is applied by the training step, not here.
    torch.use_deterministic_algorithms(settings["deterministic"])


def experiment_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Persist choices that cannot be reconstructed from weight tensor shapes."""
    return {
        "experiment": config.get("experiment"),
        "execution": config.get("execution"),
        "training_semantics": {
            key: config.get("training", {}).get(key, default)
            for key, default in (
                ("phase_a_targets", "liver_only"),
                ("loss_reduction", "weighted_mean"),
                ("step_unit", "minibatch_update"),
                ("phase_transfer_policy", "best_weights_fresh_optimizer"),
                ("optimizer_semantics", "pytorch_gradient_buffer"),
            )
        },
        # inference_batch is excluded for the same reason as in config_identity.
        "dataset_config": {
            k: v for k, v in config.get("dataset", {}).items() if k != "inference_batch"
        },
        "preprocessing_identity": preprocessing_identity(config),
    }
