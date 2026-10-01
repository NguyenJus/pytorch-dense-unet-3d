"""Explicit reconstruction contracts and diagnostic-only launch gates.

These gates describe local evidence, not permission to call an experiment an
exact reproduction. Historical configurations retain their known API contract.
"""

from __future__ import annotations

import math
from typing import Any

import torch

PREPROCESSING_SCHEMA_VERSION = 1


def preprocessing_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Identify preprocessing code semantics that configuration alone cannot express."""
    from dense_unet_3d.dataset.prepare_dataset import sampling_mode

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
    if training.get("step_unit") != "minibatch_update":
        raise ValueError("Reconstruction requires explicit minibatch_update interpretation")
    if config.get("dataset", {}).get("sampling") != "native_slabs":
        raise ValueError("Reconstruction requires explicit native_slabs sampling")
    if not config.get("model"):
        raise ValueError("Reconstruction requires explicit model configuration")
    if not experiment.get("split_manifest"):
        raise ValueError("Reconstruction requires a recorded split_manifest")
    execution = config.get("execution", {})
    if not _supported_execution(execution):
        raise ValueError("Only the explicit FP32 execution reference is supported")
    mode = experiment.get("mode")
    if mode == "reference":
        raise ValueError("Reference reconstruction is not launchable: unresolved research gates")
    if mode != "diagnostic":
        raise ValueError("Reconstruction currently supports bounded diagnostic mode only")
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


def _supported_execution(execution: Any) -> bool:
    return (
        isinstance(execution, dict)
        and set(execution) == {"precision", "tf32", "deterministic"}
        and execution["precision"] == "fp32"
        and execution["tf32"] is False
        and type(execution["deterministic"]) is bool
    )


def configure_execution(config: dict[str, Any]) -> None:
    """Apply only execution modes implemented and recorded in this experiment."""
    execution = config.get("execution")
    if execution is None:
        return
    if not _supported_execution(execution):
        raise ValueError("Unsupported execution settings; use the FP32 reference")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # cuDNN algorithm choice remains deterministic. CUDA MaxPool3d backward
    # uses atomic accumulation, so strict global determinism is a separate
    # explicit setting, with replay tolerances recorded by GPU diagnostics.
    torch.use_deterministic_algorithms(execution["deterministic"])


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
        "dataset_config": config.get("dataset", {}),
        "preprocessing_identity": preprocessing_identity(config),
    }
