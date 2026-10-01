"""Strict named model identities; no inferred topology from tensor shapes/counts."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from dense_unet_3d.model.DenseUNet3d import DenseUNet3d
from dense_unet_3d.model.reconstruction import FigureSkipReconstruction

HISTORICAL = "historical_reduced"
FIGURE = "figure_skip_reconstruction_v1"

_CONFIGS: dict[str, dict[str, Any]] = {
    HISTORICAL: {
        "name": HISTORICAL,
        "schema_version": 1,
        "status": "historical_deviation",
        "input_shape": [1, 12, 224, 224],
        "block_counts": [2, 6, 12, 18],
        "bottlenecks": [128, 128, 128, 128],
        "growth": 32,
        "compression": 0.5,
        "stem": "conv7_stride2_symmetric_pad3_bn_relu",
        "pool": "max2_stride2_valid",
        "transition": "bn_conv1_conv1_stride122",
        "dense_order": "conv1_bn_relu_depthwise3_pointwise1_bn_relu",
        "decoder": "depthwise3_pointwise1_bn_relu",
        "decoder_widths": [504, 224, 192, 96, 64],
        "skips": ["dense_block3", "dense_block2", "dense_block1", "stem", "none"],
        "resize": "main_only_trilinear_align_corners_true",
        "batch_norm": {"eps": 1e-5, "momentum": 0.1, "running_variance": "unbiased"},
        "initialization": "pytorch_conv_reset_parameters",
        "bias": "historical_mixed",
        "classifier": "conv1_3_logits",
    },
    FIGURE: {
        "name": FIGURE,
        "schema_version": 1,
        "status": "diagnostic_unresolved",
        "input_shape": [1, 12, 224, 224],
        "block_counts": [4, 12, 24, 36],
        "bottlenecks": [128, 128, 128, 32],
        "growth": 32,
        "compression": 0.5,
        "stem": "conv7_stride2_same_upper_bn_relu",
        "pool": "max3_stride2_same_upper",
        "transition": "conv1_bn_relu_conv1_stride122",
        "dense_order": "conv1_bn_relu_depthwise3_pointwise1_bn_relu",
        "decoder": "conv3_bn_relu",
        "decoder_widths": [504, 224, 192, 96, 64],
        "skips": ["dense_block4", "dense_block3", "dense_block2", "dense_block1", "stem"],
        "resize": "both_trilinear_align_corners_false",
        "batch_norm": {"eps": 1e-3, "momentum": 0.01, "running_variance": "unbiased"},
        "initialization": "glorot_uniform_keras_depthwise_fans_bias_zero",
        "bias": "all_convolutions",
        "classifier": "conv1_3_logits",
    },
}


def canonical_model_config(name: str = HISTORICAL) -> dict[str, Any]:
    if name not in _CONFIGS:
        raise ValueError(f"Unsupported model name: {name!r}")
    return copy.deepcopy(_CONFIGS[name])


def _normalize(config: Mapping[str, Any] | str | None) -> dict[str, Any]:
    if config is None or isinstance(config, str):
        return canonical_model_config(HISTORICAL if config is None else config)
    if not isinstance(config, Mapping):
        raise ValueError("model_config must be a complete named configuration")
    canonical = canonical_model_config(config.get("name", ""))
    if dict(config) != canonical:
        raise ValueError(
            "Unsupported or mismatched model configuration; graph identity must match exactly"
        )
    return canonical


def model_fingerprint(config: Mapping[str, Any] | str | None = None) -> str:
    canonical = _normalize(config)
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_model(model_config: Mapping[str, Any] | str | None = None) -> nn.Module:
    config = _normalize(model_config)
    return DenseUNet3d() if config["name"] == HISTORICAL else FigureSkipReconstruction()


def model_metadata(model: nn.Module) -> dict[str, Any]:
    if type(model) is DenseUNet3d:
        config = canonical_model_config(HISTORICAL)
    elif type(model) is FigureSkipReconstruction:
        config = canonical_model_config(FIGURE)
    else:
        raise ValueError(f"Unsupported model type: {type(model).__name__}")
    return {"model_config": config, "model_fingerprint": model_fingerprint(config)}


def validate_model_metadata(
    metadata: Mapping[str, Any], expected_config: Mapping[str, Any] | str | None = None
) -> dict[str, Any]:
    """Validate a full checkpoint's graph identity; missing metadata is not guessed."""
    if "model_config" not in metadata or "model_fingerprint" not in metadata:
        raise ValueError(
            "Checkpoint lacks model identity; explicitly select the known historical legacy path"
        )
    config = _normalize(metadata["model_config"])
    if metadata["model_fingerprint"] != model_fingerprint(config):
        raise ValueError("Checkpoint model fingerprint mismatch")
    if expected_config is not None and config != _normalize(expected_config):
        raise ValueError("Checkpoint model configuration mismatch")
    return config


def model_manifest(model: nn.Module) -> dict[str, Any]:
    """Inspect a separate meta-device graph without changing the live model or BN."""
    metadata = model_metadata(model)
    with torch.device("meta"):
        probe = build_model(metadata["model_config"])
    stages = []
    handles = []
    for name, module in probe.named_children():

        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor, stage: str = name) -> None:
            stages.append(
                {
                    "name": stage,
                    "shape": list(output.shape),
                    "parameters": sum(p.numel() for p in _module.parameters()),
                }
            )

        handles.append(module.register_forward_hook(hook))
    try:
        probe(torch.empty(1, 1, 12, 224, 224, device="meta"))
    finally:
        for handle in handles:
            handle.remove()
    return {
        **metadata,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "stages": stages,
        "skip_sources": metadata["model_config"]["skips"],
    }
