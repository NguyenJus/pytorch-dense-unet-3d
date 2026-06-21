"""Tests for DenseLayer (B2).

Acceptance criteria:
- input (N, C_in, D, H, W) -> output (N, 32, D, H, W)
- bottleneck width is 128; growth output is 32 (asserted via module introspection)
- Conv->BN->ReLU ordering verified by module structure
- forward + backward on CPU
"""

import pytest
import torch
import torch.nn as nn

from dense_unet_3d.model.building_blocks.dense_layer import DenseLayer


@pytest.mark.parametrize("c_in", [64, 96, 128, 224])
def test_output_shape_preserves_spatial_and_returns_growth_channels(c_in: int) -> None:
    """Output must be (N, 32, D, H, W) for any C_in."""
    torch.manual_seed(0)
    layer = DenseLayer(in_channels=c_in)
    x = torch.randn(1, c_in, 4, 8, 8)
    out = layer(x)
    assert out.shape == (1, 32, 4, 8, 8), f"Expected (1, 32, 4, 8, 8), got {tuple(out.shape)}"


def test_bottleneck_width_is_128() -> None:
    """The 1x1x1 bottleneck conv must map to exactly 128 channels."""
    layer = DenseLayer(in_channels=64)
    # bottleneck conv is the first Conv3d in the module
    conv_layers = [m for m in layer.modules() if isinstance(m, nn.Conv3d)]
    # first conv3d must output 128
    bottleneck_conv = conv_layers[0]
    assert bottleneck_conv.out_channels == 128, (
        f"Expected bottleneck out_channels=128, got {bottleneck_conv.out_channels}"
    )


def test_growth_output_is_32() -> None:
    """The final pointwise conv in DS-Conv must output exactly 32 channels."""
    layer = DenseLayer(in_channels=64)
    conv_layers = [m for m in layer.modules() if isinstance(m, nn.Conv3d)]
    # last conv3d is the pointwise that produces 32-channel growth output
    final_conv = conv_layers[-1]
    assert final_conv.out_channels == 32, (
        f"Expected final conv out_channels=32 (growth), got {final_conv.out_channels}"
    )


def test_conv_bn_relu_ordering_bottleneck_stage() -> None:
    """Bottleneck stage must be Conv3d -> BN -> ReLU (module order)."""
    layer = DenseLayer(in_channels=64)
    # bottleneck is a Sequential; inspect its children
    bottleneck = layer.bottleneck
    children = list(bottleneck.children())
    assert isinstance(children[0], nn.Conv3d), "bottleneck[0] must be Conv3d"
    assert isinstance(children[1], nn.BatchNorm3d), "bottleneck[1] must be BatchNorm3d"
    assert isinstance(children[2], nn.ReLU), "bottleneck[2] must be ReLU"


def test_conv_bn_relu_ordering_ds_stage() -> None:
    """DS-Conv growth stage must have Conv->BN->ReLU after the DS-Conv."""
    layer = DenseLayer(in_channels=64)
    # growth_stage is a Sequential containing ds_conv then BN then ReLU
    growth = layer.growth_stage
    children = list(growth.children())
    # children[0] is DepthwiseSeparableConv3d, children[1] BN, children[2] ReLU
    from dense_unet_3d.model.building_blocks.ds_conv import DepthwiseSeparableConv3d

    assert isinstance(children[0], DepthwiseSeparableConv3d), (
        "growth_stage[0] must be DepthwiseSeparableConv3d"
    )
    assert isinstance(children[1], nn.BatchNorm3d), "growth_stage[1] must be BatchNorm3d"
    assert isinstance(children[2], nn.ReLU), "growth_stage[2] must be ReLU"


def test_forward_backward_cpu() -> None:
    """Forward and backward must complete on CPU without error."""
    torch.manual_seed(42)
    layer = DenseLayer(in_channels=96)
    x = torch.randn(2, 96, 4, 8, 8, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient flowed to input"
    assert x.grad.shape == x.shape


def test_returns_only_new_features_not_concat() -> None:
    """DenseLayer must return only 32 new channels, not input+output concat."""
    torch.manual_seed(7)
    c_in = 160  # large input; if layer returned concat it would be 160+32=192
    layer = DenseLayer(in_channels=c_in)
    x = torch.randn(1, c_in, 4, 8, 8)
    out = layer(x)
    assert out.shape[1] == 32, f"DenseLayer must return 32 channels only; got {out.shape[1]}"
