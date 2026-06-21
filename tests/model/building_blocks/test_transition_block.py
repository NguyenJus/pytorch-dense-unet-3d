"""Tests for TransitionBlock (C1).

Spec §6 C1: BN -> 1x1x1 conv to round(in_channels * compression) ->
1x1x1 conv stride 2 (strided-conv downsample, not pooling).

Acceptance criteria:
- Halves in-plane H,W; depth stays at 3 through transitions.
- Output channels == round(in_channels * compression).
- Forward + backward on CPU.
"""

import pytest
import torch

from dense_unet_3d.model.building_blocks.TransitionBlock import TransitionBlock


@pytest.mark.parametrize(
    "in_channels,compression,expected_out",
    [
        # After DenseBlock1: in_channels = 96 + 4*32 = 224; compression=0.5 -> round(112)
        (224, 0.5, 112),
        # After DenseBlock2: 112 + 12*32 = 496; round(496*0.5) = 248
        (496, 0.5, 248),
        # After DenseBlock3: 248 + 24*32 = 1016; round(1016*0.5) = 508
        (1016, 0.5, 508),
        # Odd channels: Python round() uses banker's rounding; round(50.5)=50
        (101, 0.5, 50),
    ],
)
def test_output_channels_equal_compression_target(
    in_channels: int, compression: float, expected_out: int
) -> None:
    """Output channels must equal round(in_channels * compression)."""
    torch.manual_seed(0)
    model = TransitionBlock(in_channels, compression=compression)
    # depth=3, H=28, W=28 — representative in-plane size
    x = torch.randn(1, in_channels, 3, 28, 28)
    out = model(x)
    assert out.shape[1] == expected_out, (
        f"Expected {expected_out} output channels, got {out.shape[1]}"
    )


def test_halves_in_plane_spatial_dims_depth_unchanged() -> None:
    """H and W must be halved; depth must remain unchanged."""
    torch.manual_seed(0)
    in_channels = 224
    model = TransitionBlock(in_channels, compression=0.5)
    # depth=3, H=56, W=56 (post-DB1 resolution per §2)
    x = torch.randn(1, in_channels, 3, 56, 56)
    out = model(x)
    N, C, D, H, W = out.shape
    assert D == 3, f"Depth must stay 3 through transitions, got {D}"
    assert H == 28, f"H must halve 56->28, got {H}"
    assert W == 28, f"W must halve 56->28, got {W}"


@pytest.mark.parametrize(
    "in_h,in_w,expected_h,expected_w",
    [
        (56, 56, 28, 28),  # Transition 1: 56->28
        (28, 28, 14, 14),  # Transition 2: 28->14
        (14, 14, 7, 7),  # Transition 3: 14->7
    ],
)
def test_spatial_progression_matches_paper(
    in_h: int, in_w: int, expected_h: int, expected_w: int
) -> None:
    """In-plane dims follow the paper's §2 progression through each transition."""
    torch.manual_seed(0)
    in_channels = 224
    model = TransitionBlock(in_channels, compression=0.5)
    x = torch.randn(1, in_channels, 3, in_h, in_w)
    out = model(x)
    assert out.shape[3] == expected_h, f"Expected H={expected_h}, got {out.shape[3]}"
    assert out.shape[4] == expected_w, f"Expected W={expected_w}, got {out.shape[4]}"


def test_forward_backward_on_cpu() -> None:
    """Forward and backward passes must succeed on CPU; gradients must be non-None."""
    torch.manual_seed(42)
    in_channels = 224
    model = TransitionBlock(in_channels, compression=0.5)
    x = torch.randn(2, in_channels, 3, 56, 56, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Input gradients must be non-None after backward"
    assert not torch.isnan(x.grad).any(), "Input gradients must not contain NaN"


def test_uses_strided_conv_not_pooling() -> None:
    """Downsampling must be via strided 1x1x1 conv, not pooling (module structure check)."""
    model = TransitionBlock(64, compression=0.5)
    # Walk all submodules; must have a Conv3d with stride (1,2,2) and NO MaxPool3d/AvgPool3d
    has_strided_conv = False
    for module in model.modules():
        if isinstance(module, torch.nn.MaxPool3d) or isinstance(module, torch.nn.AvgPool3d):
            raise AssertionError("TransitionBlock must not use pooling for downsampling")
        if isinstance(module, torch.nn.Conv3d):
            strides = module.stride
            if strides == (1, 2, 2):
                has_strided_conv = True
    assert has_strided_conv, "TransitionBlock must contain a Conv3d with stride (1,2,2)"


def test_bn_comes_before_compression_conv() -> None:
    """Layer order must be BN -> conv (compression) -> conv (stride 2)."""
    model = TransitionBlock(64, compression=0.5)
    # Collect top-level sequential children in order
    children = list(model.convs.children())
    # First child must be BatchNorm3d
    assert isinstance(children[0], torch.nn.BatchNorm3d), (
        f"First layer must be BatchNorm3d, got {type(children[0])}"
    )
    # Second child must be Conv3d (compression conv)
    assert isinstance(children[1], torch.nn.Conv3d), (
        f"Second layer must be Conv3d (compression), got {type(children[1])}"
    )
    # Third child must be Conv3d (strided downsample)
    assert isinstance(children[2], torch.nn.Conv3d), (
        f"Third layer must be Conv3d (strided), got {type(children[2])}"
    )
    # Compression conv stride should be (1,1,1)
    assert children[1].stride == (1, 1, 1), (
        f"Compression conv must have stride 1, got {children[1].stride}"
    )
    # Strided conv stride should be (1,2,2)
    assert children[2].stride == (1, 2, 2), (
        f"Strided conv must have stride (1,2,2), got {children[2].stride}"
    )
