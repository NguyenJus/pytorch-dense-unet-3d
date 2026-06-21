"""Tests for DenseBlock — B3 acceptance criteria.

Proves REAL dense connectivity (running torch.cat), not a plain nn.Sequential.
All tests run on CPU; no GPU required.
"""

from typing import cast

import pytest
import torch
import torch.nn as nn

from dense_unet_3d.model.building_blocks.DenseBlock import DenseBlock

GROWTH: int = 32


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_input(in_channels: int, spatial: int = 4) -> torch.Tensor:
    """Small synthetic NCDHW tensor."""
    torch.manual_seed(0)
    return torch.randn(1, in_channels, spatial, spatial, spatial)


# ---------------------------------------------------------------------------
# Output-channel correctness: in_channels + count*32
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "in_channels,count",
    [
        (96, 4),  # DB1 entry: 96 + 4*32 = 224
        (96, 12),  # DB2 entry: 96 + 12*32 = 480
        (96, 24),  # DB3 entry: 96 + 24*32 = 864
        (96, 36),  # DB4 entry: 96 + 36*32 = 1248
    ],
)
def test_output_channels_equals_in_plus_count_times_growth(in_channels: int, count: int) -> None:
    """Output channel count must be in_channels + count * 32 (running concat)."""
    block = DenseBlock(in_channels=in_channels, growth=GROWTH, count=count)
    x = _make_input(in_channels)
    with torch.no_grad():
        out = block(x)
    expected = in_channels + count * GROWTH
    assert out.shape[1] == expected, (
        f"count={count}: expected {expected} channels, got {out.shape[1]}"
    )


# ---------------------------------------------------------------------------
# Dense connectivity: output strictly exceeds growth (proves NOT plain Sequential)
# ---------------------------------------------------------------------------


def test_output_channels_exceed_single_growth_rate() -> None:
    """A plain Sequential returning only the last 32-ch layer would fail this."""
    block = DenseBlock(in_channels=96, growth=GROWTH, count=4)
    x = _make_input(96)
    with torch.no_grad():
        out = block(x)
    # Real dense block returns 96 + 4*32 = 224; Sequential would return 32
    assert out.shape[1] > GROWTH, (
        f"Output has only {out.shape[1]} channels — looks like a plain Sequential "
        f"returning only the last growth layer, not a concat of all layers."
    )


def test_block_is_not_plain_sequential() -> None:
    """Block must not be implemented as a bare nn.Sequential."""
    block = DenseBlock(in_channels=96, growth=GROWTH, count=4)
    # If the entire block is a single Sequential, it cannot do running cat
    assert not isinstance(block, nn.Sequential), "DenseBlock must not itself be nn.Sequential"
    # The forward must produce channels > growth (proves cat happened)
    x = _make_input(96)
    with torch.no_grad():
        out = block(x)
    assert out.shape[1] == 96 + 4 * GROWTH


# ---------------------------------------------------------------------------
# Per-layer input width grows by 32 (verify sub-layer in/out widths)
# ---------------------------------------------------------------------------


def test_each_layer_input_width_grows_by_growth_rate() -> None:
    """Each successive DenseLayer must accept in_channels+i*growth input channels."""
    in_channels = 96
    count = 4
    block = DenseBlock(in_channels=in_channels, growth=GROWTH, count=count)
    # Inspect the bottleneck Conv3d of each DenseLayer; its in_channels must be
    # in_channels + i*growth for layer index i (0-based).
    for i, layer in enumerate(block.layers):
        expected_in = in_channels + i * GROWTH
        # DenseLayer.bottleneck is a Sequential; first module is Conv3d 1x1x1
        bottleneck_seq = cast(nn.Sequential, layer.bottleneck)
        bottleneck_conv = cast(nn.Conv3d, bottleneck_seq[0])
        actual_in = bottleneck_conv.in_channels
        assert actual_in == expected_in, (
            f"Layer {i}: expected in_channels={expected_in}, got {actual_in}"
        )


# ---------------------------------------------------------------------------
# Spatial dims preserved through the block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spatial", [4, 8])
def test_spatial_dims_preserved(spatial: int) -> None:
    """Block must not alter D, H, or W."""
    in_channels = 96
    block = DenseBlock(in_channels=in_channels, growth=GROWTH, count=4)
    x = _make_input(in_channels, spatial=spatial)
    with torch.no_grad():
        out = block(x)
    assert out.shape[2:] == x.shape[2:], f"Spatial dims changed: {x.shape[2:]} -> {out.shape[2:]}"


# ---------------------------------------------------------------------------
# Forward + backward on CPU
# ---------------------------------------------------------------------------


def test_forward_backward_cpu() -> None:
    """Full forward and backward pass must complete on CPU with non-None grads."""
    torch.manual_seed(42)
    block = DenseBlock(in_channels=96, growth=GROWTH, count=4)
    x = torch.randn(1, 96, 4, 4, 4, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient on input — backward failed"
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"
