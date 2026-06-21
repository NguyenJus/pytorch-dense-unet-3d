"""Tests for UpsamplingBlock — C2 acceptance criteria."""

import pytest
import torch

from dense_unet_3d.model.building_blocks.UpsamplingBlock import UpsamplingBlock

TARGET_SIZE = (4, 8, 8)  # (D, H, W)
IN_CHANNELS = 32
SKIP_CHANNELS = 16
OUT_CHANNELS = 24


@pytest.fixture()
def block() -> UpsamplingBlock:
    return UpsamplingBlock(
        in_channels=IN_CHANNELS,
        skip_channels=SKIP_CHANNELS,
        out_channels=OUT_CHANNELS,
        target_size=TARGET_SIZE,
    )


def test_only_main_path_is_upsampled(block: UpsamplingBlock) -> None:
    """Skip tensor must pass through at its input resolution (target_size), unchanged by upsample."""
    # main path is at half the target spatial size — needs upsampling
    x = torch.randn(1, IN_CHANNELS, 2, 4, 4)
    # skip is already at target_size — must NOT be upsampled again
    skip = torch.randn(1, SKIP_CHANNELS, *TARGET_SIZE)

    # Verify block internally does NOT apply upsample to the skip:
    # we hook the upsample module and record which tensors it processes
    upsampled_inputs: list[tuple[int, ...]] = []

    def capture_hook(module, inputs, output):  # noqa: ANN001
        upsampled_inputs.append(tuple(inputs[0].shape))

    hook = block.upsample.register_forward_hook(capture_hook)
    try:
        block(x, skip)
    finally:
        hook.remove()

    # Upsample should have been called exactly once (for x, not for skip)
    assert len(upsampled_inputs) == 1, (
        f"Expected upsample called 1 time (main path only), got {len(upsampled_inputs)}"
    )
    assert upsampled_inputs[0] == (1, IN_CHANNELS, 2, 4, 4), (
        f"Upsample was called on wrong tensor shape: {upsampled_inputs[0]}"
    )


def test_output_spatial_equals_target_size(block: UpsamplingBlock) -> None:
    """Output spatial dims must equal target_size."""
    x = torch.randn(1, IN_CHANNELS, 2, 4, 4)
    skip = torch.randn(1, SKIP_CHANNELS, *TARGET_SIZE)
    out = block(x, skip)
    assert tuple(out.shape[2:]) == TARGET_SIZE, (
        f"Expected spatial {TARGET_SIZE}, got {tuple(out.shape[2:])}"
    )


def test_output_channels_equals_out_channels(block: UpsamplingBlock) -> None:
    """Output channel dim must equal out_channels."""
    x = torch.randn(1, IN_CHANNELS, 2, 4, 4)
    skip = torch.randn(1, SKIP_CHANNELS, *TARGET_SIZE)
    out = block(x, skip)
    assert out.shape[1] == OUT_CHANNELS, (
        f"Expected {OUT_CHANNELS} output channels, got {out.shape[1]}"
    )


def test_forward_and_backward_cpu(block: UpsamplingBlock) -> None:
    """Full forward + backward pass on CPU must succeed without errors."""
    x = torch.randn(1, IN_CHANNELS, 2, 4, 4, requires_grad=True)
    skip = torch.randn(1, SKIP_CHANNELS, *TARGET_SIZE, requires_grad=True)
    out = block(x, skip)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient for main input x"
    assert skip.grad is not None, "No gradient for skip input"
