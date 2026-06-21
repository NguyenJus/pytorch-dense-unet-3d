"""Tests for DepthwiseSeparableConv3d (B1)."""

import torch

from dense_unet_3d.model.building_blocks.ds_conv import DepthwiseSeparableConv3d


def test_output_shape_same_resolution() -> None:
    """3x3x3 depthwise with pad=1, stride=1 preserves D, H, W."""
    C_in, C_out = 16, 32
    D, H, W = 4, 8, 8
    model = DepthwiseSeparableConv3d(
        in_channels=C_in, out_channels=C_out, kernel_size=3, padding=1, stride=1
    )
    x = torch.randn(1, C_in, D, H, W)
    out = model(x)
    assert out.shape == (1, C_out, D, H, W), (
        f"Expected (1, {C_out}, {D}, {H}, {W}), got {out.shape}"
    )


def test_output_shape_with_stride() -> None:
    """stride=2 halves the spatial dims."""
    C_in, C_out = 8, 16
    D, H, W = 6, 14, 14
    model = DepthwiseSeparableConv3d(
        in_channels=C_in, out_channels=C_out, kernel_size=3, padding=1, stride=2
    )
    x = torch.randn(1, C_in, D, H, W)
    out = model(x)
    assert out.shape == (1, C_out, D // 2, H // 2, W // 2), f"Unexpected shape {out.shape}"


def test_param_count_exact_formula() -> None:
    """
    Param count must equal:
      depthwise: C_in * k^3 + C_in  (weight + bias)
      pointwise: C_in * C_out + C_out  (weight + bias)
    This proves genuine separability (far fewer than a dense k^3 conv).
    """
    C_in, C_out, k = 16, 32, 3
    model = DepthwiseSeparableConv3d(
        in_channels=C_in, out_channels=C_out, kernel_size=k, padding=1, stride=1
    )

    expected_dw = C_in * (k**3) + C_in  # depthwise weight + bias
    expected_pw = C_in * C_out + C_out  # pointwise weight + bias
    expected_total = expected_dw + expected_pw

    actual_total = sum(p.numel() for p in model.parameters())
    assert actual_total == expected_total, (
        f"Expected {expected_total} params "
        f"(dw={expected_dw} + pw={expected_pw}), got {actual_total}"
    )

    # Also verify it's much fewer than a dense conv of the same size
    dense_params = C_in * C_out * (k**3) + C_out  # standard conv weight + bias
    assert actual_total < dense_params, (
        "DS-Conv should have fewer params than a standard conv of the same in/out/kernel"
    )


def test_forward_backward_cpu() -> None:
    """Forward + backward on CPU; gradients are non-None."""
    C_in, C_out = 8, 16
    model = DepthwiseSeparableConv3d(
        in_channels=C_in, out_channels=C_out, kernel_size=3, padding=1, stride=1
    )
    x = torch.randn(1, C_in, 4, 6, 6, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Input gradient should be non-None after backward"
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None after backward"
