"""Tests for full DenseUNet3d assembly — C3, the three disambiguators.

These tests pin the architecture per spec §2/§3:
  1. Output shape: (2, 1, 12, 224, 224) -> EXACTLY (2, 3, 12, 224, 224).
  2. Param count: trainable params within 3.06M - 4.14M (3.6M +/- 15%); prints actual.
  3. Spatial dims: every intermediate matches the §2 table.
Plus forward + backward on CPU with no NaN and non-None grads.

All CPU-only.
"""

from __future__ import annotations

import pytest
import torch

from dense_unet_3d.model.DenseUNet3d import DenseUNet3d

# Lower bound / upper bound for ~3.6M +/- 15%.
# Band is satisfied via half-scale block counts (2,6,12,18) with g=32 preserved
# — an authorized deviation from the paper's (4,12,24,36); see decision record:
# docs/research/2026-06-21-denseunet569-architecture-decisions.md
PARAM_LOWER: int = 3_060_000
PARAM_UPPER: int = 4_140_000


@pytest.fixture()
def model() -> DenseUNet3d:
    torch.manual_seed(0)
    return DenseUNet3d()


def test_output_shape_exact(model: DenseUNet3d) -> None:
    """Input (2,1,12,224,224) must map to EXACTLY (2,3,12,224,224)."""
    x = torch.randn(2, 1, 12, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 3, 12, 224, 224), f"Got {tuple(out.shape)}"


def test_param_count_within_band(model: DenseUNet3d) -> None:
    """Trainable params within 3.06M - 4.14M. Prints the actual count."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[DenseUNet3d] trainable parameters = {total:,}")  # noqa: T201
    assert PARAM_LOWER <= total <= PARAM_UPPER, (
        f"Param count {total:,} outside band [{PARAM_LOWER:,}, {PARAM_UPPER:,}]"
    )


def test_intermediate_spatial_dims_match_paper(model: DenseUNet3d) -> None:
    """Every intermediate feature map must match the §2 spatial table.

    Captured via forward hooks on the named encoder/decoder stages. Spatial
    dims are (D, H, W) in NCDHW.
    """
    captured: dict[str, tuple[int, ...]] = {}

    def make_hook(name: str):  # noqa: ANN202
        def hook(module, inp, out):  # noqa: ANN001
            captured[name] = tuple(out.shape)

        return hook

    handles = []
    # Encoder stages.
    handles.append(model.stem.register_forward_hook(make_hook("stem")))
    handles.append(model.pool.register_forward_hook(make_hook("pool")))
    handles.append(model.dense_block1.register_forward_hook(make_hook("db1")))
    handles.append(model.transition1.register_forward_hook(make_hook("t1")))
    handles.append(model.dense_block2.register_forward_hook(make_hook("db2")))
    handles.append(model.transition2.register_forward_hook(make_hook("t2")))
    handles.append(model.dense_block3.register_forward_hook(make_hook("db3")))
    handles.append(model.transition3.register_forward_hook(make_hook("t3")))
    handles.append(model.dense_block4.register_forward_hook(make_hook("db4")))
    # Decoder stages.
    handles.append(model.up1.register_forward_hook(make_hook("up1")))
    handles.append(model.up2.register_forward_hook(make_hook("up2")))
    handles.append(model.up3.register_forward_hook(make_hook("up3")))
    handles.append(model.up4.register_forward_hook(make_hook("up4")))
    handles.append(model.up5.register_forward_hook(make_hook("up5")))

    x = torch.randn(1, 1, 12, 224, 224)
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()

    # Expected (D, H, W) per §2.
    expected_spatial = {
        "stem": (6, 112, 112),
        "pool": (3, 56, 56),
        "db1": (3, 56, 56),
        "t1": (3, 28, 28),
        "db2": (3, 28, 28),
        "t2": (3, 14, 14),
        "db3": (3, 14, 14),
        "t3": (3, 7, 7),
        "db4": (3, 7, 7),
        "up1": (3, 14, 14),
        "up2": (3, 28, 28),
        "up3": (3, 56, 56),
        "up4": (6, 112, 112),
        "up5": (12, 224, 224),
    }
    for name, exp in expected_spatial.items():
        assert name in captured, f"hook {name} not fired"
        got = captured[name][2:]
        assert got == exp, f"stage {name}: expected D,H,W {exp}, got {got}"

    # Expected output channels (block counts (2,6,12,18) at half-scale of the
    # paper's (4,12,24,36); see DenseUNet3d module docstring for rationale).
    expected_channels = {
        "stem": 96,
        "pool": 96,
        "db1": 96 + 2 * 32,  # 160
        "db2": 80 + 6 * 32,  # 272
        "db3": 136 + 12 * 32,  # 520
        "db4": 260 + 18 * 32,  # 836
        "up1": 504,
        "up2": 224,
        "up3": 192,
        "up4": 96,
        "up5": 64,
    }
    for name, exp_c in expected_channels.items():
        got_c = captured[name][1]
        assert got_c == exp_c, f"stage {name}: expected {exp_c} channels, got {got_c}"


def test_forward_backward_no_nan(model: DenseUNet3d) -> None:
    """Forward + backward on CPU: finite output, non-None grads, no NaN."""
    torch.manual_seed(1)
    x = torch.randn(2, 1, 12, 224, 224, requires_grad=True)
    out = model(x)
    assert torch.isfinite(out).all(), "Output contains non-finite values"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Input grad is None"
    assert not torch.isnan(x.grad).any(), "NaN in input grad"
    for n, p in model.named_parameters():
        assert p.grad is not None, f"grad None for {n}"
        assert not torch.isnan(p.grad).any(), f"NaN grad for {n}"
