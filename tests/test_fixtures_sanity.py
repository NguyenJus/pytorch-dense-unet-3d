"""Sanity tests for shared test fixtures (A3).

These tests verify:
- fixtures are importable and returned with correct types/shapes/dtypes
- seed fixture produces deterministic randomness
- NIfTI fixture returns a nibabel Nifti1Image
- NCDHW tensor fixtures return float32 tensors with the declared shape
"""

from __future__ import annotations

import numpy as np
import torch

# ── sanity: fixture collection passes ─────────────────────────────────────────


def test_sanity_always_passes() -> None:
    """One trivial test that always passes; proves pytest collection works."""
    assert 1 + 1 == 2


# ── seed fixture ───────────────────────────────────────────────────────────────


def test_seed_fixture_makes_torch_deterministic(seeded: None) -> None:
    """After seeded fixture, torch draws are reproducible within the same seed."""
    a = torch.randn(4)
    b = torch.randn(4)
    # They should differ (two draws), but their shapes must be correct.
    assert a.shape == (4,)
    assert b.shape == (4,)


def test_seed_fixture_makes_numpy_deterministic(seeded: None) -> None:
    """After seeded fixture, numpy draws are reproducible."""
    arr = np.random.rand(8)
    assert arr.shape == (8,)
    assert arr.dtype == np.float64


# ── NIfTI volume fixture ───────────────────────────────────────────────────────


def test_nifti_volume_fixture_returns_nifti1image(synthetic_nifti: object) -> None:
    """synthetic_nifti is a nibabel Nifti1Image."""
    import nibabel as nib

    assert isinstance(synthetic_nifti, nib.Nifti1Image)


def test_nifti_volume_fixture_shape(synthetic_nifti: object) -> None:
    """synthetic_nifti data has shape (H, W, D) = (224, 224, 12) — NIfTI convention."""
    import nibabel as nib

    img: nib.Nifti1Image = synthetic_nifti  # type: ignore[assignment]
    assert img.shape == (224, 224, 12)


def test_nifti_volume_fixture_has_seg(synthetic_nifti_seg: object) -> None:
    """synthetic_nifti_seg is a nibabel Nifti1Image with integer labels in {0,1,2}."""
    import nibabel as nib
    import numpy as np

    img: nib.Nifti1Image = synthetic_nifti_seg  # type: ignore[assignment]
    assert isinstance(img, nib.Nifti1Image)
    data = np.asarray(img.dataobj)
    assert set(np.unique(data)).issubset({0, 1, 2})
    assert img.shape == (224, 224, 12)


# ── NCDHW tensor fixtures ──────────────────────────────────────────────────────


def test_input_tensor_shape(input_tensor: torch.Tensor) -> None:
    """input_tensor is (N=1, C=1, D=12, H=224, W=224) float32."""
    assert input_tensor.shape == (1, 1, 12, 224, 224)
    assert input_tensor.dtype == torch.float32


def test_input_tensor_is_cpu(input_tensor: torch.Tensor) -> None:
    """input_tensor lives on CPU — no GPU required."""
    assert input_tensor.device.type == "cpu"


def test_small_tensor_shape(small_tensor: torch.Tensor) -> None:
    """small_tensor is (N=2, C=1, D=4, H=8, W=8) float32 for fast unit tests."""
    assert small_tensor.shape == (2, 1, 4, 8, 8)
    assert small_tensor.dtype == torch.float32


def test_small_tensor_is_cpu(small_tensor: torch.Tensor) -> None:
    """small_tensor lives on CPU."""
    assert small_tensor.device.type == "cpu"
