"""Shared pytest fixtures for the dense_unet_3d test suite.

All fixtures are CPU-only and deterministic (seeded). Later test modules import
these directly by name — pytest discovers them automatically from conftest.py.

Fixtures provided
-----------------
seeded
    Sets torch and numpy random seeds before each test; returns None.
synthetic_nifti
    A nibabel Nifti1Image with float32 data, shape (224, 224, 12) — NIfTI
    HWD convention, matching what LITSDataset reads from disk.
synthetic_nifti_seg
    A nibabel Nifti1Image with int16 data, shape (224, 224, 12), labels in
    {0, 1, 2}.
input_tensor
    A random float32 CPU tensor of shape (1, 1, 12, 224, 224) — NCDHW, the
    paper's input shape with batch size 1.
small_tensor
    A random float32 CPU tensor of shape (2, 1, 4, 8, 8) — NCDHW, tiny
    spatial dims for fast unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

_SEED: int = 42


@pytest.fixture()
def seeded() -> None:
    """Fix torch + numpy seeds to ``_SEED`` for the duration of the test."""
    torch.manual_seed(_SEED)
    np.random.seed(_SEED)


@pytest.fixture()
def synthetic_nifti() -> object:
    """Return a nibabel Nifti1Image with float32 data, shape (224, 224, 12).

    Shape follows NIfTI HWD convention (not NCDHW). Values are drawn from a
    HU-like range [-200, 250] to approximate liver CT.
    """
    import nibabel as nib

    rng = np.random.default_rng(_SEED)
    data = rng.uniform(-200.0, 250.0, size=(224, 224, 12)).astype(np.float32)
    affine = np.eye(4, dtype=np.float64)
    return nib.Nifti1Image(data, affine)


@pytest.fixture()
def synthetic_nifti_seg() -> object:
    """Return a nibabel Nifti1Image with int16 labels in {0, 1, 2}, shape (224, 224, 12)."""
    import nibabel as nib

    rng = np.random.default_rng(_SEED + 1)
    # Mostly background (0), sparse liver (1) and lesion (2) voxels.
    data = rng.choice(np.array([0, 1, 2], dtype=np.int16), size=(224, 224, 12), p=[0.8, 0.15, 0.05])
    data = data.astype(np.int16)
    affine = np.eye(4, dtype=np.float64)
    return nib.Nifti1Image(data, affine)


@pytest.fixture()
def input_tensor() -> torch.Tensor:
    """Return a random float32 CPU tensor with the paper's input shape (1, 1, 12, 224, 224)."""
    torch.manual_seed(_SEED)
    return torch.randn(1, 1, 12, 224, 224, dtype=torch.float32)


@pytest.fixture()
def small_tensor() -> torch.Tensor:
    """Return a random float32 CPU tensor (2, 1, 4, 8, 8) for fast unit tests."""
    torch.manual_seed(_SEED)
    return torch.randn(2, 1, 4, 8, 8, dtype=torch.float32)
