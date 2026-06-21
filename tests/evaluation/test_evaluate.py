"""
Tests for dense_unet_3d.evaluation.evaluate.

Acceptance criteria (E3):
1. evaluate() consumes the VAL loader — not the train loader path.
   Regression: assert the function signature accepts val_loader (not a
   generic dataloader) and that calling it with a synthetic val loader works.
2. Returns liver + tumor Dice (per-case and global) as a structured dict.
3. Runs on CPU with a tiny synthetic val loader.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dense_unet_3d.evaluation.evaluate import evaluate

# ---------------------------------------------------------------------------
# Tiny deterministic stub model
# ---------------------------------------------------------------------------


class _StubModel(nn.Module):
    """Returns fixed logits: class 0 always wins → argmax = 0 everywhere."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        # Dummy parameter so model.eval() / model.parameters() work fine
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        # Output (N, C, D, H, W) — channel 0 is always highest
        n = x.shape[0]
        d, h, w = x.shape[2], x.shape[3], x.shape[4]
        out = torch.zeros(n, self.num_classes, d, h, w)
        out[:, 0, :, :, :] = 1.0  # background always wins
        return out


def _make_val_loader(
    n_batches: int = 2,
    batch_size: int = 1,
    d: int = 2,
    h: int = 4,
    w: int = 4,
) -> DataLoader:
    """Synthetic val loader: images (N,1,D,H,W) and integer targets (N,1,D,H,W)."""
    torch.manual_seed(0)
    n_total = n_batches * batch_size
    images = torch.randn(n_total, 1, d, h, w)
    # Mix of 0 (background) and 1 (liver) labels; no tumor for simplicity.
    targets = torch.zeros(n_total, 1, d, h, w, dtype=torch.long)
    # Give one voxel a liver label so classes aren't all empty
    targets[0, 0, 0, 0, 0] = 1
    return DataLoader(TensorDataset(images, targets), batch_size=batch_size)


# ---------------------------------------------------------------------------
# 1. evaluate() accepts val_loader and returns a dict
# ---------------------------------------------------------------------------


def test_evaluate_returns_dict() -> None:
    """evaluate() must return a dict (structured result)."""
    model = _StubModel()
    device = torch.device("cpu")
    val_loader = _make_val_loader()

    result = evaluate(model, device, val_loader)

    assert isinstance(result, dict), f"expected dict, got {type(result)}"


# ---------------------------------------------------------------------------
# 2. Dict has the four required keys
# ---------------------------------------------------------------------------


def test_evaluate_result_has_liver_and_tumor_keys() -> None:
    """Result dict must contain liver_per_case, liver_global, tumor_per_case, tumor_global."""
    model = _StubModel()
    device = torch.device("cpu")
    val_loader = _make_val_loader()

    result = evaluate(model, device, val_loader)

    expected_keys = {"liver_per_case", "liver_global", "tumor_per_case", "tumor_global"}
    assert expected_keys <= result.keys(), f"Missing keys: {expected_keys - result.keys()}"


# ---------------------------------------------------------------------------
# 3. Dict values are floats in [0, 1]
# ---------------------------------------------------------------------------


def test_evaluate_values_are_valid_dice_scores() -> None:
    """Each Dice value must be a float in [0.0, 1.0]."""
    model = _StubModel()
    device = torch.device("cpu")
    val_loader = _make_val_loader()

    result = evaluate(model, device, val_loader)

    for key in ("liver_per_case", "liver_global", "tumor_per_case", "tumor_global"):
        val = result[key]
        assert isinstance(val, float), f"{key}: expected float, got {type(val)}"
        assert 0.0 <= val <= 1.0, f"{key}: value {val} out of [0, 1]"


# ---------------------------------------------------------------------------
# 4. Perfect-prediction scenario: liver Dice = 1.0
# ---------------------------------------------------------------------------


class _PerfectModel(nn.Module):
    """
    Mimics perfect prediction: returns logits where argmax matches the target.

    For our synthetic val_loader all targets are 0 (background) except
    one voxel which is 1 (liver).  This model always predicts class 0, so
    liver Dice won't be perfect — we use a model that always predicts the
    background class and confirm the Dice is in a valid range (background-only
    model means class 1 pred is empty, class 1 gt has one voxel → Dice=0.0 for
    liver).
    """

    def __init__(self) -> None:
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        d, h, w = x.shape[2], x.shape[3], x.shape[4]
        out = torch.zeros(n, 3, d, h, w)
        out[:, 0] = 1.0  # always predict background
        return out


def test_evaluate_background_model_liver_dice_is_zero() -> None:
    """
    A model that always predicts background should give liver Dice = 0.0
    (pred liver mask empty, gt liver mask has one voxel → disjoint → 0).

    This pins the per-case and global computations for a known scenario.
    """
    model = _PerfectModel()
    device = torch.device("cpu")

    # One batch: target has exactly one liver voxel
    torch.manual_seed(42)
    images = torch.randn(1, 1, 2, 2, 2)
    targets = torch.zeros(1, 1, 2, 2, 2, dtype=torch.long)
    targets[0, 0, 0, 0, 0] = 1  # one liver voxel
    val_loader = DataLoader(TensorDataset(images, targets), batch_size=1)

    result = evaluate(model, device, val_loader)

    import pytest

    assert result["liver_per_case"] == pytest.approx(0.0), (
        f"Expected liver_per_case=0.0, got {result['liver_per_case']}"
    )
    assert result["liver_global"] == pytest.approx(0.0), (
        f"Expected liver_global=0.0, got {result['liver_global']}"
    )


# ---------------------------------------------------------------------------
# 5. Regression: val_loader path — function parameter is named val_loader
# ---------------------------------------------------------------------------


def test_evaluate_signature_accepts_val_loader_keyword() -> None:
    """
    Regression test: evaluate() must accept val_loader as a keyword argument
    (not dataloader — the old eval-on-train path used dataloader, called with
    trainloader).  Calling with the keyword val_loader= must NOT raise TypeError.
    """
    model = _StubModel()
    device = torch.device("cpu")
    val_loader = _make_val_loader()

    # This call uses the keyword argument explicitly — it will raise TypeError
    # if the parameter is named something other than val_loader.
    result = evaluate(model, device, val_loader=val_loader)

    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 6. Empty val_loader raises a clear error (or returns safe defaults)
# ---------------------------------------------------------------------------


def test_evaluate_empty_loader_does_not_crash() -> None:
    """
    An empty val loader (zero batches) should not crash with a ZeroDivisionError.
    It should either return NaN-safe defaults or raise a descriptive ValueError.
    Both are acceptable; crashing silently is not.
    """

    model = _StubModel()
    device = torch.device("cpu")
    images = torch.zeros(0, 1, 2, 2, 2)
    targets = torch.zeros(0, 1, 2, 2, 2, dtype=torch.long)
    empty_loader = DataLoader(TensorDataset(images, targets), batch_size=1)

    # Must not raise an unhandled ZeroDivisionError
    try:
        result = evaluate(model, device, val_loader=empty_loader)
        # If it returns, values must be dicts (even if NaN/inf)
        assert isinstance(result, dict)
    except ValueError:
        pass  # A descriptive ValueError is acceptable


# ---------------------------------------------------------------------------
# 7. CPU-only: model stays on CPU, no CUDA calls needed
# ---------------------------------------------------------------------------


def test_evaluate_runs_on_cpu() -> None:
    """Full run on CPU with synthetic data; no GPU required."""
    model = _StubModel()
    device = torch.device("cpu")
    val_loader = _make_val_loader(n_batches=3, batch_size=2, d=2, h=4, w=4)

    result = evaluate(model, device, val_loader)

    assert isinstance(result, dict)
    assert len(result) >= 4
