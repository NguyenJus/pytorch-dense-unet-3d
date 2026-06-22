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

from dense_unet_3d.evaluation.dice_score import dice_global, dice_per_case
from dense_unet_3d.evaluation.evaluate import NUM_CLASSES, evaluate

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
    """Each Dice value must be a float; per-case values may be nan when a class
    is absent from every volume (presence-aware convention, issue #4).
    Global values use the empty-class convention and remain in [0, 1]."""
    import math

    model = _StubModel()
    device = torch.device("cpu")
    val_loader = _make_val_loader()

    result = evaluate(model, device, val_loader)

    # Global values: always in [0, 1] (empty-class convention → 1.0 when absent)
    for key in ("liver_global", "tumor_global"):
        val = result[key]
        assert isinstance(val, float), f"{key}: expected float, got {type(val)}"
        assert 0.0 <= val <= 1.0, f"{key}: value {val} out of [0, 1]"

    # Per-case values: float in [0, 1] OR nan (presence-aware: absent class → nan)
    for key in ("liver_per_case", "tumor_per_case"):
        val = result[key]
        assert isinstance(val, float), f"{key}: expected float, got {type(val)}"
        assert (0.0 <= val <= 1.0) or math.isnan(val), (
            f"{key}: value {val} is not in [0, 1] and not nan"
        )


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


# ---------------------------------------------------------------------------
# 8. Streaming refactor must produce metrics IDENTICAL to the batched reference
# ---------------------------------------------------------------------------


class _VariedModel(nn.Module):
    """
    Produces non-trivial, deterministic logits that depend on the input so that
    argmax varies across voxels — exercising real intersection/union counts for
    both liver (class 1) and tumor (class 2).
    """

    def __init__(self) -> None:
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, d, h, w = x.shape
        out = torch.zeros(n, 3, d, h, w)
        # Channel logits derived deterministically from the input itself.
        out[:, 0] = x[:, 0] * 0.5
        out[:, 1] = torch.sin(x[:, 0] * 3.0)
        out[:, 2] = torch.cos(x[:, 0] * 2.0) - 0.2
        return out


def _reference_metrics(model, device, val_loader) -> dict[str, float]:
    """
    The ORIGINAL batched implementation: collect every batch's full logits and
    targets, torch.cat the whole val set, then compute Dice once.  Used as the
    ground-truth reference the streaming evaluate() must match exactly.
    """
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for volume, target in val_loader:
            volume = volume.to(device, dtype=torch.float32)
            if target.dim() == 5:
                target = target.squeeze(1)
            target = target.to(device, dtype=torch.long)
            logits = model(volume)
            all_preds.append(logits.cpu())
            all_targets.append(target.cpu())
    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    pc = dice_per_case(preds_cat, targets_cat, num_classes=NUM_CLASSES)
    gl = dice_global(preds_cat, targets_cat, num_classes=NUM_CLASSES)
    return {
        "liver_per_case": float(pc[1].item()),
        "liver_global": float(gl[1].item()),
        "tumor_per_case": float(pc[2].item()),
        "tumor_global": float(gl[2].item()),
    }


def _make_varied_val_loader(
    n_batches: int = 4, batch_size: int = 2, d: int = 3, h: int = 5, w: int = 5
) -> DataLoader:
    """Synthetic multi-batch loader with all three classes present across volumes."""
    torch.manual_seed(7)
    n_total = n_batches * batch_size
    images = torch.randn(n_total, 1, d, h, w)
    targets = torch.randint(0, 3, (n_total, 1, d, h, w), dtype=torch.long)
    return DataLoader(TensorDataset(images, targets), batch_size=batch_size)


def _folded_liver_reference_metrics(model, device, val_loader) -> dict[str, float]:
    """
    Reference implementation computing folded-liver metrics for Issue #6.

    liver_* uses folded masks: gt >= 1 (liver ∪ tumor), argmax >= 1.
    tumor_* uses strict class-2 masks.
    Per-case is presence-aware (§2.4); nan when absent (§2.5).
    Global uses empty-class convention (denom==0 → 1.0).
    """
    from dense_unet_3d.evaluation.dice_score import _binary_dice

    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for volume, target in val_loader:
            volume = volume.to(device, dtype=torch.float32)
            if target.dim() == 5:
                target = target.squeeze(1)
            target = target.to(device, dtype=torch.long)
            logits = model(volume)
            all_logits.append(logits.cpu())
            all_targets.append(target.cpu())
    preds_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    n = preds_cat.shape[0]
    hard_pred = preds_cat.argmax(dim=1)  # (N, D, H, W)

    # Folded liver masks
    pred_liver = (hard_pred >= 1).float()
    gt_liver = (targets_cat >= 1).float()
    # Strict tumor masks
    pred_tumor = (hard_pred == 2).float()
    gt_tumor = (targets_cat == 2).float()

    # Per-case presence-aware
    liver_scores: list[float] = []
    tumor_scores: list[float] = []
    for i in range(n):
        pl, gl_ = pred_liver[i], gt_liver[i]
        if pl.any() or gl_.any():
            liver_scores.append(_binary_dice(pl, gl_))
        pt, gt_ = pred_tumor[i], gt_tumor[i]
        if pt.any() or gt_.any():
            tumor_scores.append(_binary_dice(pt, gt_))

    liver_pc = sum(liver_scores) / len(liver_scores) if liver_scores else float("nan")
    tumor_pc = sum(tumor_scores) / len(tumor_scores) if tumor_scores else float("nan")

    # Global
    li = (pred_liver * gt_liver).sum()
    ld = pred_liver.sum() + gt_liver.sum()
    liver_gl = float((2.0 * li / ld).item()) if ld > 0 else 1.0
    ti = (pred_tumor * gt_tumor).sum()
    td = pred_tumor.sum() + gt_tumor.sum()
    tumor_gl = float((2.0 * ti / td).item()) if td > 0 else 1.0

    return {
        "liver_per_case": float(liver_pc),
        "liver_global": float(liver_gl),
        "tumor_per_case": float(tumor_pc),
        "tumor_global": float(tumor_gl),
    }


def test_evaluate_streaming_matches_batched_reference() -> None:
    """
    The streaming evaluate() must return metric values identical (within fp
    tolerance) to the folded-liver reference implementation (Issue #6).

    liver_* uses folded masks (liver ∪ tumor); tumor_* is strict class 2.
    """
    import math

    import pytest

    device = torch.device("cpu")
    val_loader = _make_varied_val_loader()

    reference = _folded_liver_reference_metrics(_VariedModel(), device, val_loader)
    result = evaluate(_VariedModel(), device, val_loader)

    for key in ("liver_per_case", "liver_global", "tumor_per_case", "tumor_global"):
        ref_val = reference[key]
        res_val = result[key]
        if math.isnan(ref_val):
            assert math.isnan(res_val), f"{key}: expected nan, got {res_val}"
        else:
            assert res_val == pytest.approx(ref_val, abs=1e-6, rel=1e-6), (
                f"{key}: streaming={res_val} != reference={ref_val}"
            )


# ---------------------------------------------------------------------------
# Issue #6 — Folded liver tests
# ---------------------------------------------------------------------------


class _TumorModel(nn.Module):
    """
    Predicts tumor (class 2) for all voxels.

    Used to construct a scenario where some voxels are class-2 in both GT and
    prediction, so tumor voxels contribute to the folded-liver intersection
    but NOT to strict-class-1 liver intersection.
    """

    def __init__(self) -> None:
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, d, h, w = x.shape
        out = torch.zeros(n, 3, d, h, w)
        out[:, 2] = 1.0  # class 2 (tumor) always wins
        return out


def test_folded_liver_exceeds_strict_class1_when_tumor_present() -> None:
    """
    When tumor voxels exist, folded liver_per_case must exceed strict class-1
    Dice (strict inequality).

    Construct: target has only class-2 (tumor) voxels.  Model predicts class-2
    everywhere.  Folded liver (gt>=1, argmax>=1) → perfect overlap → Dice=1.0.
    Strict class-1 (gt==1, argmax==1) → no class-1 voxels at all → nan.
    So folded_liver_per_case (1.0) > strict_class1_liver_per_case (nan is not
    a number but we verify: folded is 1.0 and strict is nan, i.e., absent).

    We also use a mixed case (some class-1 + some class-2 voxels) so both
    strict and folded are non-nan, and folded > strict because class-2 voxels
    contribute to folded but not strict class-1 numerator.
    """
    import math

    # Mixed target: half the voxels are class 1 (liver), half are class 2 (tumor).
    # Model predicts class 2 everywhere.
    # Strict class-1 Dice: pred_liver == (argmax==1) = empty, gt_liver = 4 voxels
    #   → intersection 0, denom = 0+4 = 4 → Dice = 0.0
    # Folded liver Dice: pred_liver = (argmax>=1) = all 8 voxels, gt_liver = all 8
    #   → intersection 8, denom = 8+8 → Dice = 1.0
    # So 1.0 > 0.0 — strict inequality holds.
    images = torch.randn(1, 1, 2, 2, 2)
    targets = torch.zeros(1, 1, 2, 2, 2, dtype=torch.long)
    # Half the voxels = class 1 (liver), other half = class 2 (tumor)
    targets[0, 0, 0, :, :] = 1  # 4 liver voxels
    targets[0, 0, 1, :, :] = 2  # 4 tumor voxels
    val_loader = DataLoader(TensorDataset(images, targets), batch_size=1)

    model = _TumorModel()  # predicts class 2 everywhere
    device = torch.device("cpu")

    result = evaluate(model, device, val_loader)

    # Compute strict class-1 Dice reference via dice_per_case
    # (dice_per_case is NOT folded — strict per-class)
    logits = model(images)  # (1, 3, 2, 2, 2)
    targets_squeezed = targets.squeeze(1)  # (1, 2, 2, 2)
    strict_scores = dice_per_case(logits, targets_squeezed, num_classes=3)
    strict_class1_dice = float(strict_scores[1].item())

    folded_liver_dice = result["liver_per_case"]

    # Folded must be strictly greater than strict class-1
    assert not math.isnan(folded_liver_dice), (
        f"folded liver_per_case should not be nan, got {folded_liver_dice}"
    )
    if math.isnan(strict_class1_dice):
        # Strict class-1 is nan (no class-1 prediction), folded is not nan → strictly greater
        assert True, "folded > nan (strict class-1 absent)"
    else:
        assert folded_liver_dice > strict_class1_dice, (
            f"Expected folded {folded_liver_dice} > strict {strict_class1_dice}"
        )

    # More concretely: folded liver Dice should be 1.0 (model predicts class-2,
    # gt >= 1 covers all non-background, model argmax >= 1 covers all voxels)
    import pytest

    assert folded_liver_dice == pytest.approx(1.0), (
        f"Expected folded liver_per_case=1.0, got {folded_liver_dice}"
    )


def test_tumor_metrics_unchanged_by_liver_fold() -> None:
    """
    tumor_per_case and tumor_global must reflect strict class-2 Dice,
    unaffected by the liver-fold change.

    Construct: target with class-1 and class-2 voxels; model predicts class 2
    everywhere.  Compute tumor metrics via evaluate() and compare against
    strict class-2 Dice computed inline.
    """
    import math

    import pytest

    images = torch.randn(1, 1, 2, 2, 2)
    targets = torch.zeros(1, 1, 2, 2, 2, dtype=torch.long)
    targets[0, 0, 0, :, :] = 1  # 4 liver voxels
    targets[0, 0, 1, :, :] = 2  # 4 tumor voxels
    val_loader = DataLoader(TensorDataset(images, targets), batch_size=1)

    model = _TumorModel()  # predicts class 2 everywhere
    device = torch.device("cpu")

    result = evaluate(model, device, val_loader)

    # Compute strict tumor reference via dice_per_case (strict per-class)
    logits = model(images)
    targets_squeezed = targets.squeeze(1)
    strict_scores = dice_per_case(logits, targets_squeezed, num_classes=3)
    strict_tumor_per_case = float(strict_scores[2].item())

    strict_gl_scores = dice_global(logits, targets_squeezed, num_classes=3)
    strict_tumor_global = float(strict_gl_scores[2].item())

    tumor_per_case = result["tumor_per_case"]
    tumor_global = result["tumor_global"]

    # tumor_per_case should match strict class-2
    if math.isnan(strict_tumor_per_case):
        assert math.isnan(tumor_per_case), (
            f"Expected tumor_per_case=nan, got {tumor_per_case}"
        )
    else:
        assert tumor_per_case == pytest.approx(strict_tumor_per_case, abs=1e-6), (
            f"tumor_per_case {tumor_per_case} != strict {strict_tumor_per_case}"
        )

    # tumor_global should match strict class-2
    assert tumor_global == pytest.approx(strict_tumor_global, abs=1e-6), (
        f"tumor_global {tumor_global} != strict {strict_tumor_global}"
    )
