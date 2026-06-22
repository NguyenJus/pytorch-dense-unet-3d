"""
Tests for dense_unet_3d.evaluation.dice_score.

Conventions documented in dice_score.py:

Per-case (presence-aware) convention:
    A volume counts toward class c's mean iff the class is present in GT OR
    prediction for that volume (gt_mask.any() OR pred_mask.any()).
    Both-empty volumes are EXCLUDED from the mean — not scored 1.0.
    If a class is absent everywhere in the batch → score is float('nan').

Global convention (unchanged):
    Dice = 1.0 when both prediction and ground-truth are empty for a class
    (empty-class convention, applied globally across all voxels).

Two public functions are tested:
    dice_per_case(preds, targets, num_classes) -> Tensor[C]
        Presence-aware mean per-volume Dice; absent-everywhere → nan.
    dice_global(preds, targets, num_classes) -> Tensor[C]
        Single Dice computed from intersection/union summed across ALL voxels
        in all cases.
"""

import pytest
import torch

from dense_unet_3d.evaluation.dice_score import dice_global, dice_per_case

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_perfect(
    n: int = 2, c: int = 3, d: int = 4, h: int = 4, w: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identical prediction and ground-truth filled with class 0 everywhere."""
    target = torch.zeros(n, d, h, w, dtype=torch.long)
    pred = torch.zeros(n, c, d, h, w, dtype=torch.float32)
    pred[:, 0, :, :, :] = 1.0  # all background
    return pred, target


def _make_disjoint(c: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single volume: prediction says class 1 everywhere; target says class 2
    everywhere. Classes 1 and 2 are therefore disjoint → Dice=0.
    Class 0: both empty → excluded → nan (presence-aware per-case convention).
    """
    n, d, h, w = 1, 2, 2, 2
    target = torch.full((n, d, h, w), fill_value=2, dtype=torch.long)
    pred = torch.zeros(n, c, d, h, w, dtype=torch.float32)
    pred[:, 1, :, :, :] = 1.0  # predict class 1 everywhere
    return pred, target


# ---------------------------------------------------------------------------
# 1. Perfect prediction → 1.0 per class
# ---------------------------------------------------------------------------


def test_dice_per_case_perfect_prediction_is_one() -> None:
    pred, target = _make_perfect()
    scores = dice_per_case(pred, target, num_classes=3)
    # class 0: present in both GT and pred → Dice = 1.0
    assert scores.shape == (3,)
    assert scores[0].item() == pytest.approx(1.0), (
        f"class 0: expected 1.0, got {scores[0].item()}"
    )
    # classes 1 and 2: absent in both GT and pred (no volumes present) → nan
    assert torch.isnan(scores[1]), f"class 1 (absent everywhere): expected nan, got {scores[1].item()}"
    assert torch.isnan(scores[2]), f"class 2 (absent everywhere): expected nan, got {scores[2].item()}"


def test_dice_global_perfect_prediction_is_one() -> None:
    pred, target = _make_perfect()
    scores = dice_global(pred, target, num_classes=3)
    assert scores.shape == (3,)
    for c in range(3):
        assert scores[c].item() == pytest.approx(1.0), (
            f"class {c}: expected 1.0, got {scores[c].item()}"
        )


# ---------------------------------------------------------------------------
# 2. Disjoint → 0.0 for non-empty classes; empty-both → nan (excluded)
# ---------------------------------------------------------------------------


def test_dice_per_case_disjoint_is_zero_for_nonempty_classes() -> None:
    pred, target = _make_disjoint()
    scores = dice_per_case(pred, target, num_classes=3)
    # class 0: absent in both GT and pred → excluded → nan (presence-aware)
    assert torch.isnan(scores[0]), (
        f"class 0 (both empty): expected nan, got {scores[0].item()}"
    )
    # class 1: pred has it, target doesn't → counts, Dice = 0.0
    assert scores[1].item() == pytest.approx(0.0), f"class 1: expected 0.0, got {scores[1].item()}"
    # class 2: target has it, pred doesn't → counts, Dice = 0.0
    assert scores[2].item() == pytest.approx(0.0), f"class 2: expected 0.0, got {scores[2].item()}"


def test_dice_global_disjoint_is_zero_for_nonempty_classes() -> None:
    pred, target = _make_disjoint()
    scores = dice_global(pred, target, num_classes=3)
    assert scores[0].item() == pytest.approx(1.0)
    assert scores[1].item() == pytest.approx(0.0)
    assert scores[2].item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Per-case presence-aware convention: absent class → nan
# ---------------------------------------------------------------------------


def test_dice_per_case_absent_class_is_nan() -> None:
    """A class absent in BOTH pred and target across a single volume is excluded → nan."""
    n, c, d, h, w = 1, 3, 2, 2, 2
    # Only class 0 is present in both; classes 1 and 2 are absent everywhere.
    target = torch.zeros(n, d, h, w, dtype=torch.long)
    pred = torch.zeros(n, c, d, h, w, dtype=torch.float32)
    pred[:, 0, :, :, :] = 1.0
    scores = dice_per_case(pred, target, num_classes=3)
    # classes 1 and 2: absent in both GT and pred → no volumes included → nan
    assert torch.isnan(scores[1]), "both-empty class 1 should be nan (excluded)"
    assert torch.isnan(scores[2]), "both-empty class 2 should be nan (excluded)"


def test_dice_global_empty_both_convention() -> None:
    n, c, d, h, w = 1, 3, 2, 2, 2
    target = torch.zeros(n, d, h, w, dtype=torch.long)
    pred = torch.zeros(n, c, d, h, w, dtype=torch.float32)
    pred[:, 0, :, :, :] = 1.0
    scores = dice_global(pred, target, num_classes=3)
    assert scores[1].item() == pytest.approx(1.0)
    assert scores[2].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Known partial overlap → exact expected value
# ---------------------------------------------------------------------------


def test_dice_per_case_known_partial_overlap() -> None:
    """
    Single volume, class 1 only (background and class 2 are both-empty → 1.0).

    Volume: 1×1×4 (D=1, H=1, W=4).
    Target class-1 mask: [1, 1, 0, 0]  → |B| = 2
    Pred   class-1 mask: [1, 0, 0, 0]  → |A| = 1   (only voxel 0 predicted as 1)
    Pred   class-0 mask: [0, 1, 1, 1]  → rest is background
    Intersection: [1, 0, 0, 0]         → |A∩B| = 1

    Dice = 2*1 / (1+2) = 2/3

    Use unambiguous one-hot logits so argmax is deterministic.
    """
    n, c, d, h, w = 1, 3, 1, 1, 4
    target = torch.tensor([[[[1, 1, 0, 0]]]], dtype=torch.long)  # (1,1,1,4)
    # Build one-hot logits: each voxel gets a large positive score for exactly
    # one class; all others stay 0.
    pred = torch.zeros(n, c, d, h, w, dtype=torch.float32)
    # voxel 0 → class 1
    pred[0, 1, 0, 0, 0] = 1.0
    # voxels 1,2,3 → class 0
    pred[0, 0, 0, 0, 1] = 1.0
    pred[0, 0, 0, 0, 2] = 1.0
    pred[0, 0, 0, 0, 3] = 1.0

    # pred class-1 mask: [1,0,0,0], target class-1 mask: [1,1,0,0]
    # Dice = 2*1/(1+2) = 2/3
    scores = dice_per_case(pred, target, num_classes=3)
    assert scores[1].item() == pytest.approx(2.0 / 3.0), (
        f"expected {2 / 3:.6f}, got {scores[1].item()}"
    )


def test_dice_global_known_partial_overlap() -> None:
    """Same geometry as per-case test; with a single case global == per-case."""
    n, c, d, h, w = 1, 3, 1, 1, 4
    target = torch.tensor([[[[1, 1, 0, 0]]]], dtype=torch.long)
    pred = torch.zeros(n, c, d, h, w, dtype=torch.float32)
    pred[0, 1, 0, 0, 0] = 1.0
    pred[0, 0, 0, 0, 1] = 1.0
    pred[0, 0, 0, 0, 2] = 1.0
    pred[0, 0, 0, 0, 3] = 1.0

    scores = dice_global(pred, target, num_classes=3)
    assert scores[1].item() == pytest.approx(2.0 / 3.0), (
        f"expected {2 / 3:.6f}, got {scores[1].item()}"
    )


# ---------------------------------------------------------------------------
# 5. Per-case mean PROVABLY diverges from global on a multi-case example
# ---------------------------------------------------------------------------


def test_per_case_and_global_diverge_on_multi_case_example() -> None:
    """
    Construct two volumes where per-case mean ≠ global Dice for class 1.

    Volume A (small, perfect overlap):
        target: [1]  (1 voxel)
        pred:   [1]
        Dice_A = 2*1 / (1+1) = 1.0

    Volume B (large, disjoint):
        target: [0,0,0,0,0,0,0,0]  (8 voxels, all background)
        pred class-1: [1,1,1,1,1,1,1,1]  (all wrong)
        Dice_B = 2*0 / (8+0) = 0.0

    Per-case mean = (1.0 + 0.0) / 2 = 0.5

    Global:
        total_intersection = 1 + 0 = 1
        total_union = (1+1) + (8+0) = 10
        Dice_global = 2*1 / 10 = 0.2

    0.5 ≠ 0.2  → provable divergence.

    Implementation note: both functions must accept a list/batch of identically-
    shaped volumes. We pad Volume A to the same size as B with extra background.
    """
    # Use shape (N=2, C=3, D=1, H=1, W=8); Vol A uses only the first voxel.
    c, d, h, w = 3, 1, 1, 8

    # Volume A: target class 1 in voxel 0 only; rest background
    target_a = torch.zeros(1, d, h, w, dtype=torch.long)
    target_a[0, 0, 0, 0] = 1
    pred_a = torch.zeros(1, c, d, h, w, dtype=torch.float32)
    pred_a[0, 1, 0, 0, 0] = 1.0  # correct: predict class 1 at voxel 0
    pred_a[0, 0, 0, 0, 1:] = 1.0  # rest background

    # Volume B: target all background; pred all class 1
    target_b = torch.zeros(1, d, h, w, dtype=torch.long)
    pred_b = torch.zeros(1, c, d, h, w, dtype=torch.float32)
    pred_b[0, 1, :, :, :] = 1.0  # predict class 1 everywhere (all wrong)

    # Stack into a batch of 2
    preds = torch.cat([pred_a, pred_b], dim=0)  # (2, 3, 1, 1, 8)
    targets = torch.cat([target_a, target_b], dim=0)  # (2, 1, 1, 8)

    per_case = dice_per_case(preds, targets, num_classes=3)
    glob = dice_global(preds, targets, num_classes=3)

    # class 1 Dice
    pc1 = per_case[1].item()
    gl1 = glob[1].item()

    # Per-case: vol A dice=1.0 (1 correct voxel), vol B dice=0.0 → mean=0.5
    assert pc1 == pytest.approx(0.5), f"per-case class 1 expected 0.5, got {pc1}"
    # Global: intersection=1, union=(1+1)+(8+0)=10 → 2*1/10=0.2
    assert gl1 == pytest.approx(0.2), f"global class 1 expected 0.2, got {gl1}"
    # They must differ
    assert pc1 != pytest.approx(gl1), "per-case and global must diverge on this example"


# ---------------------------------------------------------------------------
# 6. Inflation reproduction: absent volumes must NOT be included in mean
# ---------------------------------------------------------------------------


def test_per_case_not_inflated_by_absent_class_volumes() -> None:
    """
    Presence-aware per-case: absent (both-empty) volumes must be excluded.

    Batch of 5 volumes. Tumor (class 2) is present in only 1 volume (vol 0),
    and completely missed by the predictor → Dice = 0.0 for that volume.
    Volumes 1–4: class 2 absent in both GT and pred → excluded.

    Expected tumor_per_case = 0.0 (just the one present volume, missed).
    Old (inflated) average = (0.0 + 4 * 1.0) / 5 = 0.8 → must be avoided.
    """
    n, c, d, h, w = 5, 3, 2, 2, 2

    # Target: vol 0 has all class-2 voxels; vols 1-4 are all background
    target = torch.zeros(n, d, h, w, dtype=torch.long)
    target[0] = 2  # vol 0: all tumor

    # Pred: predict class 0 everywhere (miss tumor completely)
    pred = torch.zeros(n, c, d, h, w, dtype=torch.float32)
    pred[:, 0, :, :, :] = 1.0

    scores = dice_per_case(pred, target, num_classes=3)
    tumor_per_case = scores[2].item()

    # Present in only 1 volume, missed → Dice 0.0
    assert tumor_per_case == pytest.approx(0.0), (
        f"tumor_per_case expected 0.0 (single present volume, missed); got {tumor_per_case}"
    )

    # Old inflated value would have been (0.0 + 4*1.0)/5 = 0.8
    old_inflated = (0.0 + 4 * 1.0) / 5
    assert tumor_per_case < old_inflated, (
        f"tumor_per_case ({tumor_per_case}) should be strictly less than old inflated value "
        f"({old_inflated})"
    )


# ---------------------------------------------------------------------------
# 7. Absent-everywhere class → nan pin
# ---------------------------------------------------------------------------


def test_per_case_nan_when_class_absent_everywhere() -> None:
    """Class 2 never appears in GT or pred in the whole batch → score is nan."""
    n, c, d, h, w = 3, 3, 2, 2, 2

    # All volumes: GT is all liver (class 1); pred says all liver (class 1)
    target = torch.ones(n, d, h, w, dtype=torch.long)  # all class 1
    pred = torch.zeros(n, c, d, h, w, dtype=torch.float32)
    pred[:, 1, :, :, :] = 1.0  # predict class 1 everywhere

    scores = dice_per_case(pred, target, num_classes=3)

    # class 2 (tumor) never appears in GT or pred → excluded → nan
    assert torch.isnan(scores[2]), (
        f"class 2 absent everywhere: expected nan, got {scores[2].item()}"
    )
