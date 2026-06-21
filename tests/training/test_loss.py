"""Tests for weighted cross-entropy loss (E1).

Acceptance criteria:
- Loss matches hand-computed weighted-CE reference within tolerance.
- Weight ordering is [0.2, 1.2, 2.2] for [bg, liver, lesion].
- Weight tensor lands on the input's device (CPU in this test).
- Criterion object is not .to()-moved.
"""

import pytest
import torch
import torch.nn.functional as F

from dense_unet_3d.training.loss import CLASS_WEIGHTS, get_criterion

# ------------------------------------------------------------------ fixtures


@pytest.fixture
def toy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Toy (N=2, C=3, D=2, H=4, W=4) logits and (N,D,H,W) int target."""
    torch.manual_seed(42)
    logits = torch.randn(2, 3, 2, 4, 4)
    target = torch.randint(0, 3, (2, 2, 4, 4))
    return logits, target


# ------------------------------------------------------------------ tests


def test_class_weights_ordering() -> None:
    """Weight tensor must be [bg=0.2, liver=1.2, lesion=2.2] in that order."""
    expected = torch.tensor([0.2, 1.2, 2.2])
    assert torch.allclose(CLASS_WEIGHTS, expected), (
        f"CLASS_WEIGHTS mismatch: expected {expected}, got {CLASS_WEIGHTS}"
    )


def test_loss_matches_hand_computed_reference(
    toy_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Loss from get_criterion() matches torch reference with the same weights."""
    logits, target = toy_batch
    device = logits.device

    criterion = get_criterion(logits)

    # Reference: build criterion the same way but inline
    weight_ref = torch.tensor([0.2, 1.2, 2.2], dtype=torch.float32, device=device)
    ref_criterion = torch.nn.CrossEntropyLoss(weight=weight_ref)

    actual = criterion(logits, target)
    expected = ref_criterion(logits, target)

    assert torch.allclose(actual, expected, atol=1e-6), (
        f"Loss mismatch: actual={actual.item():.6f}, expected={expected.item():.6f}"
    )


def test_weight_tensor_on_input_device(
    toy_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """The criterion's weight tensor must be on the same device as the input."""
    logits, target = toy_batch
    device = logits.device

    criterion = get_criterion(logits)

    # The weight attribute of CrossEntropyLoss should be on CPU (same as input)
    assert criterion.weight is not None
    assert criterion.weight.device == device, (
        f"Weight on {criterion.weight.device}, expected {device}"
    )


def test_criterion_object_not_to_moved(
    toy_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Criterion itself must NOT be an nn.Module that was .to(device)-moved.

    We verify this structurally: CrossEntropyLoss is an nn.Module but the
    standard usage is to only move the weight tensor, not the module itself.
    The criterion should have no parameters (weight is a buffer attr, not a
    Parameter), so criterion.parameters() should be empty.
    """
    logits, _ = toy_batch
    criterion = get_criterion(logits)

    # nn.CrossEntropyLoss has no trainable parameters
    params = list(criterion.parameters())
    assert params == [], (
        "Criterion should have no nn.Parameters — do not add the weight as a "
        f"Parameter or move the criterion as a model. Got params: {params}"
    )

    # Confirm the criterion is a plain CrossEntropyLoss (not a custom wrapped module)
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)


def test_loss_is_finite_on_toy_input(
    toy_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Loss value is finite (not NaN or Inf) on valid toy logits + target."""
    logits, target = toy_batch
    criterion = get_criterion(logits)
    loss = criterion(logits, target)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


def test_loss_value_numerical_check() -> None:
    """Hand-verify loss for a single-pixel, single-class scenario.

    logits = [[10, 0, 0]] → predicts class 0 (bg) with high confidence.
    target = [0] → correct prediction.
    With weight [0.2, 1.2, 2.2], CE for class-0 with very high logit ≈ 0.
    Cross-check via F.cross_entropy with explicit weight.
    """
    torch.manual_seed(0)
    # Shape: (N=1, C=3, D=1, H=1, W=1)
    logits = torch.tensor([[[[[10.0]], [[0.0]], [[-10.0]]]]]).permute(0, 1, 2, 3, 4)
    # Reshape to (N=1, C=3, D=1, H=1, W=1)
    logits = torch.zeros(1, 3, 1, 1, 1)
    logits[0, 0, 0, 0, 0] = 10.0  # strong prediction for class 0
    target = torch.zeros(1, 1, 1, 1, dtype=torch.long)  # class 0

    criterion = get_criterion(logits)
    actual_loss = criterion(logits, target)

    weight = torch.tensor([0.2, 1.2, 2.2])
    expected_loss = F.cross_entropy(logits, target, weight=weight)

    assert torch.allclose(actual_loss, expected_loss, atol=1e-6), (
        f"Numerical check failed: actual={actual_loss.item():.8f}, "
        f"expected={expected_loss.item():.8f}"
    )
