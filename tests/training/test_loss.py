"""Tests for weighted cross-entropy loss (E1).

Acceptance criteria:
- Class weights are read from config (background/liver/lesion -> classes 0/1/2).
- Absent config weights fall back to defaults [0.2, 1.2, 2.2].
- Loss matches hand-computed weighted-CE reference within tolerance.
- Weight ordering is [bg, liver, lesion].
- Weight tensor lands on the requested device (CPU in this test).
- Criterion object is not .to()-moved (no nn.Parameters).
"""

import pytest
import torch
import torch.nn.functional as F

from dense_unet_3d.training.loss import CLASS_WEIGHTS, get_criterion

# ------------------------------------------------------------------ fixtures


@pytest.fixture
def base_config() -> dict:
    """Minimal config carrying class weights under training.class_weights."""
    return {
        "training": {
            "criterion": "CrossEntropyLoss",
            "class_weights": {
                "background": 0.2,
                "liver": 1.2,
                "lesion": 2.2,
            },
        },
    }


@pytest.fixture
def toy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Toy (N=2, C=3, D=2, H=4, W=4) logits and (N,D,H,W) int target."""
    torch.manual_seed(42)
    logits = torch.randn(2, 3, 2, 4, 4)
    target = torch.randint(0, 3, (2, 2, 4, 4))
    return logits, target


# ------------------------------------------------------------------ tests


def test_class_weights_ordering() -> None:
    """Default weight tensor must be [bg=0.2, liver=1.2, lesion=2.2]."""
    expected = torch.tensor([0.2, 1.2, 2.2])
    assert torch.allclose(CLASS_WEIGHTS, expected), (
        f"CLASS_WEIGHTS mismatch: expected {expected}, got {CLASS_WEIGHTS}"
    )


def test_weights_read_from_config() -> None:
    """get_criterion must read class weights from config, not a hardcoded list."""
    config = {
        "training": {
            "criterion": "CrossEntropyLoss",
            "class_weights": {
                "background": 0.1,
                "liver": 1.0,
                "lesion": 5.0,
            },
        },
    }
    criterion = get_criterion(config)
    expected = torch.tensor([0.1, 1.0, 5.0])
    assert criterion.weight is not None
    assert torch.allclose(criterion.weight.cpu(), expected), (
        f"Weight mismatch: got {criterion.weight}, expected {expected}"
    )


def test_weights_fall_back_to_defaults_when_absent() -> None:
    """When config has no class_weights key, fall back to [0.2, 1.2, 2.2]."""
    config = {"training": {"criterion": "CrossEntropyLoss"}}
    criterion = get_criterion(config)
    expected = torch.tensor([0.2, 1.2, 2.2])
    assert criterion.weight is not None
    assert torch.allclose(criterion.weight.cpu(), expected), (
        f"Default weight mismatch: got {criterion.weight}, expected {expected}"
    )


def test_loss_matches_hand_computed_reference(
    base_config: dict,
    toy_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Loss from get_criterion() matches torch reference with the same weights."""
    logits, target = toy_batch
    device = logits.device

    criterion = get_criterion(base_config, device=device)

    weight_ref = torch.tensor([0.2, 1.2, 2.2], dtype=torch.float32, device=device)
    ref_criterion = torch.nn.CrossEntropyLoss(weight=weight_ref)

    actual = criterion(logits, target)
    expected = ref_criterion(logits, target)

    assert torch.allclose(actual, expected, atol=1e-6), (
        f"Loss mismatch: actual={actual.item():.6f}, expected={expected.item():.6f}"
    )


def test_weight_tensor_on_requested_device(
    base_config: dict,
    toy_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """The criterion's weight tensor must be on the requested device."""
    logits, _ = toy_batch
    device = logits.device

    criterion = get_criterion(base_config, device=device)

    assert criterion.weight is not None
    assert criterion.weight.device == device, (
        f"Weight on {criterion.weight.device}, expected {device}"
    )


def test_criterion_object_not_to_moved(base_config: dict) -> None:
    """Criterion itself must NOT be an nn.Module that was .to(device)-moved.

    nn.CrossEntropyLoss has no trainable parameters; only the weight tensor is
    moved to device, not the module.
    """
    criterion = get_criterion(base_config)

    params = list(criterion.parameters())
    assert params == [], (
        "Criterion should have no nn.Parameters — do not add the weight as a "
        f"Parameter or move the criterion as a model. Got params: {params}"
    )
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)


def test_loss_is_finite_on_toy_input(
    base_config: dict,
    toy_batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Loss value is finite (not NaN or Inf) on valid toy logits + target."""
    logits, target = toy_batch
    criterion = get_criterion(base_config)
    loss = criterion(logits, target)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


def test_loss_value_numerical_check(base_config: dict) -> None:
    """Hand-verify loss for a single-pixel, strong-class-0 scenario."""
    logits = torch.zeros(1, 3, 1, 1, 1)
    logits[0, 0, 0, 0, 0] = 10.0  # strong prediction for class 0
    target = torch.zeros(1, 1, 1, 1, dtype=torch.long)  # class 0

    criterion = get_criterion(base_config)
    actual_loss = criterion(logits, target)

    weight = torch.tensor([0.2, 1.2, 2.2])
    expected_loss = F.cross_entropy(logits, target, weight=weight)

    assert torch.allclose(actual_loss, expected_loss, atol=1e-6), (
        f"Numerical check failed: actual={actual_loss.item():.8f}, "
        f"expected={expected_loss.item():.8f}"
    )


@pytest.mark.parametrize("reduction", ["valid_voxel_mean", "weighted_mean"])
def test_manual_loss_gradient_and_ignored_padding(reduction: str) -> None:
    """The two denominators differ, and padded logits have zero gradient."""
    cfg = {"training": {"loss_reduction": reduction}}
    criterion = get_criterion(cfg).double()
    logits = torch.zeros(1, 3, 1, 1, 7, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[[[0, 0, 1, 2, -100, -100, -100]]]])
    loss = criterion(logits, target)
    loss.backward()
    weights = criterion.weight
    assert weights is not None
    denominator = 4.0 if reduction == "valid_voxel_mean" else float(weights[target[..., :4]].sum())
    expected = (
        weights[target[..., :4]].sum()
        * torch.log(torch.tensor(3.0, dtype=torch.float64))
        / denominator
    )
    assert torch.allclose(loss, expected)
    expected_grad = torch.zeros_like(logits)
    for i, label in enumerate([0, 0, 1, 2]):
        expected_grad[0, :, 0, 0, i] = weights[label] / denominator / 3
        expected_grad[0, label, 0, 0, i] -= weights[label] / denominator
    assert logits.grad is not None
    assert torch.allclose(logits.grad, expected_grad)
    assert logits.grad[0, 2, 0, 0, 3] < 0
    assert logits.grad[..., 4:].count_nonzero() == 0


def test_reductions_have_distinct_gradient_scale() -> None:
    logits = torch.zeros(1, 3, 1, 1, 4, requires_grad=True)
    target = torch.tensor([[[[0, 0, 1, 2]]]])
    voxel = get_criterion({"training": {"loss_reduction": "valid_voxel_mean"}})
    weighted = get_criterion({"training": {"loss_reduction": "weighted_mean"}})
    voxel_gradient = torch.autograd.grad(voxel(logits, target), logits, retain_graph=True)[0]
    weighted_gradient = torch.autograd.grad(weighted(logits, target), logits)[0]
    assert torch.allclose(voxel_gradient, weighted_gradient * 0.95)
    assert not torch.allclose(voxel_gradient, weighted_gradient)


@pytest.mark.parametrize("reduction", ["valid_voxel_mean", "weighted_mean"])
def test_all_ignored_batch_rejected(reduction: str) -> None:
    criterion = get_criterion({"training": {"loss_reduction": reduction}})
    with pytest.raises(ValueError, match="no valid"):
        criterion(torch.zeros(1, 3, 1, 1, 2), torch.full((1, 1, 1, 2), -100))


def test_unknown_loss_reduction_rejected() -> None:
    with pytest.raises(ValueError, match="loss_reduction"):
        get_criterion({"training": {"loss_reduction": "batch_mean"}})
