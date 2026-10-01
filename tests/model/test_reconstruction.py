"""Graph identity, figure geometry, and small CPU numerical counterexamples."""

import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dense_unet_3d.model.config import (
    FIGURE,
    HISTORICAL,
    build_model,
    canonical_model_config,
    model_manifest,
    model_metadata,
    validate_model_metadata,
)
from dense_unet_3d.model.reconstruction import (
    FigureDenseBlock,
    FigureUpsample,
    SamePad3d,
    batch_norm,
)


def test_figure_stage_manifest_and_operators():
    with torch.device("meta"):
        model = build_model(FIGURE)
    manifest = model_manifest(model)
    assert manifest["trainable_parameters"] == 64_591_723
    assert manifest["skip_sources"] == [
        "dense_block4",
        "dense_block3",
        "dense_block2",
        "dense_block1",
        "stem",
    ]
    assert [(s["shape"][1], s["shape"][2:]) for s in manifest["stages"]] == [
        (96, [6, 112, 112]),
        (96, [3, 56, 56]),
        (224, [3, 56, 56]),
        (112, [3, 28, 28]),
        (496, [3, 28, 28]),
        (248, [3, 14, 14]),
        (1016, [3, 14, 14]),
        (508, [3, 7, 7]),
        (1660, [3, 7, 7]),
        (504, [3, 14, 14]),
        (224, [3, 28, 28]),
        (192, [3, 56, 56]),
        (96, [6, 112, 112]),
        (64, [12, 224, 224]),
        (3, [12, 224, 224]),
    ]
    assert [len(getattr(model, f"dense_block{i}").layers) for i in range(1, 5)] == [4, 12, 24, 36]
    assert model.dense_block4.layers[0].bottleneck[0].out_channels == 32
    assert model.dense_block1.layers[0].growth[0].groups == 128
    for i in range(1, 6):
        conv = getattr(model, f"up{i}").conv[0]
        assert conv.groups == 1 and conv.kernel_size == (3, 3, 3)
    assert [type(m) for m in model.transition1] == [nn.Conv3d, nn.BatchNorm3d, nn.ReLU, nn.Conv3d]
    assert model.transition1[-1].stride == (1, 2, 2)


def test_same_padding_is_asymmetric_and_not_historical_sampling():
    # Same output count does not establish the same sampling coordinates.
    x = torch.arange(12.0).reshape(1, 1, 12, 1, 1).expand(1, 1, 12, 8, 8)
    kernel = torch.zeros(1, 1, 7, 7, 7)
    kernel[0, 0, 3, 3, 3] = 1
    same = F.conv3d(SamePad3d(7, 2)(x), kernel, stride=2)
    symmetric = F.conv3d(x, kernel, stride=2, padding=3)
    assert same.shape == symmetric.shape
    assert same[0, 0, :, 1, 1].tolist() == [1, 3, 5, 7, 9, 11]
    assert symmetric[0, 0, :, 1, 1].tolist() == [0, 2, 4, 6, 8, 10]
    pool = nn.MaxPool3d(3, 2)(SamePad3d(3, 2, -float("inf"))(x))
    assert pool.shape[-3:] == (6, 4, 4)
    assert pool[0, 0, :, 1, 1].tolist() == [2, 4, 6, 8, 10, 11]


def test_dense_concat_and_both_skip_gradients_finite():
    torch.manual_seed(8)
    dense = FigureDenseBlock(4, 3, 8)
    x = torch.randn(2, 4, 3, 4, 4, requires_grad=True)
    y = dense(x)
    assert y.shape[1] == 100
    torch.testing.assert_close(y[:, :4], x)
    skip = torch.randn(2, 7, 3, 4, 4, requires_grad=True)
    up = FigureUpsample(100, 7, 5, (3, 8, 8))
    z = up(y, skip)
    z.square().mean().backward()
    assert torch.isfinite(z).all()
    for tensor in (x, skip, *dense.parameters(), *up.parameters()):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    assert skip.grad.abs().sum() > 0


def test_bn_explicit_equation_and_running_variance_distinction():
    x = torch.tensor([0.0, 2.0, 4.0, 6.0]).reshape(1, 1, 1, 2, 2)
    bn = batch_norm(1)
    y = bn(x)
    torch.testing.assert_close(y, (x - 3) / torch.sqrt(torch.tensor(5.001)))
    assert bn.running_mean.item() == pytest.approx(0.03)
    assert bn.running_var.item() == pytest.approx(0.99 + 0.01 * 20 / 3)
    assert bn.running_var.item() != pytest.approx(0.99 + 0.01 * 5)  # tf.keras population option


def test_metadata_roundtrip_and_same_shape_graph_rejection():
    model = build_model(HISTORICAL)
    checkpoint = {**model_metadata(model), "model_state_dict": model.state_dict()}
    stream = io.BytesIO()
    torch.save(checkpoint, stream)
    stream.seek(0)
    loaded = torch.load(stream, weights_only=True)
    config = validate_model_metadata(loaded)
    restored = build_model(config)
    restored.load_state_dict(loaded["model_state_dict"])
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name])
    changed = copy.deepcopy(loaded)
    changed["model_config"]["resize"] = "main_only_trilinear_align_corners_false"
    with pytest.raises(ValueError, match="configuration"):
        validate_model_metadata(changed)
    with pytest.raises(ValueError, match="configuration"):
        validate_model_metadata(loaded, FIGURE)
    with pytest.raises(ValueError, match="identity"):
        validate_model_metadata({})
    changed = copy.deepcopy(loaded)
    changed["model_fingerprint"] = "bad"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_model_metadata(changed)
    altered = canonical_model_config(FIGURE)
    altered["bottlenecks"][-1] = 128
    with pytest.raises(ValueError, match="configuration"):
        build_model(altered)


def test_figure_meta_checkpoint_roundtrip():
    with torch.device("meta"):
        model = build_model(FIGURE)
        state = {**model_metadata(model), "model_state_dict": model.state_dict()}
        restored = build_model(validate_model_metadata(state, FIGURE))
        restored.load_state_dict(state["model_state_dict"])
    assert model_metadata(restored) == model_metadata(model)


def test_momentum_lr_boundary_is_not_keras_velocity_equivalence():
    p = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    optimizer = torch.optim.SGD(
        [p], lr=0.1, momentum=0.5, dampening=0, nesterov=False, weight_decay=0
    )
    keras_p, velocity = 1.0, 0.0
    for lr in (0.1, 0.05):
        p.grad = torch.tensor(1.0, dtype=torch.float64)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.step()
        velocity = 0.5 * velocity - lr
        keras_p += velocity
    assert p.item() == pytest.approx(0.825)
    assert keras_p == pytest.approx(0.8)


def test_figure_full_input_forward_backward_finite():
    torch.manual_seed(42)
    model = build_model(FIGURE)
    x = torch.randn(1, 1, 12, 224, 224, requires_grad=True)
    output = model(x)
    assert output.shape == (1, 3, 12, 224, 224)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
