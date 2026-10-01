"""Independent native-grid expectations; no resize/grid helpers used as oracle."""

from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from dense_unet_3d.cli import _load_model_from_checkpoint, _TinyStub
from dense_unet_3d.dataset.slabs import NativeSlabDataset, categorical_tile, spatial_config
from dense_unet_3d.evaluation.evaluate import EvaluationInterrupted, evaluate
from dense_unet_3d.evaluation.predict import PredictionInterrupted, predict_volume
from dense_unet_3d.training.experiment import experiment_metadata
from dense_unet_3d.training.train import unpack_training_batch
from scripts.census_reconstruction import audit_case
from tests.dataset.test_slabs import config, write_case
from tests.evaluation.test_native_cases import LabelModel


def tile_config(height=3, width=4, depth=2):
    return {
        **config(height, width, depth),
        "sampling": "native_slabs",
        "resize_img": False,
        "inplane_representation": "native_tiles_v1",
    }


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 1), (3, 4, 2), (5, 7, 5), (7, 2, 3)])
def test_categorical_reassembly_components_padding_and_world(tmp_path, shape):
    affine = np.array([[0, -2, 0.1, 40], [1.5, 0, 0.2, -30], [0.3, 0, 4, 6], [0, 0, 0, 1]])
    # Edge singletons and diagonal contacts cover connectivity and tile boundaries.
    labels = np.zeros(shape, dtype=np.uint8)
    labels[0, 0, 0] = labels[-1, -1, -1] = 2
    if min(shape) > 1:
        labels[1, 1, 1] = 2
    image, target = write_case(tmp_path, "a", labels, affine)
    cfg = tile_config()
    dataset = NativeSlabDataset([str(tmp_path)], cfg)
    sums, counts = np.zeros(shape, np.int64), np.zeros(shape, np.int64)
    seen = []
    for sample in dataset:
        d, h, w = sample["start"].tolist()
        vd, vh, vw = sample["valid_extents"].tolist()
        seen.append((d, h, w))
        assert torch.equal(sample["valid_mask"], sample["target"] != -100)
        assert sample["valid_mask"].sum() == vd * vh * vw
        assert torch.all(sample["image"][~sample["valid_mask"]] == -200)
        block = sample["target"][0, :vd, :vh, :vw].numpy().transpose(1, 2, 0)
        sums[h : h + vh, w : w + vw, d : d + vd] += block
        counts[h : h + vh, w : w + vw, d : d + vd] += 1
        point = np.array([vd - 1, vh - 1, vw - 1, 1])
        source = np.array([h + vh - 1, w + vw - 1, d + vd - 1, 1])
        np.testing.assert_array_equal(sample["model_to_source"].numpy() @ point, source)
        np.testing.assert_allclose(sample["model_to_world"].numpy() @ point, affine @ source)
    assert seen == sorted(set(seen))
    assert np.all(counts > 0)
    restored = sums // counts
    np.testing.assert_array_equal(restored, labels)
    native_ids, n = ndimage.label(labels == 2, np.ones((3, 3, 3)))
    restored_ids, m = ndimage.label(restored == 2, np.ones((3, 3, 3)))
    assert n == m
    np.testing.assert_array_equal(native_ids, restored_ids)
    np.testing.assert_array_equal(
        predict_volume(LabelModel(), torch.device("cpu"), nib.load(image), cfg), labels
    )
    receipt = audit_case(image, target, dataset.geometry)
    assert receipt["gate_pass"] and receipt["erased_components"] == 0
    assert all(c["model_physical_volume_ratio"] == 1 for c in receipt["components"])


def test_index_has_declared_tail_order_and_ignores_labels(tmp_path):
    _, target = write_case(tmp_path, "a", np.zeros((5, 7, 5), np.uint8))
    dataset = NativeSlabDataset([str(tmp_path)], tile_config())
    expected = [(0, (d, h, w), (2, 3, 4)) for d in (0, 2, 3) for h in (0, 2) for w in (0, 3)]
    assert dataset.sample_index == expected
    nib.save(nib.Nifti1Image(np.full((5, 7, 5), 2, np.uint8), np.eye(4)), target)
    assert NativeSlabDataset([str(tmp_path)], tile_config()).sample_index == expected
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    _, labels = unpack_training_batch(batch, "cpu")
    assert torch.all(labels == 2)


def test_pixel_overlap_uses_probability_mean_then_argmax():
    class Conflicts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, image):
            values = [0.51, 0.48, 0.01] if self.calls == 0 else [0.01, 0.98, 0.01]
            self.calls += 1
            # Padded output strongly tumor; must not reach native grid.
            result = (
                torch.tensor(values).log()[None, :, None, None, None].expand(1, 3, 2, 3, 4).clone()
            )
            result[:, :, 1] = torch.tensor([0.01, 0.01, 0.98]).log()[None, :, None, None]
            return result

    image = nib.Nifti1Image(np.zeros((5, 1, 1)), np.eye(4))
    model = Conflicts()
    pred = predict_volume(model, torch.device("cpu"), image, tile_config())
    np.testing.assert_array_equal(pred[:, 0, 0], [0, 0, 1, 1, 1])
    assert model.calls == 2


def test_tile_case_metrics_and_interruptions(tmp_path):
    labels = np.zeros((5, 7, 5), np.uint8)
    labels[0, 0, 0] = labels[-1, -1, -1] = 2
    image, _ = write_case(tmp_path, "a", labels)
    dataset = NativeSlabDataset([str(tmp_path)], tile_config())
    result = evaluate(LabelModel(), torch.device("cpu"), DataLoader(dataset))
    assert set(result.values()) == {1.0}
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return "stop" if calls > 4 else None

    with pytest.raises(PredictionInterrupted):
        predict_volume(
            LabelModel(), torch.device("cpu"), nib.load(image), tile_config(), stop_requested=stop
        )
    with pytest.raises(EvaluationInterrupted):
        evaluate(
            LabelModel(), torch.device("cpu"), DataLoader(dataset), stop_requested=lambda: "stop"
        )


def test_native_checkpoint_identity_rejects_resize_and_changed_grid(tmp_path):
    cfg = {"dataset": tile_config()}
    path = tmp_path / "native.pt"
    model = _TinyStub(torch.nn.Conv3d(1, 3, 1))
    torch.save({"model_state_dict": model.state_dict(), **experiment_metadata(cfg)}, path)
    assert isinstance(_load_model_from_checkpoint(str(path), torch.device("cpu"), cfg), _TinyStub)
    for update in (
        {"inplane_representation": "full_fov_resize", "resize_img": True},
        {"resize_dims": {"D": 2, "H": 4, "W": 4}},
    ):
        wrong = deepcopy(cfg)
        wrong["dataset"].update(update)
        with pytest.raises(ValueError, match="preprocessing mismatch"):
            _load_model_from_checkpoint(
                str(path), torch.device("cpu"), wrong, allow_legacy_preprocessing=True
            )
    with pytest.raises(ValueError, match="requires resize_img=false"):
        spatial_config({**tile_config(), "resize_img": True})
    with pytest.raises(ValueError, match="augmentation is unresolved"):
        spatial_config({**tile_config(), "scale_img": True})


@pytest.mark.parametrize("checkpoint_explicit", [False, True])
def test_native_checkpoint_resize_default_round_trips(tmp_path, checkpoint_explicit):
    cfg = {"dataset": tile_config()}
    omitted = deepcopy(cfg)
    omitted["dataset"].pop("resize_img")
    checkpoint_config, load_config = (cfg, omitted) if checkpoint_explicit else (omitted, cfg)
    assert (
        experiment_metadata(cfg)["preprocessing_identity"]
        == experiment_metadata(omitted)["preprocessing_identity"]
    )
    model = _TinyStub(torch.nn.Conv3d(1, 3, 1))
    path = tmp_path / "native.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), **experiment_metadata(checkpoint_config)}, path
    )

    loaded = _load_model_from_checkpoint(str(path), torch.device("cpu"), load_config)

    sample = torch.arange(24, dtype=torch.float32).reshape(1, 1, 2, 3, 4)
    torch.testing.assert_close(loaded(sample), model(sample))


@pytest.mark.parametrize("compressed", [False, True])
def test_native_tile_reads_target_proxy_once_and_pads_all_axes(tmp_path, monkeypatch, compressed):
    labels = np.array([[[1], [2]]], dtype=np.uint8)
    image_path, target_path = write_case(tmp_path, "a", labels)
    if compressed:
        for source in (image_path, target_path):
            nib.save(nib.load(source), source + ".gz")
            Path(source).unlink()
        target_path += ".gz"
    dataset = NativeSlabDataset([str(tmp_path)], tile_config())
    reads = []
    original_getitem = nib.arrayproxy.ArrayProxy.__getitem__

    def read(proxy, slices):
        if str(proxy.file_like) == target_path:
            reads.append(slices)
        return original_getitem(proxy, slices)

    monkeypatch.setattr(nib.arrayproxy.ArrayProxy, "__getitem__", read)
    sample = dataset[0]

    assert len(reads) == 1
    expected = torch.full((1, 2, 3, 4), -100, dtype=torch.int64)
    expected[0, 0, 0, :2] = torch.tensor([1, 2])
    torch.testing.assert_close(sample["target"], expected)
    assert torch.equal(sample["valid_mask"], expected != -100)


@pytest.mark.parametrize("label", [np.nan, 0.5, 3.0])
def test_native_tile_validates_labels_before_integer_conversion(tmp_path, label):
    _, target_path = write_case(tmp_path, "a", np.zeros((1, 1, 1), np.uint8))
    nib.save(nib.Nifti1Image(np.full((1, 1, 1), label), np.eye(4)), target_path)
    dataset = NativeSlabDataset([str(tmp_path)], tile_config())
    with pytest.raises(ValueError, match="target tile labels must be finite integers"):
        dataset[0]


def test_categorical_helper_retains_arbitrary_grid_and_identity_values():
    geometry = spatial_config(tile_config(height=2, width=5, depth=3))
    source = np.arange(35, dtype=np.int64).reshape(5, 7, 1) + 1000
    target, valid = categorical_tile(source, (0, 3, 4), geometry)
    assert valid == (1, 2, 3)
    expected = torch.full((1, 3, 2, 5), -100, dtype=torch.int64)
    expected[0, 0, :2, :3] = torch.tensor([[1025, 1026, 1027], [1032, 1033, 1034]])
    torch.testing.assert_close(target, expected)


def test_unique_coordinate_ramp_and_uncovered_pixel_rejection(tmp_path, monkeypatch):
    shape = (5, 7, 5)
    ramp = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    image_path, _ = write_case(tmp_path, "a", np.zeros(shape, np.uint8))
    nib.save(nib.Nifti1Image(ramp, np.eye(4)), image_path)
    dataset = NativeSlabDataset([str(tmp_path)], tile_config())
    restored = np.full(shape, np.nan)
    for sample in dataset:
        d, h, w = sample["start"].tolist()
        vd, vh, vw = sample["valid_extents"].tolist()
        restored[h : h + vh, w : w + vw, d : d + vd] = (
            sample["image"][0, :vd, :vh, :vw].numpy().transpose(1, 2, 0)
        )
    np.testing.assert_array_equal(restored, ramp)
    # A missed in-plane tile must withhold the entire source prediction.
    monkeypatch.setattr(
        "dense_unet_3d.evaluation.predict.tile_bounds", lambda *_: iter([((0, 0, 0), (2, 3, 4))])
    )
    with pytest.raises(ValueError, match="uncovered native pixels"):
        predict_volume(LabelModel(), torch.device("cpu"), nib.load(image_path), tile_config())


def test_all_axis_padding_has_zero_loss_gradient(tmp_path):
    from dense_unet_3d.training.loss import get_criterion

    write_case(tmp_path, "a", np.full((1, 2, 1), 2, np.uint8))
    dataset = NativeSlabDataset([str(tmp_path)], tile_config())
    batch = next(iter(DataLoader(dataset)))
    _, target = unpack_training_batch(batch, "cpu")
    logits = torch.randn((1, 3, 2, 3, 4), requires_grad=True)
    criterion = get_criterion(
        {
            "training": {
                "criterion": "CrossEntropyLoss",
                "class_weights": {"background": 0.2, "liver": 1.2, "lesion": 2.2},
                "loss_reduction": "valid_voxel_mean",
            }
        },
        torch.device("cpu"),
    )
    criterion(logits, target).backward()
    valid = batch["valid_mask"].expand(-1, 3, -1, -1, -1)
    assert torch.all(logits.grad[~valid] == 0)
    assert torch.count_nonzero(logits.grad[valid]) > 0


@pytest.mark.parametrize("sampling", [None, "whole_volume"])
def test_tile_representation_requires_native_slab_sampling(sampling):
    from dense_unet_3d.dataset.prepare_dataset import sampling_mode
    from dense_unet_3d.training.experiment import preprocessing_identity

    cfg = {"dataset": tile_config()}
    if sampling is None:
        cfg["dataset"].pop("sampling")
    else:
        cfg["dataset"]["sampling"] = sampling
    for validate in (sampling_mode, preprocessing_identity):
        with pytest.raises(ValueError, match="requires explicit dataset.sampling"):
            validate(cfg)

        with pytest.raises(ValueError, match="requires explicit dataset.sampling"):
            spatial_config(cfg["dataset"])
    with pytest.raises(ValueError, match="requires explicit dataset.sampling"):
        predict_volume(
            LabelModel(),
            torch.device("cpu"),
            nib.Nifti1Image(np.zeros((1, 1, 1)), np.eye(4)),
            cfg["dataset"],
        )
