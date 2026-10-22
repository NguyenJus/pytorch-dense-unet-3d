import nibabel as nib
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from dense_unet_3d.dataset.slabs import NativeSlabDataset
from dense_unet_3d.evaluation.evaluate import EvaluationInterrupted, evaluate
from dense_unet_3d.evaluation.predict import PredictionInterrupted, predict_volume
from tests.dataset.test_slabs import config, write_case


class LabelModel(nn.Module):
    def forward(self, image):
        labels = image[:, 0].round().long().clamp(0, 2)
        logits = torch.full((image.shape[0], 3, *image.shape[2:]), -20.0, device=image.device)
        return logits.scatter_(1, labels[:, None], 20.0)


def test_complete_case_native_grid_and_historical_empty_policy(tmp_path):
    labels = np.zeros((5, 7, 25), dtype=np.uint8)
    labels[0, 0, 0] = labels[4, 6, -1] = 2
    labels[2, 3, :] = 1
    write_case(tmp_path, "a", labels)
    write_case(tmp_path, "b", np.zeros((5, 7, 3), dtype=np.uint8))
    dataset = NativeSlabDataset([str(tmp_path)], config())
    model, device = LabelModel(), torch.device("cpu")
    pred = predict_volume(model, device, nib.load(dataset.cases[0][0]), config())
    np.testing.assert_array_equal(pred, labels)
    result = evaluate(model, device, DataLoader(dataset, batch_size=2))
    assert result == {
        "liver_per_case": 0.5,
        "liver_global": 1.0,
        "tumor_per_case": 0.5,
        "tumor_global": 1.0,
    }
    with pytest.raises(EvaluationInterrupted, match="cohort metrics withheld"):
        evaluate(model, device, DataLoader(dataset), max_batches=1)
    with pytest.raises(EvaluationInterrupted):
        evaluate(model, device, DataLoader(dataset), stop_requested=lambda: "stop")


def test_empty_cohort_returns_numeric_zero(tmp_path):
    write_case(tmp_path, "a", np.zeros((5, 7, 3), dtype=np.uint8))
    dataset = NativeSlabDataset([str(tmp_path)], config())
    result = evaluate(LabelModel(), torch.device("cpu"), DataLoader(dataset))
    assert set(result.values()) == {0.0}


def test_overlap_averages_probabilities_before_argmax():
    class ConflictingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, image):
            # Per tile, in predictor order: the first tile conflicts with the rest.
            tiles = []
            for _ in range(image.shape[0]):
                values = [0.51, 0.48, 0.01] if self.calls == 0 else [0.01, 0.98, 0.01]
                self.calls += 1
                tiles.append(torch.tensor(values).log()[:, None, None, None].expand(3, 12, 1, 1))
            return torch.stack(tiles)

    image = nib.Nifti1Image(np.zeros((1, 1, 13)), np.eye(4))
    pred = predict_volume(ConflictingModel(), torch.device("cpu"), image, config(1, 1))
    assert pred[0, 0, 0] == 0
    assert np.all(pred[0, 0, 1:] == 1)


def test_late_corrupt_target_withholds_all_metrics(tmp_path):
    write_case(tmp_path, "a", np.ones((5, 7, 13), dtype=np.uint8))
    write_case(tmp_path, "b", np.full((5, 7, 3), 3, dtype=np.uint8))
    dataset = NativeSlabDataset([str(tmp_path)], config())
    with pytest.raises(ValueError, match="native target labels"):
        evaluate(LabelModel(), torch.device("cpu"), DataLoader(dataset))


def test_predictor_image_only_nonfinite_and_interruption():
    bad = nib.Nifti1Image(np.full((5, 7, 3), np.nan), np.eye(4))
    with pytest.raises(ValueError, match="nonfinite"):
        predict_volume(LabelModel(), torch.device("cpu"), bad, config())
    with pytest.raises(PredictionInterrupted):
        predict_volume(
            LabelModel(), torch.device("cpu"), bad, config(), stop_requested=lambda: "stop"
        )


class RampModel(nn.Module):
    """Elementwise, batch-invariant logits that differ between overlapping tiles."""

    def __init__(self):
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, image):
        self.batch_sizes.append(image.shape[0])
        x = image[:, 0]
        return torch.stack([(x * 0.37).sin(), (x * 0.11).cos(), (x * 0.05).sin() * 2], dim=1)


def tiles(height, width, depth):
    return {
        **config(height, width, depth),
        "sampling": "native_slabs",
        "resize_img": False,
        "inplane_representation": "native_tiles_v1",
    }


@pytest.mark.parametrize(
    ("shape", "cfg"),
    [
        ((5, 7, 25), config()),
        ((5, 7, 3), config()),
        ((9, 11, 13), config(4, 6, 5)),
        ((5, 7, 5), tiles(3, 4, 2)),
        ((9, 11, 13), tiles(4, 6, 5)),
        ((2, 3, 1), tiles(3, 4, 2)),
    ],
)
def test_batched_prediction_is_array_equal_to_batch_one(shape, cfg):
    ramp = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * 0.7 - 40
    image = nib.Nifti1Image(ramp, np.eye(4))
    one, eight = RampModel(), RampModel()
    single = predict_volume(one, torch.device("cpu"), image, {**cfg, "inference_batch": 1})
    batched = predict_volume(eight, torch.device("cpu"), image, {**cfg, "inference_batch": 8})
    assert single.dtype == batched.dtype == np.uint8
    np.testing.assert_array_equal(batched, single)
    assert set(one.batch_sizes) == {1}
    assert sum(eight.batch_sizes) == len(one.batch_sizes)
    assert eight.batch_sizes == [8] * (len(one.batch_sizes) // 8) + (
        [len(one.batch_sizes) % 8] if len(one.batch_sizes) % 8 else []
    )


@pytest.mark.parametrize("native", [True, False])
def test_predictor_loads_case_image_once(native):
    class CountingProxy:
        def __init__(self, array):
            self.array, self.reads = array, 0
            self.shape, self.ndim = array.shape, array.ndim

        def __array__(self, dtype=None, copy=None):
            self.reads += 1
            return self.array.astype(dtype or self.array.dtype)

        def __getitem__(self, key):
            self.reads += 1
            return self.array[key]

    labels = np.zeros((9, 11, 13), dtype=np.float32)
    labels[2:5, 3:8, 4:9] = 1
    labels[3, 4, 5:7] = 2
    image = nib.Nifti1Image(labels, np.eye(4))
    proxy = CountingProxy(labels)
    image._dataobj = proxy
    cfg = tiles(4, 6, 5) if native else config(4, 6, 5)
    pred = predict_volume(LabelModel(), torch.device("cpu"), image, cfg)
    assert proxy.reads == 1
    if native:
        np.testing.assert_array_equal(pred, labels)


def test_native_evaluation_forwards_dataset_inference_batch(tmp_path):
    write_case(tmp_path, "a", np.zeros((5, 7, 5), dtype=np.uint8))
    dataset = NativeSlabDataset([str(tmp_path)], {**tiles(3, 4, 2), "inference_batch": 5})
    model = RampModel()
    evaluate(model, torch.device("cpu"), DataLoader(dataset))
    assert model.batch_sizes == [5, 5, 2]


@pytest.mark.parametrize("bad", [0, -1, True, 2.0, "8"])
def test_predictor_rejects_invalid_inference_batch(bad):
    image = nib.Nifti1Image(np.zeros((5, 7, 3)), np.eye(4))
    with pytest.raises(ValueError, match="inference_batch must be a positive integer"):
        predict_volume(
            LabelModel(), torch.device("cpu"), image, {**config(), "inference_batch": bad}
        )
