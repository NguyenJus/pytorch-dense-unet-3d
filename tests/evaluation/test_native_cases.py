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
            values = [0.51, 0.48, 0.01] if self.calls == 0 else [0.01, 0.98, 0.01]
            self.calls += 1
            return torch.tensor(values).log()[None, :, None, None, None].expand(1, 3, 12, 1, 1)

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
