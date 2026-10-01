"""Native sample ordering, padding, and phase transfer survive exact resume."""

import os
import signal

import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dense_unet_3d.dataset.slabs import NativeSlabDataset
from dense_unet_3d.training import cascaded_driver, runtime
from tests.training.test_recovery import Tiny, config, equal, load


def native_setup(directory):
    directory.mkdir(exist_ok=True)
    for i, depth in enumerate((2, 5)):
        image = np.arange(3 * 4 * depth, dtype=np.float32).reshape(3, 4, depth) / 10
        target = (np.arange(image.size).reshape(image.shape) % 3).astype(np.uint8)
        for name, array in ((f"volume-{i}.nii", image), (f"segmentation-{i}.nii", target)):
            path = directory / name
            if not path.exists():
                nib.save(nib.Nifti1Image(array, np.eye(4)), path)
    torch.manual_seed(33)
    dataset = NativeSlabDataset(
        [str(directory)],
        {
            "resize_dims": {"D": 3, "H": 3, "W": 4},
            "clamp_hu": False,
        },
    )
    loader = DataLoader(
        dataset, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(44)
    )
    return Tiny(), loader


@pytest.mark.parametrize("boundary", [("phase_a", 1), ("phase_b", 0), ("phase_b", 1)])
def test_native_exact_resume_with_padding_and_transfer(tmp_path, boundary):
    cfg = config(tmp_path / "whole")
    cfg["training"].update(
        phase_a_epochs=1,
        phase_b_epochs=2,
        phase_a_targets="three_class",
        loss_reduction="valid_voxel_mean",
    )
    model, loader = native_setup(tmp_path / "data")
    cascaded_driver.run_cascaded_training(cfg, model, torch.device("cpu"), loader, loader)
    expected = load(cfg)
    cfg["pathing"]["model_save_dir"] = str(tmp_path / "split")
    model, loader = native_setup(tmp_path / "data")
    with runtime.RunSession(cfg) as session:
        original = session.event

        def interrupt(kind, **fields):
            original(kind, **fields)
            if kind == "checkpoint" and (fields["phase"], fields["epoch"]) == boundary:
                os.kill(os.getpid(), signal.SIGTERM)

        session.event = interrupt
        cascaded_driver.run_cascaded_training(
            cfg, model, torch.device("cpu"), loader, loader, session=session
        )
    model, loader = native_setup(tmp_path / "data")
    cascaded_driver.run_cascaded_training(
        cfg, model, torch.device("cpu"), loader, loader, resume=True
    )
    actual = load(cfg)
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng",
        "history",
        "best",
        "global_step",
        "phase_b_loaded_phase_a_state_dict",
    ):
        equal(expected[key], actual[key])
    assert actual["training_semantics"]["phase_a_targets"] == "three_class"
    assert actual["training_semantics"]["loss_reduction"] == "valid_voxel_mean"
    equal(expected["diagnostics"], actual["diagnostics"])
    assert actual["diagnostics"]["case_ids_available"]
    assert actual["diagnostics"]["tumor_positive_cases_seen"] == 2
    assert actual["diagnostics"]["unique_cases_seen"] == 2


@pytest.mark.parametrize(
    "mutate", ["source", "geometry", "sample_order", "metadata", "best_metadata"]
)
def test_native_resume_rejects_changed_identity_and_metadata(tmp_path, mutate):
    cfg = config(tmp_path)
    cfg["training"].update(
        phase_a_epochs=1,
        phase_b_epochs=1,
        phase_a_targets="three_class",
        loss_reduction="valid_voxel_mean",
    )
    model, loader = native_setup(tmp_path / "data")
    cascaded_driver.run_cascaded_training(cfg, model, torch.device("cpu"), loader)
    if mutate == "source":
        path = tmp_path / "data" / "volume-0.nii"
        image = nib.load(path)
        nib.save(nib.Nifti1Image(image.get_fdata() + 1, image.affine), path)
    elif mutate == "geometry":
        loader.dataset.geometry["height"] += 1
    elif mutate == "sample_order":
        loader.dataset.sample_index.reverse()
    else:
        state = load(cfg)
        snapshot = state if mutate == "metadata" else state["best"]["phase_a"]["checkpoint"]
        snapshot["training_semantics"]["loss_reduction"] = "weighted_mean"
        torch.save(state, tmp_path / "run" / "recovery.pt")
    with pytest.raises(ValueError, match="Incompatible"):
        cascaded_driver.run_cascaded_training(cfg, model, torch.device("cpu"), loader, resume=True)
