"""Native sample ordering, padding, and phase transfer survive exact resume."""

import json
import os
import signal

import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dense_unet_3d.dataset.foreground_crops import CropSampler, ForegroundCropDataset
from dense_unet_3d.dataset.slabs import NativeSlabDataset
from dense_unet_3d.training import cascaded_driver, runtime
from tests.training.test_recovery import (
    Tiny,
    config,
    crop_case_dir,
    crop_dataset_config,
    equal,
    load,
)


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


def crop_run_config(path, manifest_cases=2, **dataset_update):
    cfg = config(path)
    cfg["training"].update(
        phase_a_epochs=2,
        phase_b_epochs=2,
        phase_a_targets="three_class",
        loss_reduction="valid_voxel_mean",
        step_unit="training_pass",
    )
    cfg["dataset"] = crop_dataset_config(**dataset_update)
    passes = cfg["dataset"]["passes_per_epoch"]
    cfg["training"].update(phase_a_steps_per_epoch=passes, phase_b_steps_per_epoch=passes)
    manifest = path.parent / f"split-{manifest_cases}.json"
    ids = [f"-{i}" for i in range(manifest_cases)]
    manifest.write_text(json.dumps({"case_ids": {"train": ids, "validation": []}}))
    cfg["experiment"] = {
        "kind": "historical_reference",
        "split_manifest": str(manifest),
        "fidelity": {"step_unit": {"label": "DEVIATION", "evidence": "one step = one pass"}},
    }
    return cfg


def crop_setup(directory, num_workers=2, **dataset_update):
    crop_case_dir(directory)
    dataset = ForegroundCropDataset([str(directory)], crop_dataset_config(**dataset_update))
    generator = torch.Generator().manual_seed(dataset.geometry["sampling_seed"])
    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=CropSampler(len(dataset), generator),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    validation = DataLoader(
        NativeSlabDataset(
            [str(directory)], {"resize_dims": {"D": 4, "H": 8, "W": 8}, "clamp_hu": False}
        ),
        batch_size=1,
    )
    torch.manual_seed(1)
    # No dropout, so selection is comparable across a resume without comparing weights.
    return torch.nn.Conv3d(1, 3, 1), loader, validation


@pytest.fixture
def drawn_crops(monkeypatch):
    """Record each epoch's main-process (index, item_seed) draws."""
    epochs = []
    original = CropSampler.__iter__

    def record(self):
        draws = list(original(self))
        epochs.append(draws)
        return iter(draws)

    monkeypatch.setattr(CropSampler, "__iter__", record)
    return epochs


def stop_at(session, boundary):
    original = session.event

    def interrupt(kind, **fields):
        original(kind, **fields)
        if kind == "checkpoint" and (fields["phase"], fields["epoch"]) == boundary:
            session.request_stop()

    session.event = interrupt


def test_crop_resume_with_workers_replays_crop_parameters(tmp_path, drawn_crops):
    cfg = crop_run_config(tmp_path / "whole")
    model, loader, validation = crop_setup(tmp_path / "data")
    cascaded_driver.run_cascaded_training(cfg, model, torch.device("cpu"), loader, validation)
    expected = load(cfg)
    whole = [[loader.dataset.crop_parameters(*draw) for draw in epoch] for epoch in drawn_crops]
    assert len(whole) == 4 and all(len(epoch) == len(loader.dataset) for epoch in whole)
    drawn_crops.clear()
    cfg["pathing"]["model_save_dir"] = str(tmp_path / "split")
    model, loader, validation = crop_setup(tmp_path / "data")
    with runtime.RunSession(cfg) as session:
        stop_at(session, ("phase_a", 1))
        cascaded_driver.run_cascaded_training(
            cfg, model, torch.device("cpu"), loader, validation, session=session
        )
    model, loader, validation = crop_setup(tmp_path / "data")
    cascaded_driver.run_cascaded_training(
        cfg, model, torch.device("cpu"), loader, validation, resume=True
    )
    actual = load(cfg)
    split = [[loader.dataset.crop_parameters(*draw) for draw in epoch] for epoch in drawn_crops]
    assert split == whole
    # Epoch-exact, not sample-order-bitwise: compare progress, selection and LR, not weights.
    equal(expected["scheduler_state_dict"], actual["scheduler_state_dict"])
    equal(expected["rng"]["loaders"], actual["rng"]["loaders"])
    assert actual["phase"] == "completed"
    assert actual["global_step"] == expected["global_step"] == 4 * len(loader)
    for phase in ("phase_a", "phase_b"):
        assert len(actual["history"][phase]) == 2
        assert actual["best"][phase]["epoch"] == expected["best"][phase]["epoch"]
        assert actual["best"][phase]["score"] == pytest.approx(expected["best"][phase]["score"])


def test_training_pass_runs_one_loader_pass_per_epoch(tmp_path):
    # The loader, not the manifest-derived plan, defines one epoch's updates.
    cfg = crop_run_config(tmp_path / "run", manifest_cases=3)
    model, loader, _ = crop_setup(tmp_path / "data", num_workers=0)
    with runtime.RunSession(cfg) as session:
        events = []
        original = session.event

        def capture(kind, **fields):
            events.append((kind, fields))
            original(kind, **fields)

        session.event = capture
        cascaded_driver.run_cascaded_training(
            cfg, model, torch.device("cpu"), loader, session=session
        )
    assert len(loader) == 2
    assert load(cfg)["global_step"] == 4 * len(loader)
    (workload,) = [fields["phases"] for kind, fields in events if kind == "sample_workload"]
    for phase in ("phase_a", "phase_b"):
        assert workload[phase]["sample_exposures_per_epoch"] == len(loader.dataset)
        assert workload[phase]["sample_exposures_total"] == 2 * len(loader.dataset)


@pytest.mark.parametrize("mutate", ["sampling_seed", "passes_per_epoch"])
def test_crop_resume_refuses_changed_sampling(tmp_path, mutate):
    cfg = crop_run_config(tmp_path / "run")
    model, loader, _ = crop_setup(tmp_path / "data")
    with runtime.RunSession(cfg) as session:
        stop_at(session, ("phase_a", 1))
        cascaded_driver.run_cascaded_training(
            cfg, model, torch.device("cpu"), loader, session=session
        )
    if mutate == "sampling_seed":
        # Same config: the dataset manifest alone carries the change.
        model, loader, _ = crop_setup(tmp_path / "data", sampling_seed=8)
    else:
        cfg = crop_run_config(tmp_path / "run", passes_per_epoch=3)
        model, loader, _ = crop_setup(tmp_path / "data", passes_per_epoch=3)
    with pytest.raises(ValueError, match="Incompatible"):
        cascaded_driver.run_cascaded_training(cfg, model, torch.device("cpu"), loader, resume=True)
