"""Deterministic CPU continuation and durable failure injection contracts."""

from __future__ import annotations

import json
import os
import random
import signal

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dense_unet_3d.training import cascaded_driver, recovery, runtime


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv3d(1, 3, 1), nn.Dropout3d(0.2))

    def forward(self, x):
        return self.layers(x)


def config(path, scheduler=True):
    return {
        "pathing": {"model_save_dir": str(path), "run_name": "run"},
        "training": {
            "phase_a_epochs": 3,
            "phase_b_epochs": 3,
            "phase_a_steps_per_epoch": 2,
            "phase_b_steps_per_epoch": 2,
            "optimizer": "SGD",
            "learning_rate": 0.01,
            "momentum": 0.5,
            "use_scheduler": scheduler,
            "scheduler": "StepLR",
            "scheduler_step": 1,
            "scheduler_gamma": 0.8,
            "criterion": "CrossEntropyLoss",
        },
        "runtime": {"budget_seconds": 1000, "max_retries": 2},
    }


def setup():
    random.seed(11)
    np.random.seed(22)
    torch.manual_seed(33)
    data = TensorDataset(torch.randn(5, 1, 2, 2, 2), torch.randint(0, 3, (5, 2, 2, 2)))
    loader = DataLoader(
        data, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(44)
    )
    return Tiny(), loader


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal(a, b)
    else:
        assert left == right


def load(cfg):
    return torch.load(
        os.path.join(cfg["pathing"]["model_save_dir"], "run", "recovery.pt"), weights_only=False
    )


@pytest.mark.parametrize(
    "boundary", [("phase_a", 1), ("phase_a", 3), ("phase_b", 0), ("phase_b", 1)]
)
@pytest.mark.parametrize("scheduler", [True, False])
def test_exact_resume(tmp_path, boundary, scheduler):
    cfg = config(tmp_path / "whole", scheduler)
    model, loader = setup()
    cascaded_driver.run_cascaded_training(cfg, model, torch.device("cpu"), loader)
    expected = load(cfg)
    cfg = config(tmp_path / "split", scheduler)
    model, loader = setup()
    with runtime.RunSession(cfg) as session:
        original = session.event

        def interrupt(kind, **fields):
            original(kind, **fields)
            if kind == "checkpoint" and (fields["phase"], fields["epoch"]) == boundary:
                os.kill(os.getpid(), signal.SIGTERM)

        session.event = interrupt
        result = cascaded_driver.run_cascaded_training(
            cfg, model, torch.device("cpu"), loader, session=session
        )
        assert result["terminal_reason"] == "user stopped"
    model, loader = setup()
    # Restoring must override all random draws and the new model initialization.
    torch.rand(13)
    random.random()
    np.random.rand(9)
    cascaded_driver.run_cascaded_training(cfg, model, torch.device("cpu"), loader, resume=True)
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
    assert actual["phase"] == "completed"
    assert actual["global_step"] == 12
    assert actual["best"]["phase_a"]["epoch"] == 1  # plateau recovery is still every epoch


def test_atomic_write_failure_preserves_latest_and_previous(tmp_path, monkeypatch):
    path = tmp_path / "recovery.pt"
    runtime.atomic_checkpoint(path, {"epoch": 1}, retain_previous=True)
    runtime.atomic_checkpoint(path, {"epoch": 2}, retain_previous=True)

    def broken(_value, stream):
        stream.write(b"incomplete")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", broken)
    with pytest.raises(OSError, match="disk full"):
        runtime.atomic_checkpoint(path, {"epoch": 3}, retain_previous=True)
    assert torch.load(path, weights_only=False)["epoch"] == 2
    assert torch.load(tmp_path / "recovery.previous.pt", weights_only=False)["epoch"] == 1
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_duplicate_owner_and_append_attempts(tmp_path):
    cfg = config(tmp_path)
    with runtime.RunSession(cfg) as session:
        with pytest.raises(RuntimeError, match="active owner"):
            with runtime.RunSession(cfg):
                pass
        assert runtime.read_status(session.run_dir)["ownership"] == "live"
        session.finish("user stopped")
    with runtime.RunSession(cfg, resume=True) as session:
        assert len(session.state["attempts"]) == 2
        assert session.state["attempts"][0]["terminal_reason"] == "user stopped"
    records = [
        json.loads(line) for line in (session.run_dir / "events.jsonl").read_text().splitlines()
    ]
    assert len({r["attempt_id"] for r in records}) == 2


def test_budget_persists_and_lost_attempt_charges_downtime(tmp_path):
    cfg = config(tmp_path)
    with runtime.RunSession(cfg) as session:
        session.finish("user stopped")
    path = session.run_dir / "runtime.json"
    state = json.loads(path.read_text())
    state.update(terminal_reason=None, started_at=state["started_at"] - 2000)
    runtime.atomic_json(path, state)
    with runtime.RunSession(cfg, resume=True, recover=True) as resumed:
        assert resumed.stop_reason() == "budget exhausted"
        assert resumed.cumulative_seconds >= 2000
        assert resumed.state["attempts"][0]["terminal_reason"] == "unknown"
        assert resumed.state["retries_used"] == 1
    with pytest.raises(ValueError, match="cannot be changed"):
        with runtime.RunSession(cfg, resume=True, budget_seconds=3000):
            pass


def test_persistent_retry_exhaustion(tmp_path):
    cfg = config(tmp_path)
    cfg["runtime"]["max_retries"] = 1
    with pytest.raises(RuntimeError):
        with runtime.RunSession(cfg):
            raise RuntimeError("failed")
    with pytest.raises(ValueError, match="explicit --recover"):
        with runtime.RunSession(cfg, resume=True):
            pass
    with pytest.raises(RuntimeError):
        with runtime.RunSession(cfg, resume=True, recover=True):
            raise RuntimeError("failed again")
    with pytest.raises(ValueError, match="allowance exhausted"):
        with runtime.RunSession(cfg, resume=True, recover=True, max_retries=10):
            pass


@pytest.mark.parametrize("mutation", ["data", "config", "legacy", "corrupt"])
def test_incompatible_and_unreadable_recovery_refused(tmp_path, mutation):
    cfg = config(tmp_path)
    model, loader = setup()
    with runtime.RunSession(cfg) as session:
        session.request_stop()
        cascaded_driver.run_cascaded_training(
            cfg, model, torch.device("cpu"), loader, session=session
        )
    if mutation == "data":
        loader.dataset.tensors[0].add_(1)
    elif mutation == "config":
        cfg["training"]["learning_rate"] = 0.02
    elif mutation == "legacy":
        torch.save({"epoch": 1, "model_state_dict": model.state_dict()}, session.checkpoint_path)
    else:
        session.checkpoint_path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="Incompatible|schema|unreadable"):
        cascaded_driver.run_cascaded_training(cfg, model, torch.device("cpu"), loader, resume=True)


@pytest.mark.parametrize("when", ["training", "validation", "checkpoint"])
def test_signals_finish_consistent_boundary(tmp_path, monkeypatch, when):
    cfg = config(tmp_path)
    model, loader = setup()
    monkeypatch.setattr(
        "dense_unet_3d.evaluation.evaluate.evaluate",
        lambda *args: {"liver_per_case": 0.5, "tumor_per_case": float("nan")},
    )
    with runtime.RunSession(cfg) as session:
        original_event = session.event

        def event(kind, **fields):
            original_event(kind, **fields)
            if kind == when and fields.get("epoch") == 1:
                os.kill(os.getpid(), signal.SIGINT)

        session.event = event
        if when == "checkpoint":
            original_write = recovery.atomic_checkpoint

            def write(path, state, **kwargs):
                if path == session.checkpoint_path and state.get("epoch") == 1:
                    os.kill(os.getpid(), signal.SIGTERM)
                original_write(path, state, **kwargs)

            monkeypatch.setattr(recovery, "atomic_checkpoint", write)
        result = cascaded_driver.run_cascaded_training(
            cfg, model, torch.device("cpu"), loader, val_loader=loader, session=session
        )
        assert result["terminal_reason"] == "user stopped"
        assert load(cfg)["epoch"] == 1
        assert load(cfg)["global_step"] == 2


def test_nonfinite_gradient_keeps_initial_recovery(tmp_path):
    cfg = config(tmp_path)
    model, loader = setup()
    next(model.parameters()).register_hook(lambda grad: grad * float("nan"))
    with pytest.raises(FloatingPointError, match="gradients"):
        cascaded_driver.run_cascaded_training(cfg, model, torch.device("cpu"), loader)
    assert load(cfg)["global_step"] == 0
    status = runtime.read_status(tmp_path / "run")
    assert status["terminal_reason"] == "failed"


@pytest.mark.parametrize("transition_phase", ["phase_a", "phase_b"])
def test_failed_transition_commit_replays_no_updates(tmp_path, monkeypatch, transition_phase):
    cfg = config(tmp_path / "whole")
    model, loader = setup()
    cascaded_driver.run_cascaded_training(cfg, model, torch.device("cpu"), loader)
    expected = load(cfg)
    cfg = config(tmp_path / "split")
    model, loader = setup()
    original = recovery.atomic_checkpoint

    def fail_transition(path, state, **kwargs):
        target = (
            state.get("phase") == "phase_b" and state.get("epoch") == 0
            if transition_phase == "phase_a"
            else state.get("phase") == "completed"
        )
        if str(path).endswith("recovery.pt") and target:
            raise OSError("injected transition failure")
        original(path, state, **kwargs)

    monkeypatch.setattr(recovery, "atomic_checkpoint", fail_transition)
    with pytest.raises(OSError, match="transition"):
        cascaded_driver.run_cascaded_training(cfg, model, torch.device("cpu"), loader)
    monkeypatch.setattr(recovery, "atomic_checkpoint", original)
    model, loader = setup()
    with runtime.RunSession(cfg, resume=True, recover=True) as session:
        cascaded_driver.run_cascaded_training(
            cfg, model, torch.device("cpu"), loader, session=session
        )
    actual = load(cfg)
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng",
        "global_step",
        "best",
    ):
        equal(expected[key], actual[key])


def test_corrupt_latest_fallback_preserves_evidence(tmp_path):
    cfg = config(tmp_path)
    model, loader = setup()
    cascaded_driver.run_cascaded_training(cfg, model, torch.device("cpu"), loader)
    path = tmp_path / "run" / "recovery.pt"
    path.write_bytes(b"corrupt")
    with runtime.RunSession(cfg, resume=True, recover=True) as session:
        cascaded_driver.run_cascaded_training(
            cfg, model, torch.device("cpu"), loader, session=session
        )
        assert session.state["retries_used"] == 1
    assert load(cfg)["phase"] == "completed"
    assert list(path.parent.glob("recovery.corrupt.*.pt"))[0].read_bytes() == b"corrupt"


def test_checkpoint_latency_counted_and_no_restart_after_budget(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg["runtime"]["budget_seconds"] = 5
    clock = [0.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    model, loader = setup()
    original = recovery.atomic_checkpoint

    def slow_write(*args, **kwargs):
        original(*args, **kwargs)
        clock[0] += 6

    monkeypatch.setattr(recovery, "atomic_checkpoint", slow_write)
    result = cascaded_driver.run_cascaded_training(cfg, model, torch.device("cpu"), loader)
    assert result["terminal_reason"] == "budget exhausted"
    assert result["global_step"] == 0
    with runtime.RunSession(cfg, resume=True) as session:
        assert session.stop_reason() == "budget exhausted"
        assert session.cumulative_seconds == 6


def test_storage_threshold_and_background_failure_are_failures(tmp_path):
    cfg = config(tmp_path)
    cfg["runtime"]["min_free_bytes"] = 10**30
    model, loader = setup()
    with pytest.raises(OSError, match="Free storage"):
        cascaded_driver.run_cascaded_training(cfg, model, torch.device("cpu"), loader)
    assert runtime.read_status(tmp_path / "run")["terminal_reason"] == "failed"
    cfg = config(tmp_path / "other")
    with pytest.raises(RuntimeError, match="monitoring failed"):
        with runtime.RunSession(cfg) as session:
            session._background_error = OSError("disk full")
            session.stop_reason()
    assert runtime.read_status(tmp_path / "other" / "run")["terminal_reason"] == "failed"


@pytest.mark.parametrize("sharing", ["loader_sampler", "across_phases"])
def test_changed_generator_alias_topology_refused(tmp_path, sharing):
    cfg = config(tmp_path)
    model, loader = setup()
    phase_b_loader = DataLoader(
        loader.dataset, batch_size=2, shuffle=True, generator=loader.generator
    )
    with runtime.RunSession(cfg) as session:
        original = session.event

        def interrupt(kind, **fields):
            original(kind, **fields)
            if (
                kind == "checkpoint"
                and fields.get("phase") == "phase_a"
                and fields.get("epoch") == 1
            ):
                session.request_stop()

        session.event = interrupt
        cascaded_driver.run_cascaded_training(
            cfg,
            model,
            torch.device("cpu"),
            loader,
            phase_b_train_loader=phase_b_loader,
            session=session,
        )
    model, loader = setup()
    phase_b_loader = DataLoader(
        loader.dataset, batch_size=2, shuffle=True, generator=loader.generator
    )
    if sharing == "loader_sampler":
        loader.sampler.generator = torch.Generator().manual_seed(44)
    else:
        # Preserve each loader/sampler pair's aliases while changing their
        # sharing across phases. Individual loader fingerprints are identical.
        independent = torch.Generator().manual_seed(44)
        phase_b_loader.generator = independent
        phase_b_loader.sampler.generator = independent
    with pytest.raises(ValueError, match="Incompatible"):
        cascaded_driver.run_cascaded_training(
            cfg,
            model,
            torch.device("cpu"),
            loader,
            phase_b_train_loader=phase_b_loader,
            resume=True,
        )


@pytest.mark.parametrize("pipeline", ["transform", "mask_transform", "paired_transform"])
def test_changed_lits_preprocessing_refused(tmp_path, pipeline):
    import nibabel as nib
    from torchvision.transforms import Compose

    from dense_unet_3d.dataset.LITSDataset import LITSDataset
    from dense_unet_3d.dataset.transforms.ClampValues import ClampValues
    from dense_unet_3d.dataset.transforms.RandomHorizontalFlip import RandomHorizontalFlip
    from dense_unet_3d.dataset.transforms.ReshapeTensor import ReshapeTensor
    from dense_unet_3d.dataset.transforms.Resize import Resize

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name in ("volume0.nii", "segmentation0.nii"):
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), data_dir / name)
    dataset = LITSDataset(
        [str(data_dir)],
        transform=Compose([ReshapeTensor(), ClampValues((-100, 100))]),
        mask_transform=Compose([ReshapeTensor(), Resize((2, 2, 2), mode="nearest")]),
        paired_transform=Compose([RandomHorizontalFlip(0.5)]),
    )
    cfg = config(tmp_path)
    model = Tiny()
    loader = DataLoader(dataset, batch_size=1)
    with runtime.RunSession(cfg) as session:
        session.request_stop()
        cascaded_driver.run_cascaded_training(
            cfg, model, torch.device("cpu"), loader, session=session
        )
    if pipeline == "transform":
        dataset.transform.transforms.reverse()  # order is part of preprocessing
    elif pipeline == "mask_transform":
        dataset.mask_transform.transforms[-1].mode = "trilinear"
    else:
        dataset.paired_transform.transforms[0].p = 1.0
    with pytest.raises(ValueError, match="Incompatible"):
        cascaded_driver.run_cascaded_training(cfg, model, torch.device("cpu"), loader, resume=True)


def test_custom_preprocessing_local_rng_refused(tmp_path):
    from torchvision.transforms import Compose

    from dense_unet_3d.dataset.LITSDataset import LITSDataset

    class CustomFlip:
        def __init__(self):
            self.rng = np.random.default_rng(42)

        def __call__(self, tensors):
            return tuple(t.flip(-1) for t in tensors) if self.rng.random() < 0.5 else tensors

    dataset = LITSDataset([str(tmp_path)], paired_transform=Compose([CustomFlip()]))
    with pytest.raises(ValueError, match="custom callables/state"):
        recovery._dataset_identity(dataset)


@pytest.mark.parametrize("kind", ["tensor", "subset"])
def test_custom_dataset_subclasses_refused(kind):
    from torch.utils.data import Subset

    class CustomTensorDataset(TensorDataset):
        def __getitem__(self, index):
            return tuple(value + torch.rand_like(value) for value in super().__getitem__(index))

    class CustomSubset(Subset):
        def __getitem__(self, index):
            return tuple(value + torch.rand_like(value) for value in super().__getitem__(index))

        def __getitems__(self, indices):
            return [self[index] for index in indices]

    base = TensorDataset(torch.ones(2, 1))
    dataset = CustomTensorDataset(*base.tensors) if kind == "tensor" else CustomSubset(base, [0])
    with pytest.raises(ValueError, match="Exact resume supports only"):
        recovery._dataset_identity(dataset)
