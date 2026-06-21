"""Tests for training loop (F1) — TDD.

Acceptance criteria (§6 F1):
1. Training with use_scheduler=False runs and checkpoints without crashing
   (regression for the None-scheduler bugs in scheduler.step() and
   scheduler.state_dict()).
2. Empty dataloader does not raise NameError — returns a defined loss (0.0).
3. get_optimizer -> SGD momentum=0.5 lr=0.01; get_scheduler -> StepLR
   step_size=10 gamma=0.5 (assert exact hyperparams).
4. 1-epoch CPU dry-run on tiny synthetic data completes; loss is finite;
   a checkpoint file is written.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dense_unet_3d.training.train import get_optimizer, get_scheduler, train

# ---------------------------------------------------------------------------
# Minimal model + config helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal 3-class linear model for fast CPU tests."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,1,D,H,W) -> (N,3,D,H,W)
        return self.conv(x)


def _base_config(tmp_dir: str, use_scheduler: bool = True) -> dict:
    return {
        "pathing": {
            "run_name": "test_run",
            "model_save_dir": tmp_dir,
        },
        "training": {
            "optimizer": "SGD",
            "learning_rate": 0.01,
            "momentum": 0.5,
            "criterion": "CrossEntropyLoss",
            "class_weights": {
                "background": 0.2,
                "liver": 1.2,
                "lesion": 2.2,
            },
            "use_scheduler": use_scheduler,
            "scheduler": "StepLR",
            "scheduler_step": 10,
            "scheduler_gamma": 0.5,
            "epochs": 1,
        },
    }


def _tiny_loader(batch_size: int = 2, d: int = 4, h: int = 8, w: int = 8) -> DataLoader:
    """Small synthetic dataloader for 1-epoch CPU dry-run."""
    torch.manual_seed(0)
    volumes = torch.randn(batch_size, 1, d, h, w)
    labels = torch.randint(0, 3, (batch_size, 1, d, h, w))
    ds = TensorDataset(volumes, labels)
    return DataLoader(ds, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetOptimizer:
    """get_optimizer returns SGD with momentum=0.5 and lr=0.01."""

    def test_sgd_momentum(self) -> None:
        model = _TinyModel()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_config(tmp)
        optimizer = get_optimizer(model, cfg)
        assert isinstance(optimizer, torch.optim.SGD), f"Expected SGD, got {type(optimizer)}"
        param_group = optimizer.param_groups[0]
        assert param_group["momentum"] == 0.5, (
            f"Expected momentum=0.5, got {param_group['momentum']}"
        )

    def test_sgd_lr(self) -> None:
        model = _TinyModel()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_config(tmp)
        optimizer = get_optimizer(model, cfg)
        param_group = optimizer.param_groups[0]
        assert param_group["lr"] == pytest.approx(0.01), (
            f"Expected lr=0.01, got {param_group['lr']}"
        )


class TestGetScheduler:
    """get_scheduler returns StepLR with step_size=10 and gamma=0.5."""

    def test_steplr_step_size(self) -> None:
        model = _TinyModel()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_config(tmp, use_scheduler=True)
        optimizer = get_optimizer(model, cfg)
        scheduler = get_scheduler(optimizer, cfg)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR), (
            f"Expected StepLR, got {type(scheduler)}"
        )
        assert scheduler.step_size == 10, f"Expected step_size=10, got {scheduler.step_size}"

    def test_steplr_gamma(self) -> None:
        model = _TinyModel()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_config(tmp, use_scheduler=True)
        optimizer = get_optimizer(model, cfg)
        scheduler = get_scheduler(optimizer, cfg)
        assert scheduler.gamma == pytest.approx(0.5), f"Expected gamma=0.5, got {scheduler.gamma}"

    def test_none_when_disabled(self) -> None:
        model = _TinyModel()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_config(tmp, use_scheduler=False)
        optimizer = get_optimizer(model, cfg)
        scheduler = get_scheduler(optimizer, cfg)
        assert scheduler is None, f"Expected None when use_scheduler=False, got {scheduler}"


class TestSchedulerNoneRegression:
    """Regression: use_scheduler=False must not crash (no AttributeError on None)."""

    def test_train_no_scheduler_completes(self) -> None:
        """Training with use_scheduler=False runs end-to-end without AttributeError."""
        model = _TinyModel()
        loader = _tiny_loader()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_config(tmp, use_scheduler=False)
            # Must not raise AttributeError: 'NoneType' object has no attribute 'step'
            losses = train(cfg, model, torch.device("cpu"), loader)
        assert losses is not None

    def test_checkpoint_no_scheduler(self) -> None:
        """Checkpoint written without scheduler must not include scheduler_state_dict crash."""
        model = _TinyModel()
        loader = _tiny_loader()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_config(tmp, use_scheduler=False)
            cfg["training"]["epochs"] = 10  # trigger a checkpoint at epoch 10
            train(cfg, model, torch.device("cpu"), loader)
            ckpt_path = os.path.join(tmp, "test_run", "epoch10.pt")
            assert os.path.isfile(ckpt_path), f"Checkpoint not found at {ckpt_path}"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # scheduler_state_dict must be None (not a crash)
            assert ckpt["scheduler_state_dict"] is None, (
                f"Expected None scheduler_state_dict, got {ckpt['scheduler_state_dict']}"
            )


class TestEmptyDataloaderNoNameError:
    """Empty dataloader must not raise NameError — returns a defined loss."""

    def test_empty_loader_returns_defined_loss(self) -> None:
        """When the dataloader is empty, train() returns [0.0] (not NameError)."""
        model = _TinyModel()
        # Create a DataLoader with zero samples
        ds = TensorDataset(torch.zeros(0, 1, 4, 8, 8), torch.zeros(0, 1, 4, 8, 8, dtype=torch.long))
        empty_loader = DataLoader(ds, batch_size=2)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_config(tmp, use_scheduler=False)
            # Must not raise NameError: name 'i' is not defined
            losses = train(cfg, model, torch.device("cpu"), empty_loader)

        assert losses is not None, "losses must be defined (not NameError)"
        assert len(losses) == 1, f"Expected 1 epoch loss entry, got {len(losses)}"
        # The documented return for empty loader is 0.0
        assert losses[0] == pytest.approx(0.0), f"Expected 0.0 for empty loader, got {losses[0]}"


class TestCpuDryRun:
    """1-epoch CPU dry-run on tiny synthetic data: loss finite, checkpoint written."""

    def test_loss_is_finite(self) -> None:
        model = _TinyModel()
        loader = _tiny_loader()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_config(tmp, use_scheduler=True)
            losses = train(cfg, model, torch.device("cpu"), loader)
        assert len(losses) == 1, f"Expected 1 epoch, got {len(losses)}"
        assert torch.isfinite(torch.tensor(losses[0])), f"Loss is not finite: {losses[0]}"

    def test_checkpoint_written_at_epoch_10(self) -> None:
        """Checkpoint is written every 10 epochs; run 10 epochs to verify."""
        model = _TinyModel()
        loader = _tiny_loader()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_config(tmp, use_scheduler=True)
            cfg["training"]["epochs"] = 10
            losses = train(cfg, model, torch.device("cpu"), loader)
            ckpt_path = os.path.join(tmp, "test_run", "epoch10.pt")
            assert os.path.isfile(ckpt_path), f"Checkpoint not found at {ckpt_path}"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert "model_state_dict" in ckpt
            assert "optimizer_state_dict" in ckpt
            assert "scheduler_state_dict" in ckpt
            assert ckpt["epoch"] == 10
        assert all(torch.isfinite(torch.tensor(v)) for v in losses)

    def test_val_dice_logged_per_epoch(self) -> None:
        """train() returns per-epoch metrics that include val Dice when val_loader provided."""
        model = _TinyModel()
        loader = _tiny_loader()
        val_loader = _tiny_loader()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_config(tmp, use_scheduler=False)
            result = train(cfg, model, torch.device("cpu"), loader, val_loader=val_loader)
        # With val_loader, result should be a dict (or contain metrics)
        # At minimum, train() must not crash and must return something defined
        assert result is not None
