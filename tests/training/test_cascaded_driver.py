"""Tests for F2: cascaded 2-phase driver + checkpointing.

Acceptance criteria (§6 F2):
1. With tiny configs (Phase A 2x2, Phase B 2x2) on synthetic data, driver runs
   end-to-end on CPU without error.
2. Phase B PROVABLY reloads Phase A's best weights: assert the loaded
   state_dict matches the saved best checkpoint's state_dict.
3. Produces best + last checkpoints for each phase.
4. Checkpoint round-trips: save then load restores model + optimizer
   (+ scheduler when present) and resumes without error.
5. Phase/step/epoch counts are config-driven (configurable via config dict).

Design notes:
- "10 steps per epoch" is configurable; default = 10 (literal paper reading).
  This is documented in code + decision record (§6 F2).
- Best checkpoint is selected by highest val Dice across epochs.
- Phase A saves: best + last checkpoints.
- Phase B loads Phase A best, then saves: best + last checkpoints.
"""

from __future__ import annotations

import os
import tempfile
from unittest import mock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dense_unet_3d.training import cascaded_driver as cascaded_mod
from dense_unet_3d.training.cascaded_driver import (
    _run_epoch,
    load_checkpoint,
    run_cascaded_training,
    run_phase_a,
    run_phase_b,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal 3-class linear model for fast CPU tests."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _loader(n: int = 4, d: int = 4, h: int = 8, w: int = 8) -> DataLoader:
    torch.manual_seed(42)
    vols = torch.randn(n, 1, d, h, w)
    masks = torch.randint(0, 3, (n, d, h, w))
    return DataLoader(TensorDataset(vols, masks), batch_size=2)


def _base_cfg(tmp_dir: str, *, use_scheduler: bool = True) -> dict:
    """Full config matching the schema expected by cascaded driver."""
    return {
        "pathing": {
            "run_name": "test_cascaded",
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
            # Phase A: 2 epochs x 2 steps (tiny values for fast CPU test)
            "phase_a_epochs": 2,
            "phase_a_steps_per_epoch": 2,
            # Phase B: 2 epochs x 2 steps
            "phase_b_epochs": 2,
            "phase_b_steps_per_epoch": 2,
            # Default steps per epoch documented per the literal paper reading
            "steps_per_epoch": 10,
        },
    }


class TestCriterionBuiltOncePerEpoch:
    """_run_epoch must build the criterion once, not once per mini-batch step."""

    def test_criterion_built_once_over_multi_step_epoch(self) -> None:
        model = _TinyModel()
        loader = _loader(n=6)  # batch_size 2 -> enough for 3 steps
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp, use_scheduler=False)

        real_get_criterion = cascaded_mod.get_criterion
        with mock.patch.object(
            cascaded_mod, "get_criterion", side_effect=real_get_criterion
        ) as spy:
            _run_epoch(
                config=cfg,
                model=model,
                device=torch.device("cpu"),
                loader=loader,
                optimizer=optimizer,
                steps_per_epoch=3,
            )

        assert spy.call_count == 1, (
            f"get_criterion should be built once per epoch, got {spy.call_count} "
            "calls (rebuilt per step?)"
        )


# ---------------------------------------------------------------------------
# Test: save_checkpoint / load_checkpoint round-trip (with scheduler)
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    """save_checkpoint + load_checkpoint must restore all state exactly."""

    def test_round_trip_with_scheduler(self) -> None:
        """Save then load restores model + optimizer + scheduler state."""
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # Advance state so values differ from defaults
        dummy = model(torch.randn(1, 1, 4, 8, 8))
        dummy.sum().backward()
        optimizer.step()
        scheduler.step()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            save_checkpoint(
                path=path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=5,
                metrics={"val_dice": 0.85},
            )

            # Load into fresh instances
            model2 = _TinyModel()
            optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01, momentum=0.5)
            scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=10, gamma=0.5)

            meta = load_checkpoint(
                path=path,
                model=model2,
                optimizer=optimizer2,
                scheduler=scheduler2,
            )

        # Model weights must match
        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), model2.state_dict().items(), strict=True
        ):
            assert k1 == k2
            assert torch.allclose(v1, v2), f"Weight mismatch at {k1}"

        # Optimizer lr must be restored
        assert optimizer2.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])

        # Scheduler state must match
        assert scheduler2.last_epoch == scheduler.last_epoch

        # Metadata preserved
        assert meta["epoch"] == 5
        assert meta["metrics"]["val_dice"] == pytest.approx(0.85)

    def test_round_trip_without_scheduler(self) -> None:
        """save/load with scheduler=None stores None and loads cleanly."""
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt_no_sched.pt")
            save_checkpoint(
                path=path,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=3,
                metrics={"val_dice": 0.5},
            )
            raw = torch.load(path, map_location="cpu", weights_only=False)
            assert raw["scheduler_state_dict"] is None, (
                "scheduler_state_dict must be None when no scheduler"
            )

            model2 = _TinyModel()
            optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01, momentum=0.5)
            meta = load_checkpoint(path=path, model=model2, optimizer=optimizer2, scheduler=None)

        assert meta["epoch"] == 3

    def test_checkpoint_contains_required_keys(self) -> None:
        """Checkpoint dict must have model/optimizer/scheduler/epoch/metrics keys."""
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt_keys.pt")
            save_checkpoint(
                path=path,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=1,
                metrics={"val_dice": 0.0},
            )
            raw = torch.load(path, map_location="cpu", weights_only=False)

        required = {
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "epoch",
            "metrics",
        }
        assert required.issubset(raw.keys()), f"Missing keys: {required - set(raw.keys())}"


# ---------------------------------------------------------------------------
# Test: Phase A end-to-end + best/last checkpoints
# ---------------------------------------------------------------------------


class TestPhaseA:
    """run_phase_a produces best + last checkpoints."""

    def test_phase_a_produces_best_checkpoint(self) -> None:
        model = _TinyModel()
        loader = _loader()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp)
            run_phase_a(cfg, model, torch.device("cpu"), loader, val_loader=loader)
            ckpt_dir = os.path.join(tmp, "test_cascaded", "phase_a")
            best_path = os.path.join(ckpt_dir, "best.pt")
            assert os.path.isfile(best_path), f"Phase A best checkpoint not found: {best_path}"

    def test_phase_a_produces_last_checkpoint(self) -> None:
        model = _TinyModel()
        loader = _loader()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp)
            run_phase_a(cfg, model, torch.device("cpu"), loader, val_loader=loader)
            ckpt_dir = os.path.join(tmp, "test_cascaded", "phase_a")
            last_path = os.path.join(ckpt_dir, "last.pt")
            assert os.path.isfile(last_path), f"Phase A last checkpoint not found: {last_path}"

    def test_phase_a_best_has_required_keys(self) -> None:
        model = _TinyModel()
        loader = _loader()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp)
            run_phase_a(cfg, model, torch.device("cpu"), loader, val_loader=loader)
            ckpt_dir = os.path.join(tmp, "test_cascaded", "phase_a")
            raw = torch.load(
                os.path.join(ckpt_dir, "best.pt"), map_location="cpu", weights_only=False
            )

        required = {
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "epoch",
            "metrics",
        }
        assert required.issubset(raw.keys())

    def test_phase_a_config_driven_epochs(self) -> None:
        """Phase A epoch count comes from config['training']['phase_a_epochs']."""
        model = _TinyModel()
        loader = _loader()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp)
            cfg["training"]["phase_a_epochs"] = 3  # override
            result = run_phase_a(cfg, model, torch.device("cpu"), loader, val_loader=loader)
        # result is a dict with 'epoch_losses' of length phase_a_epochs
        assert len(result["epoch_losses"]) == 3, (
            f"Expected 3 epoch losses, got {len(result['epoch_losses'])}"
        )

    def test_phase_a_without_scheduler(self) -> None:
        """Phase A with use_scheduler=False completes without AttributeError."""
        model = _TinyModel()
        loader = _loader()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp, use_scheduler=False)
            run_phase_a(cfg, model, torch.device("cpu"), loader, val_loader=loader)
            ckpt_dir = os.path.join(tmp, "test_cascaded", "phase_a")
            assert os.path.isfile(os.path.join(ckpt_dir, "best.pt"))


# ---------------------------------------------------------------------------
# Test: Phase B reloads Phase A best (THE KEY TEST)
# ---------------------------------------------------------------------------


class TestPhaseBReloadsPhaseABest:
    """Phase B PROVABLY reloads Phase A best weights."""

    def test_phase_b_reloads_phase_a_best_weights(self) -> None:
        """
        The loaded model state_dict at the start of Phase B must EXACTLY
        match the Phase A best checkpoint's model_state_dict.

        This proves the reload is not just an optimizer reset but a real
        weight transfer from Phase A's best checkpoint.
        """
        model = _TinyModel()
        loader = _loader()

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp)

            # Run Phase A
            run_phase_a(cfg, model, torch.device("cpu"), loader, val_loader=loader)

            # Read the saved best state dict from Phase A
            phase_a_best_path = os.path.join(tmp, "test_cascaded", "phase_a", "best.pt")
            phase_a_ckpt = torch.load(phase_a_best_path, map_location="cpu", weights_only=False)
            phase_a_best_state = phase_a_ckpt["model_state_dict"]

            # Run Phase B with a NEW fresh model — it must load Phase A best
            model_b = _TinyModel()
            result = run_phase_b(
                cfg,
                model_b,
                torch.device("cpu"),
                loader,
                val_loader=loader,
                phase_a_best_path=phase_a_best_path,
            )

            # The loaded state dict at Phase B start must match Phase A best
            loaded_state = result["loaded_phase_a_state_dict"]
            for key in phase_a_best_state:
                assert key in loaded_state, f"Key {key!r} missing from loaded state"
                assert torch.allclose(phase_a_best_state[key], loaded_state[key]), (
                    f"Weight mismatch at {key!r}: "
                    f"phase_a={phase_a_best_state[key]} loaded={loaded_state[key]}"
                )

    def test_phase_b_produces_best_and_last(self) -> None:
        model = _TinyModel()
        loader = _loader()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp)
            run_phase_a(cfg, model, torch.device("cpu"), loader, val_loader=loader)
            phase_a_best_path = os.path.join(tmp, "test_cascaded", "phase_a", "best.pt")

            model_b = _TinyModel()
            run_phase_b(
                cfg,
                model_b,
                torch.device("cpu"),
                loader,
                val_loader=loader,
                phase_a_best_path=phase_a_best_path,
            )

            ckpt_dir = os.path.join(tmp, "test_cascaded", "phase_b")
            assert os.path.isfile(os.path.join(ckpt_dir, "best.pt")), "Phase B best missing"
            assert os.path.isfile(os.path.join(ckpt_dir, "last.pt")), "Phase B last missing"

    def test_phase_b_config_driven_epochs(self) -> None:
        model = _TinyModel()
        loader = _loader()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp)
            run_phase_a(cfg, model, torch.device("cpu"), loader, val_loader=loader)
            phase_a_best_path = os.path.join(tmp, "test_cascaded", "phase_a", "best.pt")

            cfg["training"]["phase_b_epochs"] = 3
            model_b = _TinyModel()
            result = run_phase_b(
                cfg,
                model_b,
                torch.device("cpu"),
                loader,
                val_loader=loader,
                phase_a_best_path=phase_a_best_path,
            )
        assert len(result["epoch_losses"]) == 3, (
            f"Expected 3 epoch losses for phase B, got {len(result['epoch_losses'])}"
        )


# ---------------------------------------------------------------------------
# Test: run_cascaded_training (full driver)
# ---------------------------------------------------------------------------


class TestRunCascadedTraining:
    """Full two-phase driver: Phase A -> Phase B end-to-end."""

    def test_cascaded_training_end_to_end(self) -> None:
        """
        Full cascaded training (Phase A + Phase B) with tiny config
        completes on CPU without error and produces all checkpoints.
        """
        model = _TinyModel()
        loader = _loader()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp)
            result = run_cascaded_training(
                cfg,
                model,
                torch.device("cpu"),
                loader,
                val_loader=loader,
            )

            phase_a_dir = os.path.join(tmp, "test_cascaded", "phase_a")
            phase_b_dir = os.path.join(tmp, "test_cascaded", "phase_b")

            assert os.path.isfile(os.path.join(phase_a_dir, "best.pt")), "Phase A best missing"
            assert os.path.isfile(os.path.join(phase_a_dir, "last.pt")), "Phase A last missing"
            assert os.path.isfile(os.path.join(phase_b_dir, "best.pt")), "Phase B best missing"
            assert os.path.isfile(os.path.join(phase_b_dir, "last.pt")), "Phase B last missing"

        assert result is not None

    def test_cascaded_training_phase_b_weights_from_phase_a_best(self) -> None:
        """
        Full driver: Phase B must start from Phase A best weights.
        Verify via run_cascaded_training's returned metadata.
        """
        model = _TinyModel()
        loader = _loader()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp)
            result = run_cascaded_training(
                cfg,
                model,
                torch.device("cpu"),
                loader,
                val_loader=loader,
            )

            # Load Phase A best from disk
            phase_a_best = torch.load(
                os.path.join(tmp, "test_cascaded", "phase_a", "best.pt"),
                map_location="cpu",
                weights_only=False,
            )

        # result must carry proof that phase_b started from phase_a best
        loaded_state = result["phase_b_loaded_phase_a_state_dict"]
        for key, val in phase_a_best["model_state_dict"].items():
            assert torch.allclose(loaded_state[key], val), (
                f"Phase B did not load Phase A best at key {key!r}"
            )

    def test_cascaded_all_losses_finite(self) -> None:
        model = _TinyModel()
        loader = _loader()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp)
            result = run_cascaded_training(
                cfg, model, torch.device("cpu"), loader, val_loader=loader
            )
        all_losses = result["phase_a"]["epoch_losses"] + result["phase_b"]["epoch_losses"]
        for v in all_losses:
            assert torch.isfinite(torch.tensor(v)), f"Non-finite loss encountered: {v}"

    def test_steps_per_epoch_is_configurable(self) -> None:
        """steps_per_epoch comes from config, not hardcoded."""
        model = _TinyModel()
        loader = _loader()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp)
            cfg["training"]["phase_a_steps_per_epoch"] = 3
            cfg["training"]["phase_b_steps_per_epoch"] = 3
            # Must not error — just runs with 3 steps
            result = run_cascaded_training(
                cfg, model, torch.device("cpu"), loader, val_loader=loader
            )
        assert result is not None


# ---------------------------------------------------------------------------
# Test: Phase B balanced selection (Issue #5)
# ---------------------------------------------------------------------------


class TestPhaseCheckpointSelection:
    """Phase-specific checkpoint selection and strict > tie-break (Issue #5)."""

    def test_phase_b_selects_balanced_over_tumor_blind(self) -> None:
        """Phase B must select on nanmean(liver_per_case, tumor_per_case).

        Epoch 1: liver=0.9, tumor=0.05  -> nanmean=0.475  (tumor-blind)
        Epoch 2: liver=0.7, tumor=0.65  -> nanmean=0.675  (balanced) <- should win
        """
        model = _TinyModel()
        loader = _loader()

        epoch_metrics = [
            {
                "liver_per_case": 0.9,
                "liver_global": 0.9,
                "tumor_per_case": 0.05,
                "tumor_global": 0.05,
            },
            {
                "liver_per_case": 0.7,
                "liver_global": 0.7,
                "tumor_per_case": 0.65,
                "tumor_global": 0.65,
            },
        ]
        call_count = 0

        def _fake_evaluate(model, device, loader):
            nonlocal call_count
            idx = call_count % len(epoch_metrics)
            call_count += 1
            return epoch_metrics[idx]

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp)
            cfg["training"]["phase_b_epochs"] = 2
            cfg["training"]["phase_b_steps_per_epoch"] = 2

            # Run Phase A first to produce the best.pt checkpoint Phase B needs
            run_phase_a(cfg, model, torch.device("cpu"), loader, val_loader=None)
            phase_a_best_path = os.path.join(tmp, "test_cascaded", "phase_a", "best.pt")

            with mock.patch(
                "dense_unet_3d.evaluation.evaluate.evaluate", side_effect=_fake_evaluate
            ):
                model_b = _TinyModel()
                result = run_phase_b(
                    cfg,
                    model_b,
                    torch.device("cpu"),
                    loader,
                    val_loader=loader,
                    phase_a_best_path=phase_a_best_path,
                )

            # The balanced epoch (epoch 2) has the higher combined nanmean -> must be selected
            assert result["best_epoch"] == 2, (
                f"Expected best_epoch=2 (balanced), got {result['best_epoch']}"
            )

            # Verify best.pt on disk also records epoch 2
            best_ckpt = torch.load(
                os.path.join(tmp, "test_cascaded", "phase_b", "best.pt"),
                map_location="cpu",
                weights_only=False,
            )
            assert best_ckpt["epoch"] == 2, (
                f"best.pt should record epoch=2, got {best_ckpt['epoch']}"
            )

    def test_strict_gt_tie_break_retains_earlier_epoch(self) -> None:
        """Equal selection score -> earlier epoch is retained (>  not >=).

        Both epochs return the same liver+tumor nanmean. With >= the second
        epoch would overwrite; with > the first epoch must be kept.
        Tested on Phase B (and Phase A variant).
        """
        model = _TinyModel()
        loader = _loader()

        same_metrics = {
            "liver_per_case": 0.75,
            "liver_global": 0.75,
            "tumor_per_case": 0.55,
            "tumor_global": 0.55,
        }

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp)
            cfg["training"]["phase_b_epochs"] = 2
            cfg["training"]["phase_b_steps_per_epoch"] = 2

            run_phase_a(cfg, model, torch.device("cpu"), loader, val_loader=None)
            phase_a_best_path = os.path.join(tmp, "test_cascaded", "phase_a", "best.pt")

            with mock.patch(
                "dense_unet_3d.evaluation.evaluate.evaluate", return_value=same_metrics
            ):
                model_b = _TinyModel()
                result = run_phase_b(
                    cfg,
                    model_b,
                    torch.device("cpu"),
                    loader,
                    val_loader=loader,
                    phase_a_best_path=phase_a_best_path,
                )

            # Strict > means equal score must NOT overwrite -> best_epoch stays at 1
            assert result["best_epoch"] == 1, (
                f"Tie should keep epoch 1 (strict >), got best_epoch={result['best_epoch']}"
            )
            best_ckpt = torch.load(
                os.path.join(tmp, "test_cascaded", "phase_b", "best.pt"),
                map_location="cpu",
                weights_only=False,
            )
            assert best_ckpt["epoch"] == 1, (
                f"best.pt should record epoch=1 on tie, got {best_ckpt['epoch']}"
            )

    def test_strict_gt_tie_break_phase_a(self) -> None:
        """Phase A also retains earlier epoch on tie (strict > not >=)."""
        model = _TinyModel()
        loader = _loader()

        same_metrics = {
            "liver_per_case": 0.8,
            "liver_global": 0.8,
            "tumor_per_case": 0.5,
            "tumor_global": 0.5,
        }

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp)
            cfg["training"]["phase_a_epochs"] = 2
            cfg["training"]["phase_a_steps_per_epoch"] = 2

            with mock.patch(
                "dense_unet_3d.evaluation.evaluate.evaluate", return_value=same_metrics
            ):
                result = run_phase_a(cfg, model, torch.device("cpu"), loader, val_loader=loader)

            assert result["best_epoch"] == 1, (
                f"Phase A tie should keep epoch 1 (strict >), got best_epoch={result['best_epoch']}"
            )

    def test_phase_b_nanmean_falls_back_on_nan_component(self) -> None:
        """When tumor_per_case is NaN, nanmean falls back to liver_per_case.

        No RuntimeWarning should propagate. The NaN component is silently ignored,
        and selection is driven by the liver component alone.
        """
        import warnings

        model = _TinyModel()
        loader = _loader()

        # Epoch 1: liver=0.6, tumor=nan -> nanmean=0.6
        # Epoch 2: liver=0.5, tumor=0.5 -> nanmean=0.5
        # Epoch 1 should win despite having NaN tumor
        epoch_metrics = [
            {
                "liver_per_case": 0.6,
                "liver_global": 0.6,
                "tumor_per_case": float("nan"),
                "tumor_global": float("nan"),
            },
            {
                "liver_per_case": 0.5,
                "liver_global": 0.5,
                "tumor_per_case": 0.5,
                "tumor_global": 0.5,
            },
        ]
        call_count = 0

        def _fake_evaluate(model, device, loader):
            nonlocal call_count
            idx = call_count % len(epoch_metrics)
            call_count += 1
            return epoch_metrics[idx]

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _base_cfg(tmp)
            cfg["training"]["phase_b_epochs"] = 2
            cfg["training"]["phase_b_steps_per_epoch"] = 2

            run_phase_a(cfg, model, torch.device("cpu"), loader, val_loader=None)
            phase_a_best_path = os.path.join(tmp, "test_cascaded", "phase_a", "best.pt")

            with warnings.catch_warnings(record=True) as warning_list:
                warnings.simplefilter("always")
                with mock.patch(
                    "dense_unet_3d.evaluation.evaluate.evaluate", side_effect=_fake_evaluate
                ):
                    model_b = _TinyModel()
                    result = run_phase_b(
                        cfg,
                        model_b,
                        torch.device("cpu"),
                        loader,
                        val_loader=loader,
                        phase_a_best_path=phase_a_best_path,
                    )

            # Epoch 1 (liver=0.6, nanmean=0.6) > Epoch 2 (nanmean=0.5) -> epoch 1 wins
            assert result["best_epoch"] == 1, (
                f"NaN tumor fallback: expected epoch 1 to win, got {result['best_epoch']}"
            )

            # No RuntimeWarning about all-NaN slice should have leaked
            runtime_warnings = [w for w in warning_list if issubclass(w.category, RuntimeWarning)]
            assert not runtime_warnings, (
                f"Unexpected RuntimeWarning(s) leaked: {[str(w.message) for w in runtime_warnings]}"
            )
