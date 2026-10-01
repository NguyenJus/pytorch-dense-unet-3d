"""Expose the effective LR horizon without changing the reproduction schedule."""

import json

import pytest
import torch

from dense_unet_3d.training.cascaded_driver import run_cascaded_training
from dense_unet_3d.training.runtime import describe_schedule
from tests.training.test_recovery import config, setup


def test_literal_paper_schedule_reports_collapse(tmp_path):
    cfg = config(tmp_path)
    cfg["training"].update(
        phase_a_epochs=100,
        phase_b_epochs=1000,
        phase_a_steps_per_epoch=10,
        phase_b_steps_per_epoch=10,
        scheduler_step=10,
        scheduler_gamma=0.5,
    )
    plan = describe_schedule(cfg)
    rates = plan["phases"]["phase_b"]["learning_rate"]
    assert rates["updates_between_decays"] == 100
    assert rates["final_epoch"] == 0.01 * 0.5**99
    assert rates["after_phase"] == 0.01 * 0.5**100
    assert len(plan["warnings"]) == 1
    assert "phase_b" in plan["warnings"][0]


def test_unsupported_scheduler_is_rejected_by_prelaunch_plan(tmp_path):
    cfg = config(tmp_path)
    cfg["training"]["scheduler"] = "BogusScheduler"
    with pytest.raises(ValueError, match="Unknown scheduler: 'BogusScheduler'"):
        describe_schedule(cfg)


@pytest.mark.parametrize("value", ["false", 1, [], None])
def test_scheduler_enable_flag_requires_boolean(tmp_path, value):
    cfg = config(tmp_path)
    cfg["training"]["use_scheduler"] = value
    with pytest.raises(ValueError, match="use_scheduler must be a boolean"):
        describe_schedule(cfg)


def test_missing_scheduler_enable_flag_defaults_to_disabled(tmp_path):
    cfg = config(tmp_path)
    del cfg["training"]["use_scheduler"]
    plan = describe_schedule(cfg)
    assert plan["warnings"] == []
    assert all("learning_rate" not in phase for phase in plan["phases"].values())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scheduler_step", "10", "scheduler_step must be a positive integer"),
        ("scheduler_step", True, "scheduler_step must be a positive integer"),
        ("scheduler_gamma", "0.5", "scheduler_gamma must be finite and nonnegative"),
        ("scheduler_gamma", float("nan"), "scheduler_gamma must be finite and nonnegative"),
        ("learning_rate", None, "learning_rate must be finite and nonnegative"),
        ("learning_rate", float("inf"), "learning_rate must be finite and nonnegative"),
    ],
)
def test_malformed_scheduler_fields_raise_clean_value_error(tmp_path, field, value, message):
    cfg = config(tmp_path)
    cfg["training"][field] = value
    with pytest.raises(ValueError, match=message):
        describe_schedule(cfg)


def test_disabled_scheduler_ignores_scheduler_configuration(tmp_path):
    cfg = config(tmp_path, scheduler=False)
    cfg["training"].update(
        scheduler="unsupported but disabled",
        scheduler_step=True,
        scheduler_gamma="invalid",
        learning_rate=None,
    )
    plan = describe_schedule(cfg)
    assert plan["warnings"] == []
    assert all("learning_rate" not in phase for phase in plan["phases"].values())


@pytest.mark.parametrize("enabled", [True, False])
def test_epoch_events_report_used_and_next_lr(tmp_path, enabled, capsys):
    cfg = config(tmp_path, scheduler=enabled)
    cfg["training"].update(phase_a_epochs=2, phase_b_epochs=2, scheduler_gamma=0.5)
    model, loader = setup()
    run_cascaded_training(cfg, model, torch.device("cpu"), loader)
    events = [
        json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()
    ]
    epochs = [event for event in events if event["event"] == "epoch_completed"]
    assert len(epochs) == 4
    for event in epochs:
        exponent = event["epoch"] - 1 if enabled else 0
        assert event["learning_rates"] == [0.01 * 0.5**exponent]
        assert event["next_learning_rates"] == [0.01 * 0.5 ** (exponent + int(enabled))]
    output = capsys.readouterr().out
    assert "next_lr=" in output
    assert "tumor_dice=" in output
