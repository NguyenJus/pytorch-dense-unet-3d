"""CPU-only command contracts for recoverable run operations."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from dense_unet_3d import cli
from dense_unet_3d.evaluation.evaluate import EvaluationInterrupted, evaluate
from dense_unet_3d.training.runtime import RunSession


@pytest.fixture
def config_path(tmp_path):
    config = {
        "pathing": {
            "model_save_dir": str(tmp_path),
            "run_name": "run",
            "test_img_dirs": ["validation"],
        },
        "training": {
            "phase_a_epochs": 1,
            "phase_b_epochs": 1,
            "phase_a_steps_per_epoch": 1,
            "phase_b_steps_per_epoch": 1,
        },
        "dataset": {"batch_size": 2},
        "gpu": {"use_gpu": True, "gpu_name": "cuda:0"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def args(config_path, *options):
    return cli._build_parser().parse_args(["train", "--config", str(config_path), *options])


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_budget_options_reject_invalid_values(config_path, value):
    with pytest.raises(SystemExit):
        args(config_path, "--wall-seconds", value)


@pytest.mark.parametrize("command", ["resume", "status", "stop"])
def test_runtime_subcommands_help(command):
    with pytest.raises(SystemExit) as exc:
        cli._build_parser().parse_args([command, "--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize(
    "options,message",
    [(["--force"], "--force requires --wait-seconds")]
    + [
        ([f"--wait-seconds={value}"], "--wait-seconds")
        for value in ("0", "-1", "nan", "inf", "-inf", "invalid")
    ],
)
def test_stop_entrypoint_rejects_invalid_usage(tmp_path, options, message):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dense_unet_3d.cli",
            "stop",
            "--run-dir",
            str(tmp_path / "missing"),
            *options,
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "usage:" in result.stderr
    assert message in result.stderr
    assert "Traceback" not in result.stderr
    assert "FileNotFoundError" not in result.stderr


def test_plan_and_accounting_precede_preflight_and_allocation(config_path, monkeypatch, capsys):
    from dense_unet_3d.dataset import prepare_dataset

    def preflight(config, **kwargs):
        output = capsys.readouterr().out
        assert "Resolved training plan" in output
        assert '"total_updates": 2' in output
        assert '"validation_passes": 1' in output
        assert "Effective persistent budget" in output
        assert "Validation workload before preflight: 2 cases, 1 batches" in output
        assert (config_path.parent / "run" / "runtime.json").exists()
        raise ValueError("preflight sentinel")

    monkeypatch.setattr(prepare_dataset, "discover_pairs", lambda *_: [("v", "s")] * 2)
    monkeypatch.setattr(prepare_dataset, "preflight_config", preflight)
    monkeypatch.setattr(
        cli, "_device_from_config", lambda *_: pytest.fail("allocated before preflight")
    )
    with pytest.raises(ValueError, match="preflight sentinel"):
        cli._cmd_train(args(config_path, "--budget-seconds", "30"))
    state = json.loads((config_path.parent / "run" / "runtime.json").read_text())
    assert state["terminal_reason"] == "failed"
    assert state["cumulative_seconds"] > 0


def test_dry_run_forces_cpu_and_preserves_global_rng(config_path, monkeypatch):
    from dense_unet_3d.training import cascaded_driver

    torch.manual_seed(31)
    rng = torch.get_rng_state().clone()
    cli._make_dry_run_loader()
    assert torch.equal(rng, torch.get_rng_state())
    devices = []

    def driver(config, model, device, *loaders, **kwargs):
        devices.append(device.type)
        assert kwargs["resume"] is False
        return {"terminal_reason": "user stopped"}

    monkeypatch.setattr(cli, "_device_from_config", lambda *_: pytest.fail("CUDA resolver called"))
    monkeypatch.setattr(cascaded_driver, "run_cascaded_training", driver)
    cli._cmd_train(args(config_path, "--dry-run"))
    assert devices == ["cpu"]
    state = json.loads((config_path.parent / "run" / "runtime.json").read_text())
    assert state["terminal_reason"] == "user stopped"


def test_resume_plan_uses_persisted_budget(config_path, capsys):
    config = cli._load_config(str(config_path))
    with RunSession(config, budget_seconds=30) as session:
        session.finish("user stopped")
    options = cli._build_parser().parse_args(["resume", "--config", str(config_path)])
    cli._print_training_plan(config, options)
    assert "persistent cumulative limit: 30 seconds" in capsys.readouterr().out


def test_status_reads_durable_json(config_path, capsys):
    config = cli._load_config(str(config_path))
    with RunSession(config) as session:
        session.finish("user stopped")
    options = cli._build_parser().parse_args(
        ["status", "--run-dir", str(config_path.parent / "run")]
    )
    cli._cmd_status(options)
    state = json.loads(capsys.readouterr().out)
    assert state["terminal_reason"] == "user stopped"
    assert state["run_id"]


def test_final_evaluation_stop_preserves_training_completion(config_path, monkeypatch, capsys):
    from dense_unet_3d.evaluation import evaluate as evaluation
    from dense_unet_3d.training import cascaded_driver

    monkeypatch.setattr(
        cascaded_driver, "run_cascaded_training", lambda *a, **kw: {"terminal_reason": "completed"}
    )
    monkeypatch.setattr(cli, "_load_model_from_checkpoint", lambda *a: torch.nn.Conv3d(1, 3, 1))

    def exhausted(*a, **kw):
        raise EvaluationInterrupted("evaluation ceiling")

    monkeypatch.setattr(evaluation, "evaluate", exhausted)
    cli._cmd_train(args(config_path, "--dry-run", "--final-eval"))
    assert "Training budget exhausted" in capsys.readouterr().out
    records = [
        json.loads(line)
        for line in (config_path.parent / "run" / "events.jsonl").read_text().splitlines()
    ]
    assert any(row.get("training_completed") for row in records)
    assert any(row["event"] == "final_evaluation_stopped" for row in records)


def test_bounded_eval_withholds_incomplete_metrics():
    loader = DataLoader(
        TensorDataset(torch.zeros(2, 1, 2, 2, 2), torch.zeros(2, 1, 2, 2, 2, dtype=torch.long)),
        batch_size=1,
    )
    model = torch.nn.Conv3d(1, 3, 1)
    with pytest.raises(EvaluationInterrupted, match="full-split metrics withheld"):
        evaluate(model, torch.device("cpu"), loader, max_batches=1)
    with pytest.raises(EvaluationInterrupted, match="user stopped"):
        evaluate(model, torch.device("cpu"), loader, stop_requested=lambda: "user stopped")


def test_bounded_eval_rejects_nonfinite_logits():
    loader = DataLoader(
        TensorDataset(torch.zeros(1, 1, 2, 2, 2), torch.zeros(1, 1, 2, 2, 2, dtype=torch.long))
    )
    model = torch.nn.Conv3d(1, 3, 1)
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        evaluate(model, torch.device("cpu"), loader)


def test_cpu_cli_completed_resume_does_not_replay_updates(config_path):
    config = cli._load_config(str(config_path))
    config["training"].update(
        optimizer="SGD",
        learning_rate=0.01,
        momentum=0.5,
        criterion="CrossEntropyLoss",
        use_scheduler=False,
        class_weights={"background": 0.2, "liver": 1.2, "lesion": 2.2},
    )
    config_path.write_text(yaml.safe_dump(config))
    cli._cmd_train(args(config_path, "--dry-run", "--budget-seconds", "300"))
    run_dir = config_path.parent / "run"
    before = torch.load(run_dir / "recovery.pt", map_location="cpu", weights_only=False)
    resume_args = cli._build_parser().parse_args(
        ["resume", "--config", str(config_path), "--dry-run"]
    )
    cli._cmd_train(resume_args)
    after = torch.load(run_dir / "recovery.pt", map_location="cpu", weights_only=False)
    assert after["global_step"] == before["global_step"] == 2
    for key, value in before["model_state_dict"].items():
        assert torch.equal(value, after["model_state_dict"][key])
    state = json.loads((run_dir / "runtime.json").read_text())
    assert state["terminal_reason"] == "completed"
    assert state["budget_seconds"] == 300
    assert len(state["attempts"]) == 2
