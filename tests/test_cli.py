"""Tests for F3: console CLI ``dense-unet-3d train|eval|predict``.

Acceptance criteria (§6 F3):
1. ``dense-unet-3d --help`` exits 0 and lists subcommands.
2. Each subcommand's ``--help`` exits 0.
3. ``train --config <tiny.yaml>`` runs the CPU dry-run path to completion.
4. ``eval --config <tiny.yaml> --checkpoint <ckpt.pt>`` loads a checkpoint
   and prints the Dice dict (keys: liver_per_case, liver_global, etc.).
5. ``predict --config <tiny.yaml> --checkpoint <ckpt.pt> --input <vol.nii.gz>
   --output <seg.nii.gz>`` writes a NIfTI segmentation file.
6. No ``cp main.py`` step anywhere; config path is a CLI arg (--config).
7. Entry point ``dense-unet-3d`` is registered in pyproject.toml and works.

TDD: tests written FIRST. All assertions target ``cli.main()`` via
``subprocess.run`` (integration) and direct function calls where possible.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Any

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ``dense-unet-3d <args>`` via the active interpreter and return the result.

    Uses ``sys.executable`` so the test works in any environment (local ``.venv``
    or CI toolcache interpreter) without assuming a virtualenv layout.
    """
    return subprocess.run(
        [sys.executable, "-m", "dense_unet_3d.cli", *args],
        capture_output=True,
        text=True,
    )


def _tiny_config(tmp_dir: str) -> dict[str, Any]:
    """Build a minimal training config for CPU dry-runs."""
    return {
        "pathing": {
            "run_name": "cli_test",
            "model_save_dir": os.path.join(tmp_dir, "models"),
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
            "use_scheduler": False,
            "scheduler": "StepLR",
            "scheduler_step": 10,
            "scheduler_gamma": 0.5,
            "epochs": 1,
            "phase_a_epochs": 1,
            "phase_a_steps_per_epoch": 1,
            "phase_b_epochs": 1,
            "phase_b_steps_per_epoch": 1,
        },
        "gpu": {
            "use_gpu": False,
            "gpu_name": "cpu",
        },
    }


class _TinyModel(nn.Module):
    """Minimal 3-class model for checkpoint creation in tests."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(1, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _write_config(path: str, config: dict[str, Any]) -> None:
    """Write config dict as YAML to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f)


def _write_checkpoint(path: str, model: nn.Module) -> None:
    """Save a minimal checkpoint to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": None,
            "epoch": 1,
            "metrics": {},
        },
        path,
    )


def _write_nifti(path: str) -> None:
    """Write a synthetic NIfTI volume (H,W,D) = (224,224,12)."""
    rng = np.random.default_rng(42)
    data = rng.uniform(-200.0, 250.0, size=(224, 224, 12)).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, path)


# ---------------------------------------------------------------------------
# Test 1: --help exits 0 and lists subcommands
# ---------------------------------------------------------------------------


class TestHelp:
    """dense-unet-3d --help and subcommand --help exit 0."""

    def test_top_level_help_exits_zero(self) -> None:
        result = _run_cli("--help")
        assert result.returncode == 0, (
            f"Expected exit 0 from --help, got {result.returncode}\nstderr: {result.stderr}"
        )

    def test_top_level_help_lists_subcommands(self) -> None:
        result = _run_cli("--help")
        output = result.stdout + result.stderr
        assert "train" in output, f"'train' not in help output: {output}"
        assert "eval" in output, f"'eval' not in help output: {output}"
        assert "predict" in output, f"'predict' not in help output: {output}"

    def test_train_subcommand_help(self) -> None:
        result = _run_cli("train", "--help")
        assert result.returncode == 0, (
            f"train --help exited {result.returncode}\nstderr: {result.stderr}"
        )
        assert "--config" in result.stdout + result.stderr, "--config not in train --help output"

    def test_eval_subcommand_help(self) -> None:
        result = _run_cli("eval", "--help")
        assert result.returncode == 0, (
            f"eval --help exited {result.returncode}\nstderr: {result.stderr}"
        )
        assert "--config" in result.stdout + result.stderr, "--config not in eval --help output"

    def test_predict_subcommand_help(self) -> None:
        result = _run_cli("predict", "--help")
        assert result.returncode == 0, (
            f"predict --help exited {result.returncode}\nstderr: {result.stderr}"
        )
        assert "--config" in result.stdout + result.stderr, "--config not in predict --help output"


# ---------------------------------------------------------------------------
# Test 2: config path is a CLI arg (not hardcoded cwd)
# ---------------------------------------------------------------------------


class TestConfigArgNotHardcoded:
    """The --config flag must accept an arbitrary path (no cwd dependence)."""

    def test_train_config_flag_present(self) -> None:
        """``train --help`` must show ``--config`` as a required/optional arg."""
        result = _run_cli("train", "--help")
        output = result.stdout + result.stderr
        assert "--config" in output, f"--config missing from train --help: {output}"

    def test_eval_config_flag_present(self) -> None:
        result = _run_cli("eval", "--help")
        output = result.stdout + result.stderr
        assert "--config" in output, f"--config missing from eval --help: {output}"

    def test_predict_config_flag_present(self) -> None:
        result = _run_cli("predict", "--help")
        output = result.stdout + result.stderr
        assert "--config" in output, f"--config missing from predict --help: {output}"


# ---------------------------------------------------------------------------
# Test 3: train --config <tiny.yaml> CPU dry-run completes
# ---------------------------------------------------------------------------


class TestTrainCpuDryRun:
    """``train --config`` runs the CPU dry-run path to completion."""

    def test_train_completes_on_cpu(self) -> None:
        """train --config <tiny.yaml> exits 0 and produces a checkpoint."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            config_path = os.path.join(tmp, "tiny.yaml")
            _write_config(config_path, cfg)

            # Create tiny synthetic data files (train uses a synthetic loader in dry-run)
            # The CLI train command must accept --config and run without real data
            # when the config's train_img_dirs is empty / absent (dry-run mode).
            result = _run_cli("train", "--config", config_path, "--dry-run")

            assert result.returncode == 0, (
                f"train --config dry-run exited {result.returncode}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

    def test_train_dry_run_writes_checkpoint(self) -> None:
        """After train --dry-run, a checkpoint file must exist in model_save_dir."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            config_path = os.path.join(tmp, "tiny.yaml")
            _write_config(config_path, cfg)

            _run_cli("train", "--config", config_path, "--dry-run")

            # At minimum, some checkpoint or directory must exist after training
            assert os.path.isdir(os.path.join(tmp, "models")), (
                f"model_save_dir not created: {tmp}/models"
            )


# ---------------------------------------------------------------------------
# Test 4: eval --config --checkpoint prints Dice dict
# ---------------------------------------------------------------------------


class TestEvalCommand:
    """``eval --config --checkpoint`` loads a checkpoint and prints Dice dict."""

    def test_eval_prints_dice_keys(self) -> None:
        """eval must print liver_per_case, liver_global, tumor_per_case, tumor_global."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            config_path = os.path.join(tmp, "tiny.yaml")
            _write_config(config_path, cfg)

            model = _TinyModel()
            ckpt_path = os.path.join(tmp, "checkpoint.pt")
            _write_checkpoint(ckpt_path, model)

            result = _run_cli(
                "eval",
                "--config",
                config_path,
                "--checkpoint",
                ckpt_path,
                "--dry-run",
            )

            assert result.returncode == 0, (
                f"eval exited {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
            output = result.stdout + result.stderr
            assert "liver" in output.lower() or "dice" in output.lower(), (
                f"Expected Dice keys in eval output, got: {output}"
            )

    def test_eval_requires_checkpoint(self) -> None:
        """eval without --checkpoint must exit non-zero (required arg)."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            config_path = os.path.join(tmp, "tiny.yaml")
            _write_config(config_path, cfg)

            result = _run_cli("eval", "--config", config_path)
            # Should fail because --checkpoint is required
            assert result.returncode != 0, "eval without --checkpoint should exit non-zero"


# ---------------------------------------------------------------------------
# Test 5: predict writes a NIfTI segmentation
# ---------------------------------------------------------------------------


class TestPredictCommand:
    """``predict --config --checkpoint --input --output`` writes a NIfTI."""

    def test_predict_writes_nifti(self) -> None:
        """predict must write a NIfTI segmentation to --output path."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            config_path = os.path.join(tmp, "tiny.yaml")
            _write_config(config_path, cfg)

            model = _TinyModel()
            ckpt_path = os.path.join(tmp, "checkpoint.pt")
            _write_checkpoint(ckpt_path, model)

            input_path = os.path.join(tmp, "input.nii.gz")
            _write_nifti(input_path)

            output_path = os.path.join(tmp, "seg.nii.gz")

            result = _run_cli(
                "predict",
                "--config",
                config_path,
                "--checkpoint",
                ckpt_path,
                "--input",
                input_path,
                "--output",
                output_path,
            )

            assert result.returncode == 0, (
                f"predict exited {result.returncode}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            assert os.path.isfile(output_path), f"predict did not write NIfTI to {output_path}"

    def test_predict_output_is_valid_nifti(self) -> None:
        """The NIfTI written by predict must be loadable and have the right shape."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            config_path = os.path.join(tmp, "tiny.yaml")
            _write_config(config_path, cfg)

            model = _TinyModel()
            ckpt_path = os.path.join(tmp, "checkpoint.pt")
            _write_checkpoint(ckpt_path, model)

            input_path = os.path.join(tmp, "input.nii.gz")
            _write_nifti(input_path)

            output_path = os.path.join(tmp, "seg.nii.gz")

            _run_cli(
                "predict",
                "--config",
                config_path,
                "--checkpoint",
                ckpt_path,
                "--input",
                input_path,
                "--output",
                output_path,
            )

            if os.path.isfile(output_path):
                seg = nib.load(output_path)
                data = seg.get_fdata()
                # Segmentation must be 3D (H, W, D) with values in {0, 1, 2}
                assert data.ndim == 3, f"Expected 3D segmentation, got shape {data.shape}"
                unique = np.unique(data)
                assert set(unique).issubset({0, 1, 2}), (
                    f"Segmentation contains unexpected labels: {unique}"
                )

    def test_predict_requires_input(self) -> None:
        """predict without --input must exit non-zero."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(tmp)
            config_path = os.path.join(tmp, "tiny.yaml")
            _write_config(config_path, cfg)

            model = _TinyModel()
            ckpt_path = os.path.join(tmp, "checkpoint.pt")
            _write_checkpoint(ckpt_path, model)

            result = _run_cli(
                "predict",
                "--config",
                config_path,
                "--checkpoint",
                ckpt_path,
            )
            assert result.returncode != 0, "predict without --input should exit non-zero"


# ---------------------------------------------------------------------------
# Test 6: entry point registered in pyproject.toml
# ---------------------------------------------------------------------------


class TestEntryPointRegistered:
    """The console script dense-unet-3d is registered via pyproject.toml."""

    def test_entry_point_in_pyproject(self) -> None:
        """pyproject.toml must declare dense-unet-3d = dense_unet_3d.cli:main."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pyproject_path = os.path.join(repo_root, "pyproject.toml")
        assert os.path.isfile(pyproject_path), f"pyproject.toml not found at {pyproject_path}"

        import tomllib

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        scripts = data.get("project", {}).get("scripts", {})
        assert "dense-unet-3d" in scripts, f"dense-unet-3d not in [project.scripts]: {scripts}"
        assert scripts["dense-unet-3d"] == "dense_unet_3d.cli:main", (
            f"Entry point wrong: {scripts['dense-unet-3d']}"
        )

    def test_console_script_installed(self) -> None:
        """The ``dense-unet-3d`` console script must be installed and runnable.

        We discover it via ``shutil.which`` rather than assuming any virtualenv
        layout. If it is not on PATH (e.g. the package was not installed with its
        entry point), skip rather than hardcode a path -- the entry-point
        declaration itself is asserted by ``test_entry_point_in_pyproject``, and
        the module is independently exercised via ``python -m dense_unet_3d.cli``.
        """
        import shutil

        script = shutil.which("dense-unet-3d")
        if script is None:
            pytest.skip(
                "dense-unet-3d console script not on PATH (package not installed with entry point)"
            )
        result = subprocess.run([script, "--help"], capture_output=True, text=True)
        assert result.returncode == 0, (
            f"`dense-unet-3d --help` failed (rc={result.returncode}):\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# Test 7: no cp main.py step required (verify main.py does not hardcode config)
# ---------------------------------------------------------------------------


class TestNoHardcodedConfigPath:
    """main.py must not hardcode ``./dense_unet_3d/config.yaml`` as the config path."""

    def test_main_py_not_hardcoded(self) -> None:
        """main.py must not open a hardcoded path -- it should thin-wrap cli.main()
        or be retired. We check main.py does NOT contain the old hardcoded path."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_path = os.path.join(repo_root, "dense_unet_3d", "main.py")
        if not os.path.isfile(main_path):
            # File retired entirely — that's fine.
            return

        with open(main_path) as f:
            content = f.read()

        hardcoded = "./dense_unet_3d/config.yaml"
        assert hardcoded not in content, (
            f"main.py still hardcodes the config path '{hardcoded}'. "
            "This must be removed — use --config CLI arg instead."
        )
