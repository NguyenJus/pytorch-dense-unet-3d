import sys

import pytest

from scripts import diagnose_reconstruction as diagnostic


def test_reference_config_is_rejected_before_artifact_creation(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose_reconstruction.py",
            "--config",
            "configs/reconstruction-reference.yaml",
            "--artifacts",
            str(artifacts),
            "--report",
            str(report),
        ],
    )
    with pytest.raises(ValueError, match="Reference reconstruction is not launchable"):
        diagnostic.main()
    assert not artifacts.exists()
    assert not report.exists()


def test_zero_step_markdown_uses_receipt_values(tmp_path):
    report = tmp_path / "diagnostics.json"
    diagnostic.write_markdown(
        report,
        {
            "status": "failed diagnostic gates",
            "updates": 3,
            "budget": {"max_updates": 7, "max_gpu_wall_seconds": 11},
            "gpu_wall_seconds": 2.5,
            "model": {
                "model_config": {"name": "receipt_model"},
                "parameters": 123,
            },
            "gpu_info": "Receipt GPU, driver, memory",
            "execution": {"deterministic": False},
            "resolved_config": {"dataset": {"resize_dims": {"D": 2, "H": 3, "W": 4}}},
            "selected_samples": [
                {
                    "case_id": "-7",
                    "native_depth_bounds": [5, 7],
                    "class_voxel_counts": [1, 2, 3],
                    "tumor_fraction": 0.5,
                    "valid_voxels": 6,
                }
            ],
            "overfit": {
                "before": {"mean_loss": 2.0, "mean_tumor_dice": 0.0},
                "after": {
                    "mean_loss": 1.0,
                    "mean_tumor_dice": 0.0,
                    "samples": [{"predicted_class_counts": [1, 2, 3], "tumor_overlap_voxels": 0}],
                },
                "steps": [],
                "passed": False,
            },
            "validation_eval": {"all_finite": True},
            "validation_sample": {"case_id": "-9"},
        },
    )
    text = report.with_suffix(".md").read_text()
    assert "3/7 updates" in text
    assert "2.500/11 seconds" in text
    assert "receipt_model (123 parameters), Receipt GPU" in text
    assert "`(1,1,2,3,4)`" not in text  # No profile means no fabricated shape claim.
    assert "case `-7`, depth [5, 7], 3 tumor voxels" in text
    assert "case `-9`" in text
    assert "No overfit update fit within the remaining allocation" in text
    assert "64,591,723" not in text
