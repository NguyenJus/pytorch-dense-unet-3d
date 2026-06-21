"""Tests for CI workflow YAML (A2).

Acceptance criteria (§6 A2):
- YAML is valid (parseable).
- Triggers on `push` and `pull_request`.
- Runs the four gate commands: ruff check, ruff format --check, mypy, pytest -q.
- No GPU steps (no 'gpu', 'cuda', 'nvidia' in the workflow).
- Python version pinned to 3.10+.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

WORKFLOW_PATH = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"


WorkflowDoc = dict  # yaml.safe_load returns an untyped dict


def _load_workflow() -> WorkflowDoc:
    """Parse the CI YAML and return the parsed dict."""
    assert WORKFLOW_PATH.exists(), f"CI workflow not found at {WORKFLOW_PATH}"
    with WORKFLOW_PATH.open() as fh:
        doc: WorkflowDoc = yaml.safe_load(fh)
    assert doc is not None, "YAML parsed to None (empty file?)"
    return doc


def _all_run_steps(workflow: WorkflowDoc) -> list[str]:
    """Collect every 'run:' string from all jobs/steps."""
    runs: list[str] = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            if "run" in step:
                runs.append(step["run"])
    return runs


def test_yaml_is_parseable() -> None:
    """YAML must load without error."""
    doc = _load_workflow()
    assert isinstance(doc, dict)


def test_triggers_push_and_pull_request() -> None:
    """Workflow must trigger on both push and pull_request events."""
    doc = _load_workflow()
    # PyYAML parses 'on:' as the Python bool True (a known quirk)
    on = doc.get("on") or doc.get(True)  # noqa: FBT003
    assert on is not None, "'on:' key missing from workflow"
    # on may be a list or dict
    if isinstance(on, list):
        keys = set(on)
    elif isinstance(on, dict):
        keys = set(on.keys())
    else:
        keys = {on}
    assert "push" in keys, f"'push' trigger missing; triggers found: {keys}"
    assert "pull_request" in keys, f"'pull_request' trigger missing; triggers found: {keys}"


def test_ruff_check_step_present() -> None:
    """Workflow must run 'ruff check'."""
    runs = _all_run_steps(_load_workflow())
    combined = "\n".join(runs)
    assert re.search(r"ruff\s+check", combined), (
        f"'ruff check' not found in any run step.\nSteps:\n{combined}"
    )


def test_ruff_format_check_step_present() -> None:
    """Workflow must run 'ruff format --check'."""
    runs = _all_run_steps(_load_workflow())
    combined = "\n".join(runs)
    assert re.search(r"ruff\s+format\s+--check", combined), (
        f"'ruff format --check' not found in any run step.\nSteps:\n{combined}"
    )


def test_mypy_step_present() -> None:
    """Workflow must run mypy."""
    runs = _all_run_steps(_load_workflow())
    combined = "\n".join(runs)
    assert re.search(r"\bmypy\b", combined), (
        f"'mypy' not found in any run step.\nSteps:\n{combined}"
    )


def test_pytest_step_present() -> None:
    """Workflow must run pytest -q."""
    runs = _all_run_steps(_load_workflow())
    combined = "\n".join(runs)
    assert re.search(r"pytest\s+-q", combined), (
        f"'pytest -q' not found in any run step.\nSteps:\n{combined}"
    )


def test_no_gpu_steps() -> None:
    """No GPU-related keywords should appear anywhere in the workflow."""
    assert WORKFLOW_PATH.exists()
    raw = WORKFLOW_PATH.read_text()
    gpu_patterns = [r"\bgpu\b", r"\bcuda\b", r"\bnvidia\b", r"runs-on:.*gpu"]
    for pattern in gpu_patterns:
        assert not re.search(pattern, raw, re.IGNORECASE), (
            f"GPU-related pattern '{pattern}' found in CI workflow"
        )


def test_python_version_is_310_or_higher() -> None:
    """Python version must be pinned to 3.10 or higher."""
    assert WORKFLOW_PATH.exists()
    raw = WORKFLOW_PATH.read_text()
    # Look for python-version: "3.X" or '3.X'
    versions = re.findall(r"""python-version['":\s]+['"]([\d.]+)['"]""", raw)
    if not versions:
        # Also check matrix style: ["3.10", "3.11"]
        versions = re.findall(r"""(3\.\d+)""", raw)
    assert versions, "No Python version specification found in CI workflow"
    for v in versions:
        try:
            major, minor = v.split(".")[:2]
            assert int(major) == 3 and int(minor) >= 10, f"Python version {v} is below 3.10"
        except (ValueError, IndexError):
            pass  # skip unparseable tokens
