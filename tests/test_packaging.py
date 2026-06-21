"""Tests for A1 — pyproject.toml + packaging acceptance criteria.

Acceptance criteria (from spec §6 A1):
1. pyproject.toml parses
2. `import dense_unet_3d` works (after editable install)
3. Console script entry point `dense-unet-3d` is declared in pyproject.toml
4. ruff + mypy config sections are present in pyproject.toml
5. pandas and torchaudio are NOT in dependencies (audited away)
6. torchvision IS in dependencies (still used by transforms)
"""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def _load_pyproject() -> dict:
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)


# ── 1. pyproject.toml parses ──────────────────────────────────────────────────


def test_pyproject_parses() -> None:
    data = _load_pyproject()
    assert isinstance(data, dict), "pyproject.toml must parse to a dict"


# ── 2. Package name + Python floor ───────────────────────────────────────────


def test_package_name_and_python_floor() -> None:
    data = _load_pyproject()
    project = data["project"]
    assert project["name"] == "dense-unet-3d", f"name={project['name']!r}"
    requires = project["requires-python"]
    # Must be 3.10+
    assert "3.10" in requires or "3.11" in requires or "3.12" in requires, (
        f"requires-python={requires!r} should be >=3.10"
    )


# ── 3. Console entry point declared ──────────────────────────────────────────


def test_console_entry_point_declared() -> None:
    data = _load_pyproject()
    scripts = data["project"].get("scripts", {})
    assert "dense-unet-3d" in scripts, (
        f"'dense-unet-3d' entry point not found in [project.scripts]; got: {scripts}"
    )
    ep = scripts["dense-unet-3d"]
    assert "dense_unet_3d" in ep and "main" in ep, (
        f"Entry point {ep!r} should point to dense_unet_3d.cli:main"
    )


# ── 4. ruff + mypy config sections present ───────────────────────────────────


def test_ruff_config_section_present() -> None:
    data = _load_pyproject()
    assert "ruff" in data.get("tool", {}), "Missing [tool.ruff] section"


def test_mypy_config_section_present() -> None:
    data = _load_pyproject()
    assert "mypy" in data.get("tool", {}), "Missing [tool.mypy] section"


# ── 5. Dropped deps: pandas and torchaudio NOT present ───────────────────────


def test_pandas_not_in_deps() -> None:
    data = _load_pyproject()
    deps: list[str] = data["project"].get("dependencies", [])
    dep_names = [d.split(">=")[0].split("==")[0].split("[")[0].strip().lower() for d in deps]
    assert "pandas" not in dep_names, f"pandas must be dropped (deps={dep_names})"


def test_torchaudio_not_in_deps() -> None:
    data = _load_pyproject()
    deps: list[str] = data["project"].get("dependencies", [])
    dep_names = [d.split(">=")[0].split("==")[0].split("[")[0].strip().lower() for d in deps]
    assert "torchaudio" not in dep_names, f"torchaudio must be dropped (deps={dep_names})"


# ── 6. Required deps present ─────────────────────────────────────────────────


def test_required_deps_present() -> None:
    data = _load_pyproject()
    deps: list[str] = data["project"].get("dependencies", [])
    dep_names = {d.split(">=")[0].split("==")[0].split("[")[0].strip().lower() for d in deps}
    required = {"torch", "numpy", "nibabel", "pyyaml", "tqdm", "matplotlib", "torchvision"}
    missing = required - dep_names
    assert not missing, f"Missing required deps: {missing} (found: {dep_names})"


# ── 7. import dense_unet_3d works ────────────────────────────────────────────


def test_import_dense_unet_3d() -> None:
    import dense_unet_3d  # noqa: F401


# ── 8. cli module importable and has main() ──────────────────────────────────


def test_cli_module_has_main() -> None:
    from dense_unet_3d import cli

    assert callable(cli.main), "dense_unet_3d.cli.main must be callable"


# ── 9. New cli.py passes ruff check ──────────────────────────────────────────


def test_cli_passes_ruff_check() -> None:
    import subprocess
    import sys

    cli_path = ROOT / "dense_unet_3d" / "cli.py"
    python_dir = Path(sys.executable).parent
    ruff_bin = python_dir / "ruff"
    result = subprocess.run(
        [str(ruff_bin), "check", str(cli_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ruff check cli.py failed:\n{result.stdout}\n{result.stderr}"


# ── 10. New cli.py passes mypy ────────────────────────────────────────────────


def test_cli_passes_mypy() -> None:
    import subprocess
    import sys

    cli_path = ROOT / "dense_unet_3d" / "cli.py"
    python_dir = Path(sys.executable).parent
    mypy_bin = python_dir / "mypy"
    result = subprocess.run(
        [str(mypy_bin), str(cli_path), "--ignore-missing-imports"],
        capture_output=True,
        text=True,
    )
    # Filter to only errors that originate from cli.py itself (not transitively
    # from imported modules that have their own pre-existing mypy issues).
    cli_rel = str(cli_path.relative_to(ROOT))
    errors = [line for line in result.stdout.splitlines() if "error:" in line and cli_rel in line]
    assert not errors, "mypy errors in cli.py:\n" + "\n".join(errors)
