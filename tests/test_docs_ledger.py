"""The paper-faithful run documentation must stay tied to the shipped config."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "research" / "2026-10-21-paper-faithful-run.md"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """Return one Markdown section with whitespace collapsed to single spaces."""
    level = heading.split(" ", 1)[0]
    start = text.index(f"\n{heading}\n")
    end = text.find(f"\n{level} ", start + len(heading) + 2)
    return " ".join(text[start : end if end != -1 else None].split())


def test_ledger_covers_every_config_fidelity_key_and_label() -> None:
    config = yaml.safe_load(_text(ROOT / "configs" / "reconstruction-run.yaml"))
    fidelity = config["experiment"]["fidelity"]
    ledger = _text(LEDGER)
    for key, entry in fidelity.items():
        assert f"| `{key}` | {entry['label']} | {entry['evidence']} |" in ledger, key
    for heading in (
        "## Results",
        "### 2.1 Architecture",
        "### 2.4 Execution",
        "## 4. Alternatives considered",
    ):
        assert heading in ledger


def test_ledger_records_decisions_and_expected_counts() -> None:
    ledger = _text(LEDGER)
    for decision in ("D1", "D2", "D3", "D4", "D5", "D6", "D7"):
        assert re.search(rf"\*\*{decision} ", ledger), decision
    assert "525 updates/epoch" in ledger
    assert "105,000 total" in ledger


def test_readme_and_operations_point_at_ledger_and_run_config() -> None:
    readme = _text(ROOT / "README.md")
    ops = _text(ROOT / "docs" / "training-operations.md")
    assert "docs/research/2026-10-21-paper-faithful-run.md" in readme
    assert "configs/reconstruction-run.yaml" in readme
    assert "reconstruction-run.yaml" in ops
    assert "reconstruction-smoke.yaml" in ops
    assert "epoch-exact" in ops
    assert (ROOT / "configs" / "reconstruction-smoke.yaml").is_file()


def test_readme_architecture_fidelity_retires_resolved_gates() -> None:
    section = _section(_text(ROOT / "README.md"), "## Architecture fidelity")
    for stale in ("remain gates", "remain open before production", "diagnostic reconstruction"):
        assert stale not in section, stale
    assert "docs/research/2026-10-21-paper-faithful-run.md" in section
    assert "is launchable" in section


def test_operations_recipe_resumes_a_clean_stop_without_recover() -> None:
    section = _section(
        _text(ROOT / "docs" / "training-operations.md"), "### Paper-faithful run (2026-10-21)"
    )
    assert "resume --config configs/reconstruction-run.yaml --wall-seconds 86400" in section
    assert "--recover" not in section.split("```")[1]
    assert "--run-dir models/paper-faithful-20261021" in section
    # The subsection closes the 2026-09-30 section instead of swallowing its paragraphs.
    assert "Execution reference: FP32, TF32 off" not in section


def test_ledger_alternatives_are_one_bullet_each() -> None:
    ledger = _text(LEDGER)
    start = ledger.index("## 4. Alternatives considered")
    lines = ledger[start : ledger.index("\n## ", start + 1)].splitlines()
    bold = [line for line in lines if line.lstrip("- ").startswith("**")]
    assert len(bold) == 6
    assert all(line.startswith("- **") for line in bold), bold


def test_ledger_defines_labels_as_the_design_does() -> None:
    ledger = " ".join(_text(LEDGER).split())
    assert "| **LITERAL** | The paper states it; copy it exactly. |" in ledger
    assert "| **INFERRED** | The paper omits it; one choice is made" in ledger
    assert "| **DEVIATION** | The paper states it but the literal reading cannot work;" in ledger
    assert "an **unlabelled** deviation is." in ledger


def test_ledger_records_owner_decision_against_compatibility_forms() -> None:
    from dense_unet_3d.training.experiment import EXECUTION_KEYS

    ledger = " ".join(_text(LEDGER).split())
    start = ledger.index("**D6 ")
    d6 = ledger[start : ledger.index("- **D7 ", start)]
    assert "stays valid" not in d6
    assert all(f"`{key}`" in d6 for key in EXECUTION_KEYS), d6
    assert "owner decision 1" in d6
    notes = _section(_text(LEDGER), "### As-built notes")
    assert "only when non-default" not in notes
    assert "always records the `workers` mapping" in notes
