"""Keep the architecture prose tied to executable repository contracts."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_architecture_document_names_every_import_contract() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    import_contract_section = pyproject.split("[tool.importlinter]", 1)[1]
    contract_names = set(
        re.findall(r'^name = "([^"]+)"$', import_contract_section, flags=re.MULTILINE)
    )
    architecture = (REPO_ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    missing = sorted(name for name in contract_names if name not in architecture)
    assert missing == []


def test_architecture_document_points_to_phase_two_contract_tests() -> None:
    architecture = (REPO_ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    for test_file in (
        "tests/test_sim_config_ssot.py",
        "tests/test_optional_dependency_boundaries.py",
        "tests/test_st_lrps_ui_modularity.py",
    ):
        assert test_file in architecture
