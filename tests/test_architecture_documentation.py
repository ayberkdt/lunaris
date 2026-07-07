"""Keep the architecture prose tied to executable repository contracts."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFACTOR_NOTE = REPO_ROOT / "docs/development/FRAME_HANDLING_AND_PHYSICS_REFACTOR.md"


def test_architecture_document_names_every_import_contract() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    import_contract_section = pyproject.split("[tool.importlinter]", 1)[1]
    contract_names = set(
        re.findall(r'^name = "([^"]+)"$', import_contract_section, flags=re.MULTILINE)
    )
    architecture = (REPO_ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    missing = sorted(name for name in contract_names if name not in architecture)
    assert missing == []


def test_readme_names_every_st_lrps_gpu_backend() -> None:
    """README's ST-LRPS capability block must not drift behind the registry.

    The registry (backend_capabilities) is the executable source of truth for
    which GPU ST-LRPS variants exist; the README must name each one so users
    are not misled into thinking GPU ST-LRPS is a single gravity-only backend.
    """
    from lunaris.core.backend_capabilities import list_backend_names

    gpu_st_lrps_backends = [
        name for name in list_backend_names() if name.startswith("gpu_st_lrps")
    ]
    assert gpu_st_lrps_backends, "registry no longer exposes any gpu_st_lrps backend"

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    missing = sorted(name for name in gpu_st_lrps_backends if f"`{name}`" not in readme)
    assert missing == [], f"README does not mention ST-LRPS backend(s): {missing}"


def test_architecture_document_points_to_phase_two_contract_tests() -> None:
    architecture = (REPO_ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    for test_file in (
        "tests/test_sim_config_ssot.py",
        "tests/test_optional_dependency_boundaries.py",
        "tests/test_st_lrps_ui_modularity.py",
    ):
        assert test_file in architecture


def test_refactor_note_covers_gpu_smoke_backends_and_cuda_marker() -> None:
    doc = REFACTOR_NOTE.read_text(encoding="utf-8")
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    for backend in (
        "torch_cpu_sh",
        "torch_cuda_sh",
        "gpu_st_lrps_potential",
        "gpu_st_lrps_third_body",
    ):
        assert backend in doc
    assert "requires_cuda" in doc
    assert "requires_cuda" in pyproject


def test_refactor_note_records_third_body_future_contract() -> None:
    doc = REFACTOR_NOTE.read_text(encoding="utf-8")

    for token in (
        "ExternalBodyPerturber",
        "mu_m3s2",
        "position_table_m",
        "velocity_table_m_s",
        "enabled_for_point_mass",
        "enabled_for_relativity",
        "enabled_for_tides",
    ):
        assert token in doc


def test_refactor_note_documents_optional_force_probe_boundary() -> None:
    doc = REFACTOR_NOTE.read_text(encoding="utf-8")

    assert "get_acceleration_breakdown" in doc
    assert "PropagatorConfig.enable_force_probe" in doc
    assert "force_probe_schema_version" in doc
