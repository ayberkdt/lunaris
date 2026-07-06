"""Injection Dispersion Analysis terminology smoke test (Phase 1 / Phase 10).

The batch/ensemble workflow perturbs injection conditions and spacecraft
parameters; its user-facing language must be the specific "injection dispersion"
terminology, not the generic "uncertainty analysis" label. These checks read the
source/docs text directly (no Qt/display needed) so they run in plain CI.

They also lock the compatibility contract: the dispersion-model dataclass names
(`StateUncertainty`, `SpacecraftUncertainty`) are preserved through the
batch-subsystem rework (the top-level config is now `BatchPropagationConfig`),
and the UQ covariance report keeps its own defined meaning.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


def test_ui_page_uses_injection_dispersion_card_labels():
    text = _read("src/lunaris/ui/pages/batch_propagation_page.py")
    assert "Injection State Dispersion" in text
    assert "Spacecraft Property Dispersion" in text
    # The generic card titles must be gone.
    assert "Initial State Uncertainty" not in text
    assert "Spacecraft Property Uncertainty" not in text


def test_ui_validation_messages_use_dispersion_language():
    text = _read("src/lunaris/ui/pages/batch_propagation_page.py")
    assert "Injection position dispersion" in text
    assert "Injection velocity dispersion" in text


def test_cli_help_uses_injection_dispersion():
    text = _read("src/lunaris/cli/batch_runner.py")
    assert "injection-dispersion" in text


def test_docs_use_injection_dispersion_terminology():
    readme = _read("README.md")
    assert "injection dispersion analysis" in readme.lower()
    hpc = _read("docs/HPC.md")
    assert "injection dispersion" in hpc.lower()


def test_public_dataclass_names_are_preserved():
    # The dispersion-model dataclass names are kept even though the top-level
    # config was renamed to BatchPropagationConfig in the batch-subsystem rework.
    from lunaris.common.batch_defs import (  # noqa: F401
        BatchPropagationConfig,
        SpacecraftUncertainty,
        StateUncertainty,
    )

    cfg = BatchPropagationConfig()
    assert hasattr(cfg, "state")
    assert hasattr(cfg, "spacecraft")


def test_uq_report_terminology_is_defined_not_generic():
    text = _read("docs/UQ_COVARIANCE.md")
    # UQ report keeps its own explicit, provenance-stamped meaning …
    assert "Uncertainty Quantification" in text
    # … and it names the injection-dispersion workflow it is derived from.
    assert "Injection Dispersion Analysis" in text


@pytest.mark.parametrize(
    "rel",
    [
        "src/lunaris/common/batch_defs.py",
        "src/lunaris/batch/engine.py",
    ],
)
def test_batch_config_and_engine_use_dispersion_prose(rel):
    text = _read(rel).lower()
    assert "injection-dispersion" in text
