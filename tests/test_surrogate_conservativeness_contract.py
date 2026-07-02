# tests/test_surrogate_conservativeness_contract.py
"""Audit F2 — the conservative / non-conservative taxonomy flag.

``DirectForceRuntime`` inherits from ``SurrogateForceModel`` (which inherits
from ``PotentialAutogradRuntime``) for implementation reuse, so ``isinstance``
against the potential classes is True for ``force_direct`` runtimes too. The
scientific distinction therefore lives in the explicit ``is_conservative``
flag (and ``runtime_model_kind``), never in the class hierarchy. These tests
pin the flag values and forbid new isinstance-based conservativeness checks in
the source tree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lunaris.surrogate.runtime import SurrogateGravityModel
from lunaris.surrogate.st_lrps.runtime.force_model import (
    BaseSurrogateRuntime,
    DirectForceRuntime,
    PotentialAutogradRuntime,
    SurrogateForceModel,
    load_surrogate_force_model,
)
from st_lrps_contract_test_utils import make_contract_run

REPO_SRC = Path(__file__).resolve().parents[1] / "src" / "lunaris"


def test_class_level_flags():
    # Unknown kinds default to non-conservative (fail-safe).
    assert BaseSurrogateRuntime.is_conservative is False
    assert PotentialAutogradRuntime.is_conservative is True
    assert SurrogateForceModel.is_conservative is True
    assert DirectForceRuntime.is_conservative is False


def test_isinstance_cannot_distinguish_but_flag_can(tmp_path):
    run = make_contract_run(
        tmp_path,
        cfg_overrides={
            "runtime_model_kind": "force_direct",
            "prediction_kind": "residual_force",
            "output_dim": 3,
        },
    )
    fm = load_surrogate_force_model(run["run_dir"], device="cpu")
    # This is exactly why isinstance is banned for the physics distinction:
    assert isinstance(fm, SurrogateForceModel)
    assert isinstance(fm, PotentialAutogradRuntime)
    # ... and why the flag is authoritative:
    assert fm.is_conservative is False
    assert fm.runtime_model_kind == "force_direct"


def test_loaded_potential_runtime_is_conservative(tmp_path):
    run = make_contract_run(tmp_path)
    fm = load_surrogate_force_model(run["run_dir"], device="cpu")
    assert fm.is_conservative is True
    assert fm.runtime_model_kind == "potential_autograd"


@pytest.mark.requires_data
def test_gravity_provider_mirrors_runtime_flag(tmp_path):
    run = make_contract_run(tmp_path)
    provider = SurrogateGravityModel.from_model_dir(
        str(run["run_dir"]), device_preference="cpu"
    )
    assert provider.is_conservative is True

    direct = make_contract_run(
        tmp_path / "direct",
        cfg_overrides={
            "runtime_model_kind": "force_direct",
            "prediction_kind": "residual_force",
            "output_dim": 3,
        },
    )
    provider_direct = SurrogateGravityModel.from_model_dir(
        str(direct["run_dir"]), device_preference="cpu"
    )
    assert provider_direct.is_conservative is False


def test_no_isinstance_conservativeness_checks_in_source():
    """Forbid new isinstance checks against the potential runtime classes.

    ``isinstance(x, DirectForceRuntime)`` is fine (it identifies the
    non-conservative subclass unambiguously); isinstance against the potential
    classes is the trap this contract exists to prevent.
    """
    pattern = re.compile(
        r"isinstance\([^)]*,\s*\(?[^)]*\b(SurrogateForceModel|PotentialAutogradRuntime)\b"
    )
    offenders = []
    for path in REPO_SRC.rglob("*.py"):
        if path.name == "force_model.py" and "st_lrps" in path.parts:
            continue  # the defining module may use its own classes freely
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(REPO_SRC.parent)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "isinstance against the potential runtime classes cannot distinguish "
        "conservative from force_direct runtimes; use the is_conservative flag "
        "or runtime_model_kind instead:\n" + "\n".join(offenders)
    )
