# tests/test_surrogate_conservativeness_contract.py
"""Audit F2 — the conservative / non-conservative taxonomy flag.

The scientific "is this field the gradient of a scalar potential" distinction
lives in the explicit ``is_conservative`` flag (and ``runtime_model_kind``),
never in the class hierarchy. On main only the conservative
``potential_autograd`` surrogate is loadable (the non-conservative force_direct
variant is archived), but the flag remains authoritative and these tests forbid
isinstance-based conservativeness checks so a future non-conservative kind can
never be mis-identified by class.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
_ = pytest.importorskip("torch.nn")

torch = pytest.importorskip("torch")

from lunaris.surrogate.runtime import SurrogateGravityModel
from lunaris.surrogate.st_lrps.runtime.force_model import (
    BaseSurrogateRuntime,
    PotentialAutogradRuntime,
    SurrogateForceModel,
    load_surrogate_force_model,
)
from st_lrps_contract_test_utils import make_contract_run

REPO_SRC = Path(__file__).resolve().parents[1] / "src" / "lunaris"


def test_class_level_flags():
    # Unknown kinds default to non-conservative (fail-safe); the supported
    # potential runtime is conservative by construction.
    assert BaseSurrogateRuntime.is_conservative is False
    assert PotentialAutogradRuntime.is_conservative is True
    assert SurrogateForceModel.is_conservative is True


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


def test_no_isinstance_conservativeness_checks_in_source():
    """Forbid isinstance checks against the potential runtime classes.

    isinstance against the potential classes cannot distinguish a conservative
    runtime from a hypothetical non-conservative one (a future kind would
    inherit the same base for implementation reuse); the ``is_conservative``
    flag is the only safe discriminator.
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
        "conservative from non-conservative runtimes; use the is_conservative "
        "flag or runtime_model_kind instead:\n" + "\n".join(offenders)
    )
