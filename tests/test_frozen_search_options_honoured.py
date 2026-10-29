"""Frozen-search options must be honoured, recorded, or refused - never dropped.

Each test pins one silent trap found by the pre-run audit of the production
search (``python -m lunaris.cli.frozen_search``):

1. the default frame was the identity diagnostic (Moon-fixed field evaluated in
   the integration frame) and the gravity file actually used was not recorded;
2. ``--sensitivity-degree`` was not part of the resume contract;
3. a requested SH degree above the file's degree was clipped silently;
4. a screening step that does not divide the output interval was rounded
   silently while only the requested step was recorded;
5. stage-4 refinement ignored ``--hp-km`` / ``--argp-*`` and used ``--a-km``;
6. stage JSON files did not say which element frame ``i_deg``/``raan_deg`` use.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.analysis.frozen import search_backends as backends  # noqa: E402
from lunaris.analysis.frozen.search import (  # noqa: E402
    STAGE2_CANDIDATES,
    STAGE3_VALIDATION,
    STAGE4_FAMILIES,
    FrozenSearchPipeline,
)
from lunaris.cli import frozen_search as cli  # noqa: E402
from lunaris.common.frame_policy import (  # noqa: E402
    FRAME_MODE_IDENTITY_DIAGNOSTIC,
    FRAME_MODE_MOON_FIXED_EPHEMERIS,
)
from test_frozen_search_pipeline import (  # noqa: E402
    KeplerScreeningPropagator,
    KeplerValidationPropagator,
    _config,
)

EPHEMERIS_SENTINEL = object()


# ---------------------------------------------------------------------------
# CLI harness: real argument handling, fake backends and pipeline
# ---------------------------------------------------------------------------


class _FakeBackend:
    backend_name = "fake_backend"
    backend_label = "classical_sh_fake"
    gravity_mu_m3s2 = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.provenance: dict = {}


class _FakePipeline:
    instances: ClassVar[list[_FakePipeline]] = []

    def __init__(self, config, **kwargs) -> None:
        self.config = config
        self.kwargs = kwargs
        _FakePipeline.instances.append(self)

    def run(self, *, resume: bool = True) -> dict:
        return {"candidates": [], "validated": [], "family_report": {"families": []}}


@pytest.fixture
def cli_run(monkeypatch, tmp_path, capsys):
    from lunaris.analysis.frozen import search as search_mod

    _FakePipeline.instances = []
    monkeypatch.setattr(backends, "TorchSHScreeningPropagator", _FakeBackend)
    monkeypatch.setattr(backends, "ClassicalSHValidationPropagator", _FakeBackend)
    monkeypatch.setattr(
        backends,
        "build_ephemeris_manager_for_frozen_search",
        lambda **kw: EPHEMERIS_SENTINEL,
    )
    monkeypatch.setattr(search_mod, "FrozenSearchPipeline", _FakePipeline)

    def run(*extra: str) -> tuple[int, str, _FakePipeline | None]:
        rc = cli.main(["--out", str(tmp_path / "run"), "--no-figures", *extra])
        err = capsys.readouterr().err
        pipe = _FakePipeline.instances[-1] if _FakePipeline.instances else None
        return rc, err, pipe

    return run


# ---------------------------------------------------------------------------
# 1. strict frame is the default
# ---------------------------------------------------------------------------


def test_default_search_runs_in_the_moon_fixed_frame(cli_run) -> None:
    rc, _err, pipe = cli_run()
    assert rc == 0 and pipe is not None
    assert pipe.config.strict_frame is True
    assert pipe.kwargs["screening"].kwargs["frame_mode"] == FRAME_MODE_MOON_FIXED_EPHEMERIS
    assert pipe.kwargs["screening"].kwargs["ephem_manager"] is EPHEMERIS_SENTINEL
    assert pipe.kwargs["validation"].kwargs["frame_mode"] == FRAME_MODE_MOON_FIXED_EPHEMERIS
    assert pipe.kwargs["validation"].kwargs["ephem_manager"] is EPHEMERIS_SENTINEL


def test_no_strict_frame_opts_into_identity_with_a_warning(cli_run) -> None:
    rc, err, pipe = cli_run("--no-strict-frame")
    assert rc == 0 and pipe is not None
    assert pipe.config.strict_frame is False
    assert pipe.kwargs["screening"].kwargs["frame_mode"] == FRAME_MODE_IDENTITY_DIAGNOSTIC
    assert pipe.kwargs["validation"].kwargs["frame_mode"] == FRAME_MODE_IDENTITY_DIAGNOSTIC
    assert "WARNING" in err and "identity" in err


def test_strict_frame_default_prints_no_identity_warning(cli_run) -> None:
    _rc, err, _pipe = cli_run()
    assert "WARNING" not in err


# ---------------------------------------------------------------------------
# 1b. the gravity file actually used is recorded with path and sha256
# ---------------------------------------------------------------------------


def _write_shadr(path: Path, degree: int) -> Path:
    rows = [f"1738.0, 4902.8, 0, {degree}, {degree}, 1, 0, 0"]
    for n in range(1, degree + 1):
        for m in range(n + 1):
            c = 1.0 if (n, m) == (1, 0) else 1e-6 / (n + m + 1)
            rows.append(f"{n}, {m}, {c if n > 1 else 0.0}, 0.0")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_default_gravity_file_is_recorded_with_path_and_sha256(monkeypatch, tmp_path) -> None:
    import lunaris.core.config as config_mod

    path = _write_shadr(tmp_path / "default_model.tab", 4)
    monkeypatch.setattr(
        config_mod,
        "load_default_config",
        lambda: SimpleNamespace(gravity=SimpleNamespace(file_path=path)),
    )
    model = backends._load_gravity_model(None, 4)
    record = backends._gravity_provenance(model, 4, None)["gravity_model"]

    assert record["file"] == str(path)
    assert record["file"] != "default_config"
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["name"] != "unknown"


# ---------------------------------------------------------------------------
# 2. sensitivity set is part of the resume contract
# ---------------------------------------------------------------------------


def _pipeline(tmp_path, sensitivity_labels: list[str]) -> FrozenSearchPipeline:
    return FrozenSearchPipeline(
        _config(n_samples=8, top_k=2),
        screening=KeplerScreeningPropagator(),
        validation=KeplerValidationPropagator("classical_sh_deg8"),
        out_dir=tmp_path,
        sensitivity_validations=[KeplerValidationPropagator(label) for label in sensitivity_labels],
    )


def test_resume_refuses_a_changed_sensitivity_set(tmp_path) -> None:
    _pipeline(tmp_path, ["classical_sh_deg16"]).run(resume=True)

    with pytest.raises(RuntimeError, match="cannot resume"):
        _pipeline(tmp_path, ["classical_sh_deg32"]).run(resume=True)
    with pytest.raises(RuntimeError, match="cannot resume"):
        _pipeline(tmp_path, []).run(resume=True)


def test_resume_accepts_an_unchanged_sensitivity_set(tmp_path) -> None:
    _pipeline(tmp_path, ["classical_sh_deg16"]).run(resume=True)
    _pipeline(tmp_path, ["classical_sh_deg16"]).run(resume=True)


def test_resume_tolerates_a_manifest_written_before_sensitivity_was_recorded(tmp_path) -> None:
    _pipeline(tmp_path, ["classical_sh_deg16"]).run(resume=True)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("sensitivity", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _pipeline(tmp_path, ["classical_sh_deg16"]).run(resume=True)


# ---------------------------------------------------------------------------
# 3. a degree above the file's degree is refused, not clipped
# ---------------------------------------------------------------------------


def test_requested_degree_above_the_file_degree_raises(tmp_path) -> None:
    path = _write_shadr(tmp_path / "deg4.tab", 4)

    assert backends._load_gravity_model(str(path), 4).degree_max == 4
    with pytest.raises(ValueError, match=r"degree 6.*degree 4"):
        backends._load_gravity_model(str(path), 6)


# ---------------------------------------------------------------------------
# 4. screening step must divide the output interval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_cls", [backends.TorchSHScreeningPropagator, backends.STLRPSScreeningPropagator])
def test_screening_propagate_refuses_a_step_that_does_not_divide_the_output_dt(backend_cls) -> None:
    prop = object.__new__(backend_cls)
    prop._dt_s = 70.0
    prop._propagator = SimpleNamespace(
        propagate=lambda *a, **k: pytest.fail("propagation must not start")
    )
    with pytest.raises(ValueError, match="does not divide"):
        prop.propagate(np.zeros((2, 6)), 7 * 86_400.0, 3_600.0)


def test_screening_propagate_accepts_a_dividing_step() -> None:
    called = []
    prop = object.__new__(backends.TorchSHScreeningPropagator)
    prop._dt_s = 30.0
    prop._propagator = SimpleNamespace(propagate=lambda *a, **k: called.append(1) or (1, 2, 3, 4))
    prop.propagate(np.zeros((2, 6)), 7 * 86_400.0, 3_600.0)
    assert called == [1]


def test_cli_refuses_a_screening_dt_that_does_not_divide_the_output_dt(cli_run) -> None:
    rc, err, pipe = cli_run("--screening-dt-s", "70", "--screening-output-dt-s", "3600")
    assert rc == 2 and pipe is None
    assert "[FATAL]" in err and "divide" in err


# ---------------------------------------------------------------------------
# 5. refinement bounds come from the search's own element bounds
# ---------------------------------------------------------------------------


def test_refinement_bounds_follow_hp_and_argp_choices(cli_run) -> None:
    rc, _err, pipe = cli_run(
        "--hp-km", "100", "200", "--e", "0.01", "0.1",
        "--argp-choices", "90,270", "--refine-top-n", "2",
    )
    assert rc == 0 and pipe is not None
    search_bounds = pipe.config.bounds
    rb = pipe.config.refinement.bounds
    assert tuple(rb.a_km) == tuple(search_bounds.a_km)
    assert tuple(rb.e) == tuple(search_bounds.e)
    assert tuple(rb.i_deg) == tuple(search_bounds.i_deg)
    assert tuple(rb.argp_deg) == (90.0, 270.0)
    # --a-km default (1838, 2238) must not leak into a perilune-band search
    assert tuple(rb.a_km) != (1_838.0, 2_238.0)


def test_refinement_bounds_follow_argp_deg(cli_run) -> None:
    rc, _err, pipe = cli_run("--argp-deg", "80", "100", "--refine-top-n", "1")
    assert rc == 0 and pipe is not None
    assert tuple(pipe.config.refinement.bounds.argp_deg) == (80.0, 100.0)


# ---------------------------------------------------------------------------
# 6. element frame is stated in every stage JSON
# ---------------------------------------------------------------------------


def test_stage_json_files_state_the_element_frame(tmp_path) -> None:
    pipe = _pipeline(tmp_path, [])
    pipe.run(resume=True)
    expected = pipe.element_frame["frame"]
    for name in (STAGE2_CANDIDATES, STAGE3_VALIDATION, STAGE4_FAMILIES):
        payload = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        assert payload["element_frame"] == expected, name
