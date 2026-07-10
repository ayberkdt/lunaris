"""Shared batch output time-grid contract (reviewer §5).

``build_batch_output_grid`` is the single source of truth for the snapshot grid
used by every batch backend (CPU DOP853, Numba CUDA, torch_cuda_sh / torch_cpu_sh,
ST-LRPS torch), so trajectories are index-comparable across backends. These
tests lock the contract independently of any propagator / torch / CUDA.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lunaris.batch.engine import _batch_timestep_provenance, _force_model_fidelity_provenance
from lunaris.batch.provenance import build_degraded_batch_backend_metadata
from lunaris.common.batch_defs import (
    build_batch_output_grid,
    build_fixed_step_grid_metadata,
)
from lunaris.common.type_defs import PerturbationFlags


def test_grid_contract_divisible() -> None:
    t, n_snaps, snap = build_batch_output_grid(1200.0, 300.0)
    assert t[0] == 0.0
    assert t[-1] == 1200.0
    assert n_snaps == 4
    assert len(t) == n_snaps + 1
    assert np.all(np.diff(t) > 0.0)
    assert snap == pytest.approx(300.0)


def test_grid_exact_final_when_not_divisible() -> None:
    # 1000 / 600 -> round() = 2 snaps. The last sample must land exactly on the
    # requested duration (1000 s), never overshoot to 1200 s as the old
    # ``n_snaps * out_dt`` endpoint did.
    t, n_snaps, snap = build_batch_output_grid(1000.0, 600.0)
    assert n_snaps == 2
    assert t[0] == 0.0
    assert t[-1] == 1000.0
    assert np.all(np.diff(t) > 0.0)
    assert snap == pytest.approx(500.0)


def test_grid_minimum_one_snapshot() -> None:
    # output_dt larger than duration still yields a valid 2-point grid.
    t, n_snaps, _ = build_batch_output_grid(100.0, 10_000.0)
    assert n_snaps == 1
    assert list(t) == [0.0, 100.0]


def test_grid_realized_spacing_never_exceeds_request() -> None:
    t, n_snaps, snap = build_batch_output_grid(149.0, 100.0)
    assert n_snaps == 2
    assert snap == pytest.approx(74.5)
    assert np.max(np.diff(t)) <= 100.0


def test_fixed_step_metadata_records_requested_and_effective_dt() -> None:
    meta = build_fixed_step_grid_metadata(
        duration_s=1000.0,
        output_dt_s=600.0,
        dt_s=60.0,
    )

    assert meta["requested_dt_s"] == pytest.approx(60.0)
    assert meta["effective_output_dt_s"] == pytest.approx(500.0)
    assert meta["steps_per_snapshot"] == 8
    assert meta["effective_dt_s"] == pytest.approx(62.5)
    assert meta["n_output_snapshots"] == 3


def test_engine_timestep_provenance_marks_fixed_step_adjustment() -> None:
    meta = _batch_timestep_provenance(
        SimpleNamespace(dt_s=60.0),
        duration_s=1000.0,
        output_dt_s=600.0,
        actual_backend="numba_cuda_sh",
    )

    assert meta["fixed_step_grid_aligned"] is True
    assert meta["requested_dt_s"] == pytest.approx(60.0)
    assert meta["effective_dt_s"] == pytest.approx(62.5)
    assert meta["steps_per_snapshot"] == 8


def test_force_model_fidelity_provenance_labels_engineering_scope() -> None:
    cfg = SimpleNamespace(
        flags=PerturbationFlags(
            enable_srp=True,
            enable_albedo=True,
            enable_thermal=True,
            enable_relativity_1pn=True,
        ),
        srp=SimpleNamespace(enable_moon_eclipse=True, enable_earth_eclipse=True),
        albedo=SimpleNamespace(albedo_model="lambert_facets", enable_eclipse=True),
        thermal=SimpleNamespace(
            thermal_mode="equilibrium_temperature",
            enable_eclipse=True,
        ),
    )
    meta = _force_model_fidelity_provenance(
        cfg,
        backend_diag={
            "srp_shadow_model": "cylindrical_moon_umbra_no_earth_eclipse",
            "srp_shadow_model_fidelity": "reduced_gpu_approximation",
            "srp_earth_eclipse_supported": False,
        },
    )

    assert meta["srp_force_model"] == "cannonball_cr_area_over_mass"
    assert meta["srp_attitude_model"] == "none"
    assert meta["srp_shadow_model"] == "cylindrical_moon_umbra_no_earth_eclipse"
    assert meta["relativity_model"] == "selected_1pn_corrections"
    assert "full_eih_n_body" in meta["relativity_excluded_terms"]
    assert meta["albedo_eclipse_fidelity"] == "global_moon_center_proxy_not_per_facet"
    assert (
        meta["thermal_ir_eclipse_model"]
        == "moon_center_global_earth_shadow_factor_on_solar_input"
    )
    assert meta["thermal_ir_model_fidelity"] == "engineering_approximation"


def test_degraded_backend_provenance_never_copies_request_as_actual() -> None:
    plan = SimpleNamespace(
        requested_backend="numba_cuda_sh",
        actual_backend="cpu_sh",
        final_backend=SimpleNamespace(value="cpu"),
    )

    meta = build_degraded_batch_backend_metadata(
        requested_backend="numba_cuda_sh",
        backend_plan=plan,
        backend_diagnostics={},
        requested_sh_degree=80,
        error=RuntimeError("metadata enrichment failed"),
    )

    assert meta["requested_batch_backend"] == "numba_cuda_sh"
    assert meta["actual_batch_backend"] == "cpu_sh"
    assert meta["batch_backend"] == "cpu"
    assert meta["provenance_status"] == "degraded"
    assert meta["provenance_error_type"] == "RuntimeError"


def test_degraded_backend_provenance_marks_unknown_actual_backend() -> None:
    meta = build_degraded_batch_backend_metadata(
        requested_backend="numba_cuda_sh",
        backend_plan=None,
        backend_diagnostics={},
        requested_sh_degree=80,
        error=RuntimeError("metadata enrichment failed"),
    )

    assert meta["requested_batch_backend"] == "numba_cuda_sh"
    assert meta["actual_batch_backend"] == "unknown"
    assert meta["batch_backend"] == "unknown"


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_grid_rejects_nonpositive_duration(bad: float) -> None:
    with pytest.raises(ValueError):
        build_batch_output_grid(bad, 60.0)


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_grid_rejects_nonpositive_output_dt(bad: float) -> None:
    with pytest.raises(ValueError):
        build_batch_output_grid(1000.0, bad)
