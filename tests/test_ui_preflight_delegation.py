"""
The desktop pre-flight worker delegates to the shared core.preflight service.

These lock the §5 wiring: the UI maps its snapshot to a registry backend +
flags, and reaches its output-directory and backend verdicts through the same
``lunaris.core.preflight`` service the CLI / batch / benchmark use. CPU-only; the
desktop mission-propagation path is the complete supported-force CPU backend.
"""

from __future__ import annotations

import pytest
from tests.ui_qt_helpers import QtCore, QtGui, QtWidgets

torch = pytest.importorskip('torch')


from lunaris.core.preflight import check_backend_capability  # noqa: E402
from lunaris.ui.core.preflight_validation import (  # noqa: E402
    PreFlightWorker,
    backend_request_from_snapshot,
)

# ---------------------------------------------------------------------------
# Snapshot -> registry request mapping (pure, Qt-free)
# ---------------------------------------------------------------------------

def test_snapshot_maps_classic_to_cpu_sh() -> None:
    backend, flags = backend_request_from_snapshot(
        {"gravity_backend": "classic_sh", "albedo_enabled": True}
    )
    assert backend == "cpu_sh"
    assert flags.enable_albedo is True
    assert flags.enable_sh is False  # gravity_enabled not present in snapshot


def test_snapshot_maps_surrogate_to_cpu_st_lrps() -> None:
    backend, _flags = backend_request_from_snapshot({"gravity_backend": "st_lrps"})
    assert backend == "cpu_st_lrps"


def test_snapshot_translates_force_flags() -> None:
    snap = {
        "gravity_backend": "classic_sh",
        "gravity_enabled": True,
        "sun_enabled": True,
        "earth_j2_enabled": True,
        "relativity_1pn_enabled": True,
    }
    _backend, flags = backend_request_from_snapshot(snap)
    assert flags.enable_sh and flags.enable_3rd_body_sun
    assert flags.enable_earth_j2 and flags.enable_relativity_1pn


def test_cpu_request_passes_capability_check() -> None:
    backend, flags = backend_request_from_snapshot(
        {"gravity_backend": "classic_sh", "albedo_enabled": True, "earth_j2_enabled": True}
    )
    # The CPU route supports every currently implemented force flag.
    assert check_backend_capability(requested_backend=backend, flags=flags) == []


# ---------------------------------------------------------------------------
# Worker methods delegate to the shared service (CPU happy path → no signals)
# ---------------------------------------------------------------------------

def _worker(command_data: dict) -> PreFlightWorker:
    # Bypass QThread.__init__: the methods under test only read command_data and
    # call pure shared-service functions. The CPU happy path emits no signals.
    worker = PreFlightWorker.__new__(PreFlightWorker)
    worker.command_data = dict(command_data)
    return worker


def test_worker_output_directory_delegates(tmp_path) -> None:
    worker = _worker({"output_dir": str(tmp_path / "run")})
    ok, _msg = worker._validate_output_directory()
    assert ok is True
    assert (tmp_path / "run").is_dir()


def test_worker_output_directory_empty_fails() -> None:
    worker = _worker({"output_dir": ""})
    ok, _msg = worker._validate_output_directory()
    assert ok is False


def test_worker_backend_capability_passes_for_cpu() -> None:
    worker = _worker({"gravity_backend": "classic_sh", "albedo_enabled": True, "earth_j2_enabled": True})
    ok, _msg = worker._validate_backend_capability()
    assert ok is True


def _finite_numeric_command() -> dict:
    return {
        "mass_kg": 1000.0,
        "area_m2": 5.0,
        "cd": 2.2,
        "cr": 1.5,
        "rtol": 1e-12,
        "atol": 1e-14,
        "duration_val": 10.0,
    }


def test_numeric_ranges_pass_for_finite_positive_values() -> None:
    worker = _worker(_finite_numeric_command())
    ok, _msg = worker._validate_numeric_ranges()
    assert ok is True


@pytest.mark.parametrize("field", ["rtol", "atol", "duration_val", "mass_kg"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_numeric_ranges_reject_non_finite(field: str, bad: float) -> None:
    # nan/inf slip past a bare ``<= 0`` gate; the finite guard must catch them so
    # the UI matches the core config contract (np.isfinite rejection).
    command = _finite_numeric_command()
    command[field] = bad
    worker = _worker(command)
    ok, msg = worker._validate_numeric_ranges()
    assert ok is False
    assert "finite" in msg.lower()


def test_numeric_ranges_reject_non_finite_max_step() -> None:
    command = _finite_numeric_command()
    command["max_step"] = float("inf")
    worker = _worker(command)
    ok, msg = worker._validate_numeric_ranges()
    assert ok is False
    assert "finite" in msg.lower()
