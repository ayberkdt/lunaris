"""
The desktop pre-flight worker delegates to the shared core.preflight service.

These lock the §5 wiring: the UI maps its snapshot to a registry backend +
flags, and reaches its output-directory and backend verdicts through the same
``lunaris.core.preflight`` service the CLI / batch / benchmark use. CPU-only; the
desktop mission-propagation path is the CPU full-fidelity backend.
"""

from __future__ import annotations

import pytest

pytest.importorskip('PySide6.QtWidgets')


import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

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
    # CPU full-fidelity supports every force model → no capability issues.
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
