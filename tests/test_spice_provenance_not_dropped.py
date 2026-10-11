"""SPICE kernel provenance must not vanish silently from run manifests.

The batch archive and single-run diagnostics wrapped
``ephem.kernel_provenance()`` in ``contextlib.suppress(Exception)``: any failure
removed the ``spice_kernels`` key with no reason and no warning, and an
ephemeris archive without ``kernel_provenance_json`` reported ``kernels: []``,
which reads as "built from no kernels".
"""

from __future__ import annotations

import types

import numpy as np
import pytest

pytest.importorskip("spiceypy")

from lunaris.physics.ephemeris import EphemerisManager, spice_provenance_fields  # noqa: E402


class _Boom:
    def kernel_provenance(self):
        raise RuntimeError("tables not built")


class _Ok:
    def kernel_provenance(self):
        return {"kernels": [{"name": "naif0012.tls"}]}


def test_failure_is_recorded_and_warned_not_dropped() -> None:
    with pytest.warns(RuntimeWarning, match="SPICE kernel provenance unavailable"):
        out = spice_provenance_fields(_Boom())
    assert out == {"spice_kernels_unavailable_reason": "RuntimeError: tables not built"}


def test_success_and_absent_manager() -> None:
    assert spice_provenance_fields(_Ok()) == {"spice_kernels": {"kernels": [{"name": "naif0012.tls"}]}}
    assert spice_provenance_fields(None) == {}
    assert spice_provenance_fields(object()) == {}


def test_empty_kernel_list_carries_a_reason() -> None:
    tables = types.SimpleNamespace(
        et0=0.0, t_tab_s=np.array([0.0, 60.0]), kernel_provenance=(),
        time_scale_note="", inertial_frame="J2000", fixed_frame="MOON_ME",
        observer="MOON", schema_version=2, aberration_correction="NONE",
        interpolation_kind="cubic_hermite_position_velocity",
    )
    fake = types.SimpleNamespace(tables=tables)
    prov = EphemerisManager.kernel_provenance(fake)  # type: ignore[arg-type]
    assert prov["kernels"] == []
    assert "kernel_provenance_json" in prov["kernels_unavailable_reason"]

    tables.kernel_provenance = ({"name": "de440.bsp"},)
    prov = EphemerisManager.kernel_provenance(fake)  # type: ignore[arg-type]
    assert "kernels_unavailable_reason" not in prov
