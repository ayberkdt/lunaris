"""A UTC start epoch before 1972 must be refused, not silently mis-converted.

Measured with naif0012.tls: ``str2et`` applies ET-UTC = 41.184 s (TAI-UTC = 9 s)
to every pre-1972 UTC string, e.g. 1969-07-20T20:17:40, whose historical TAI-UTC
was 7.5746 s (USNO tai-utc.dat: 4.21317 + (MJD - 39126) * 0.002592) -- a 1.43 s
epoch error with no warning.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("spiceypy")

from lunaris.physics.ephemeris import _require_utc_after_1972  # noqa: E402

_ET_1972 = -883655957.8160794  # str2et("1972-01-01T00:00:00") with naif0012.tls


def _str2et(_s: str) -> float:
    return _ET_1972


def test_pre_1972_utc_is_refused() -> None:
    with pytest.raises(ValueError, match="before 1972"):
        _require_utc_after_1972("1969-07-20T20:17:40Z", -960910898.8164499, _str2et)


def test_tdb_or_tt_and_post_1972_utc_pass() -> None:
    _require_utc_after_1972("1969-07-20T20:17:40 TDB", -960910898.8, _str2et)
    _require_utc_after_1972("1969 JUL 20 20:17:40 TT", -960910898.8, _str2et)
    _require_utc_after_1972("1972-01-01T00:00:00Z", _ET_1972, _str2et)
    _require_utc_after_1972("2025-01-01T00:00:00Z", 788961669.18, _str2et)


_LSK = Path(r"D:\Masaustu\LUNAR_SIMULATION\data\ephemeris_models\naif0012.tls")


@pytest.mark.requires_data
@pytest.mark.skipif(not _LSK.exists(), reason="naif0012.tls not available")
def test_real_lsk_reference_epoch() -> None:
    import spiceypy as spice

    spice.furnsh(str(_LSK))
    try:
        assert spice.str2et("1972-01-01T00:00:00") == pytest.approx(_ET_1972, abs=1e-6)
        with pytest.raises(ValueError, match="before 1972"):
            _require_utc_after_1972("1971-12-31T23:59:59", spice.str2et("1971-12-31T23:59:59"), spice.str2et)
    finally:
        spice.unload(str(_LSK))
