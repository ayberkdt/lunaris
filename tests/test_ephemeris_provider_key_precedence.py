"""The canonical ephemeris provider keys win over the legacy ones.

EphemerisManager.get_data_provider() writes only the canonical keys (dt_s,
r_sun_tab_m, r_earth_tab_m, q_i2f_tab). extract_ephem_tables_strict let the
legacy keys (dt, sun_table, earth_table, rot_table) override them, so a
provider carrying both flew the legacy tables.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.core.dynamics import extract_ephem_tables_strict


class _Provider:
    def __init__(self, data):
        self._data = data

    def get_data_provider(self):
        return self._data


def _tables(scale: float, n: int = 4) -> dict[str, object]:
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return {
        "dt": 60.0 * scale,
        "sun": np.full((n, 3), 1.5e11 * scale),
        "earth": np.full((n, 3), 3.8e8 * scale),
        "q": q * (1.0 if scale == 1.0 else -1.0),
    }


def _canonical(t):
    return {"dt_s": t["dt"], "r_sun_tab_m": t["sun"], "r_earth_tab_m": t["earth"], "q_i2f_tab": t["q"]}


def _legacy(t):
    return {"dt": t["dt"], "sun_table": t["sun"], "earth_table": t["earth"], "rot_table": t["q"]}


def test_canonical_keys_win_when_both_naming_sets_are_present() -> None:
    canonical, legacy = _tables(1.0), _tables(2.0)
    dt_s, sun, earth, q = extract_ephem_tables_strict(_Provider({**_legacy(legacy), **_canonical(canonical)}))
    assert dt_s == canonical["dt"]
    np.testing.assert_array_equal(sun, canonical["sun"])
    np.testing.assert_array_equal(earth, canonical["earth"])
    np.testing.assert_array_equal(q, canonical["q"])


@pytest.mark.parametrize("make", [_canonical, _legacy])
def test_either_naming_set_alone_is_still_read(make) -> None:
    t = _tables(1.0)
    dt_s, sun, earth, q = extract_ephem_tables_strict(_Provider(make(t)))
    assert dt_s == t["dt"]
    np.testing.assert_array_equal(sun, t["sun"])
    np.testing.assert_array_equal(earth, t["earth"])
    np.testing.assert_array_equal(q, t["q"])
