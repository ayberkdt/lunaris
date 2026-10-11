"""History extractors read the state layout from the time grid and trust SI units.

* A row-major ``(N, 6)`` state with ``N >= 6`` used to be read as ``(6, N)``,
  so ``r``/``v`` (and the derived elements) were rows of the wrong axis.
* Units were guessed from magnitudes: a canonical ``r_norm_km`` above 1e5 km
  (an orbit reaching beyond ~100 000 km) was divided by 1000 again.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.analysis.postprocess import (
    extract_elements,
    extract_invariants,
    extract_rv_vectors,
)
from lunaris.common.constants import MU_MOON


def _circular_states(n: int, r_m: float = 1_838_000.0) -> tuple[np.ndarray, np.ndarray]:
    """Row-major (N, 6) states on a circular equatorial orbit, SI units."""
    v = math.sqrt(MU_MOON / r_m)
    period = 2.0 * math.pi * math.sqrt(r_m**3 / MU_MOON)
    t = np.linspace(0.0, period / 2.0, n)
    th = 2.0 * math.pi * t / period
    y = np.column_stack(
        [r_m * np.cos(th), r_m * np.sin(th), np.zeros(n),
         -v * np.sin(th), v * np.cos(th), np.zeros(n)]
    )
    return t, y


@pytest.mark.parametrize("n", [3, 10])
def test_row_major_and_column_major_states_give_the_same_vectors(n) -> None:
    t, y_rows = _circular_states(n)
    r_rows, v_rows = extract_rv_vectors({"t_s": t, "y": y_rows})
    r_cols, v_cols = extract_rv_vectors({"t_s": t, "y": y_rows.T})

    np.testing.assert_array_equal(r_rows, y_rows[:, :3])
    np.testing.assert_array_equal(v_rows, y_rows[:, 3:6])
    np.testing.assert_array_equal(r_cols, y_rows[:, :3])
    np.testing.assert_array_equal(v_cols, y_rows[:, 3:6])


def test_square_state_follows_the_canonical_column_layout() -> None:
    # (6, 6) with six epochs matches both axes; compute_history's canonical
    # layout is (6, N), so columns are epochs.
    t, y_rows = _circular_states(6)
    r, v = extract_rv_vectors({"t_s": t, "y": y_rows.T})
    np.testing.assert_array_equal(r, y_rows[:, :3])
    np.testing.assert_array_equal(v, y_rows[:, 3:6])


def test_layout_that_matches_no_time_axis_is_refused() -> None:
    t, y_rows = _circular_states(10)
    with pytest.raises(ValueError, match="time grid"):
        extract_rv_vectors({"t_s": t[:4], "y": y_rows})


def test_elements_from_row_major_states_are_correct() -> None:
    t, y_rows = _circular_states(10)
    elems = extract_elements({"t_s": t, "y": y_rows, "mu_m3s2": float(MU_MOON)})
    np.testing.assert_allclose(elems["a_km"], 1838.0, rtol=1e-12)
    np.testing.assert_allclose(elems["e"], 0.0, atol=1e-12)


def test_canonical_km_norms_are_not_rescaled() -> None:
    # An orbit out to 150 000 km: the canonical km key is already in km.
    r_km = np.array([150_000.0, 149_000.0, 148_000.0])
    v_kmps = np.array([0.04, 0.03, 0.02])
    inv = extract_invariants({"r_norm_km": r_km, "v_norm_kmps": v_kmps})
    np.testing.assert_array_equal(inv["r_norm_km"], r_km)
    np.testing.assert_array_equal(inv["v_norm_kmps"], v_kmps)


def test_si_vectors_are_not_rescaled_by_magnitude() -> None:
    # A slow, distant arc in SI units: 30 m/s speeds are m/s, not km/s.
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([[9.0e7, 0.0, 0.0, 0.0, 30.0, 0.0]] * 3)
    r, v = extract_rv_vectors({"t_s": t, "y": y})
    np.testing.assert_array_equal(v[:, 1], 30.0)
    inv = extract_invariants({"t_s": t, "y": y})
    np.testing.assert_allclose(inv["r_norm_km"], 9.0e4)
    np.testing.assert_allclose(inv["v_norm_kmps"], 0.03)
