"""Optional second independent SH reference backed by ``pyshtools``.

``pyshtools`` is a mature external geodesy library with a completely separate
spherical-harmonic implementation, so it makes a strong *third opinion* next to
:mod:`validation.independent.independent_sh` (scipy + numerical gradient) and the
Lunaris analytic recurrence under test.

It is an OPTIONAL dependency: ``pyshtools`` ships compiled Fortran and is awkward
to install on some platforms (notably Windows), so importing this module never
hard-fails. Callers check :data:`PYSHTOOLS_AVAILABLE` or call
:func:`require_pyshtools`. The accompanying test is gated behind the
``requires_pyshtools`` marker and skips cleanly when the library is absent.

Convention note (pinned by the gated test, not assumed)
-------------------------------------------------------
Lunaris stores 4π fully-normalized coefficients whose ALFs carry NO
Condon–Shortley phase (geodesy/GRAIL convention; verified in
:mod:`validation.independent.independent_sh`). The matching pyshtools settings
are ``normalization='4pi'`` and ``csphase=1`` (no phase), which is pyshtools'
default for ``MakeGravGridPoint``. The gated cross-check compares pyshtools
against the scipy reference and will fail loudly if these flags are wrong on a
given pyshtools release, at which point they should be corrected here.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised only where pyshtools is installed
    import pyshtools as _pysh  # type: ignore

    PYSHTOOLS_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure means "unavailable"
    _pysh = None  # type: ignore
    PYSHTOOLS_AVAILABLE = False

__all__ = [
    "PYSHTOOLS_AVAILABLE",
    "require_pyshtools",
    "acceleration",
]


def require_pyshtools() -> None:
    """Raise a clear error if ``pyshtools`` is not importable."""
    if not PYSHTOOLS_AVAILABLE:
        raise ModuleNotFoundError(
            "pyshtools is not installed. It is an optional validation dependency; "
            "install it (e.g. `conda install -c conda-forge pyshtools`) to enable "
            "the second independent spherical-harmonic cross-check."
        )


def acceleration(
    position: np.ndarray,
    *,
    mu: float,
    r_ref: float,
    c_coeffs: np.ndarray,
    s_coeffs: np.ndarray,
    degree: int | None = None,
) -> np.ndarray:  # pragma: no cover - requires the optional pyshtools dependency
    """Body-fixed gravitational acceleration ``a = +∇U`` via ``pyshtools``.

    ``c_coeffs``/``s_coeffs`` are ``(N+1, N+1)`` 4π-normalized matrices in the
    Lunaris convention (no Condon–Shortley phase, matching geodesy/GRAIL and
    pyshtools' default). The result is returned in Cartesian body-fixed
    components (m/s²) to match the Lunaris :meth:`GravityModel.accel_fixed`
    output.
    """
    require_pyshtools()

    pos = np.asarray(position, dtype=np.float64).reshape(3)
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    r = float(np.linalg.norm(pos))
    if r <= 0.0:
        raise ValueError("position must be non-zero")

    c = np.asarray(c_coeffs, dtype=np.float64)
    s = np.asarray(s_coeffs, dtype=np.float64)
    n_avail = min(c.shape[0], s.shape[0]) - 1
    lmax = n_avail if degree is None else min(int(degree), n_avail)

    # pyshtools 4π real coefficients in cilm[2, l+1, m+1] layout: [0]=cosine (C),
    # [1]=sine (S). Lunaris stores the same C/S in (l, m) matrices.
    cilm = np.zeros((2, lmax + 1, lmax + 1), dtype=np.float64)
    cilm[0, : lmax + 1, : lmax + 1] = c[: lmax + 1, : lmax + 1]
    cilm[1, : lmax + 1, : lmax + 1] = s[: lmax + 1, : lmax + 1]

    # Geocentric latitude/longitude in degrees, as pyshtools expects.
    lat_deg = float(np.degrees(np.arcsin(z / r)))
    lon_deg = float(np.degrees(np.arctan2(y, x)))

    # MakeGravGridPoint uses 4pi-normalized coefficients WITHOUT the
    # Condon-Shortley phase (csphase=1), which is exactly the Lunaris/GRAIL
    # convention, so the coefficients are passed through unchanged.

    # MakeGravGridPoint returns the spherical gravity components (g_r, g_theta,
    # g_phi) of -∇U; we negate to report a = +∇U and rotate to Cartesian.
    g_r, g_theta, g_phi = _pysh.gravmag.MakeGravGridPoint(
        cilm,
        gm=float(mu),
        r0=float(r_ref),
        r=r,
        lat=lat_deg,
        lon=lon_deg,
        lmax=lmax,
        omega=0.0,
    )
    # MakeGravGridPoint documentation states "the gravitational acceleration is B = Grad V"
    # so it already returns +∇U (plus centrifugal, which is 0 here since omega=0).
    colat = np.radians(90.0 - lat_deg)
    lam = np.radians(lon_deg)
    st, ct = np.sin(colat), np.cos(colat)
    sl, cl = np.sin(lam), np.cos(lam)
    # Unit vectors of the spherical basis in Cartesian coordinates.
    e_r = np.array([st * cl, st * sl, ct])
    e_theta = np.array([ct * cl, ct * sl, -st])
    e_phi = np.array([-sl, cl, 0.0])
    g_vec = g_r * e_r + g_theta * e_theta + g_phi * e_phi
    return g_vec
