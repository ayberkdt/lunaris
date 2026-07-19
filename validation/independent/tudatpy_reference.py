"""Optional independent SH reference backed by ``tudatpy`` (TU Delft Tudat).

``tudatpy`` is an industry/academic-grade astrodynamics toolkit with a completely
separate spherical-harmonic gravity implementation (C++ core). It is the
strongest available *independent* cross-check for the Lunaris gravity engine —
stronger than a second pure-Python library — because it shares neither code nor
language runtime with Lunaris.

Status
------
**Scaffold, pending first-run verification.** ``tudatpy`` is a heavy conda-only
dependency and is not installed in CI or in the reference dev environment, so the
exact point-acceleration API below has NOT yet been executed here. It follows the
documented ``environment_setup`` gravity-field interface and is written to fail
*loudly* (clear, actionable error) if the installed tudatpy version exposes a
different method, rather than silently returning wrong numbers. The accompanying
test is gated behind the ``requires_tudatpy`` marker and skips cleanly when the
library is absent.

This status applies only to this legacy point-gradient adapter. The separately
pinned rotating-trajectory harness under
``validation/gravity_reference/generators/trajectory/tudatpy_rotating`` was
executed with TudatPy 1.0.0 for one-, five-, and thirty-day arcs; do not conflate
that verified trajectory evidence with this still-unverified convenience API.

Install (conda-forge):  ``conda install -c tudat-team tudatpy``

Convention
----------
Body-fixed Cartesian ``(x, y, z)``; 4pi-normalized coefficients with NO
Condon-Shortley phase (geodesy/GRAIL, matching Lunaris and pyshtools). Output is
``a = +grad U`` in body-fixed Cartesian components (m/s^2), matching
:meth:`lunaris.physics.spherical_harmonics.GravityModel.accel_fixed`.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised only where tudatpy is installed
    import tudatpy  # type: ignore  # noqa: F401

    TUDATPY_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure means "unavailable"
    TUDATPY_AVAILABLE = False

__all__ = ["TUDATPY_AVAILABLE", "require_tudatpy", "acceleration"]


def require_tudatpy() -> None:
    """Raise a clear error if ``tudatpy`` is not importable."""
    if not TUDATPY_AVAILABLE:
        raise ModuleNotFoundError(
            "tudatpy is not installed. It is an optional, industry-grade validation "
            "dependency (conda-forge: `conda install -c tudat-team tudatpy`). Install "
            "it to enable the independent tudatpy spherical-harmonic cross-check."
        )


def acceleration(
    position: np.ndarray,
    *,
    mu: float,
    r_ref: float,
    c_coeffs: np.ndarray,
    s_coeffs: np.ndarray,
    degree: int | None = None,
) -> np.ndarray:  # pragma: no cover - requires the optional tudatpy dependency
    """Body-fixed gravitational acceleration ``a = +grad U`` via ``tudatpy``.

    ``c_coeffs``/``s_coeffs`` are ``(N+1, N+1)`` 4pi-normalized matrices in the
    Lunaris convention (no Condon-Shortley phase). Returns Cartesian body-fixed
    components (m/s^2).

    Raises a clear, actionable error if the installed tudatpy version does not
    expose the expected gravity-field gradient method, so a version mismatch can
    never masquerade as a numerical disagreement.
    """
    require_tudatpy()

    pos = np.asarray(position, dtype=np.float64).reshape(3)
    c = np.asarray(c_coeffs, dtype=np.float64)
    s = np.asarray(s_coeffs, dtype=np.float64)
    n_avail = min(c.shape[0], s.shape[0]) - 1
    lmax = n_avail if degree is None else min(int(degree), n_avail)
    cosine = c[: lmax + 1, : lmax + 1].copy()
    sine = s[: lmax + 1, : lmax + 1].copy()

    try:
        from tudatpy.numerical_simulation import environment_setup
        from tudatpy.numerical_simulation.environment_setup import gravity_field

        # Build a minimal body carrying only the SH gravity field, in a fixed
        # (non-rotating) body frame so the body-fixed position maps directly.
        field_settings = gravity_field.spherical_harmonic(
            gravitational_parameter=float(mu),
            reference_radius=float(r_ref),
            normalized_cosine_coefficients=cosine,
            normalized_sine_coefficients=sine,
            associated_reference_frame="IAU_Moon",
        )
        body_settings = environment_setup.get_default_body_settings(["Moon"])
        body_settings.get("Moon").gravity_field_settings = field_settings
        bodies = environment_setup.create_system_of_bodies(body_settings)
        grav_model = bodies.get("Moon").gravity_field_model

        # Acceleration is the gradient of the potential in the body-fixed frame.
        grad = np.asarray(
            grav_model.get_gradient_of_potential(pos), dtype=np.float64
        ).reshape(3)
        return grad
    except AttributeError as exc:
        raise NotImplementedError(
            "The installed tudatpy version does not expose the expected "
            "gravity-field point-gradient API used here "
            f"({exc}). Confirm the correct method for your tudatpy release "
            "(e.g. gravity_field_model.get_gradient_of_potential) and update "
            "validation/independent/tudatpy_reference.py before trusting this "
            "cross-check."
        ) from exc
