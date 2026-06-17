"""Independent external-reference validation harnesses.

This subpackage exists to break the *correlated-error* problem: the in-tree
gravity/ephemeris validation reuses Lunaris' own code, so a bug in that code can
hide inside both the model and its "validation". The harnesses here compute
reference values through deliberately independent code paths:

* :mod:`validation.independent.independent_sh` evaluates the geopotential with a
  separate associated-Legendre implementation (``scipy.special``) and takes the
  acceleration as the *numerical gradient* of that potential — a completely
  different algorithm from the analytic Numba recurrence in
  :mod:`lunaris.physics.spherical_harmonics`.
* :mod:`validation.independent.pyshtools_reference` provides an optional second
  cross-check against the external ``pyshtools`` geodesy library.
* :mod:`validation.independent.naif_ephemeris` queries SPICE directly (bypassing
  the Lunaris ephemeris wrapper) for a frame/target sanity cross-check.

These modules MUST NOT import the Lunaris SH/ephemeris *math* to build a
reference value; they may import the Lunaris object under test only so the
cross-check can compare against it.
"""

from __future__ import annotations

__all__ = [
    "independent_sh",
    "pyshtools_reference",
    "naif_ephemeris",
]
