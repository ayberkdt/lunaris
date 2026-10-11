"""Alignment between the Moon-fixed propagation frame and lunar surface-map frames.

The SPICE attitude tables (``q_i2f``) and the lunar gravity field use the
principal-axes frame (``MOON_PA``), but LOLA topography products are tabulated
in the mean-Earth/polar-axis frame: the LDEM PDS labels read
``COORDINATE_SYSTEM_NAME = "MEAN EARTH/POLAR AXIS OF DE421"``. The two frames
differ by a constant rotation of about 103.9 arcsec (up to 0.88 km on the
surface), so a latitude/longitude computed on principal axes must be rotated
before it indexes such a map.

The DE440 lunar frame kernel (``moon_de440_250416.tf``) defines
``MOON_ME_DE440_ME421`` relative to ``MOON_PA_DE440`` as a TK frame with
``TKFRAME_31009_ANGLES = (67.8526, 78.6944, 0.2785)`` arcsec about
``TKFRAME_31009_AXES = (3, 2, 1)``; ``MOON_PA`` and ``MOON_ME`` are identity
aliases of ``MOON_PA_DE440`` and ``MOON_ME_DE440_ME421``. The matrix built here
is that definition and is checked against SPICE ``pxform("MOON_PA", "MOON_ME",
et)`` in ``tests/test_lunar_map_frame.py`` when the kernels are present.

Only the DE440 principal-axes realisation is covered. For any other fixed frame
(``MOON_ME`` already is the map frame; ``IAU_MOON`` and other realisations are
not modelled) :func:`surface_map_rotation` returns ``None`` and callers sample
the map with the body-fixed coordinates unchanged.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: TKFRAME_31009 angles [arcsec] (axes 3, 2, 1): MOON_ME_DE440_ME421 relative to MOON_PA_DE440.
MOON_PA_DE440_TO_ME_ANGLES_ARCSEC: tuple[float, float, float] = (67.8526, 78.6944, 0.2785)

_ARCSEC_RAD = math.pi / (180.0 * 3600.0)
_PRINCIPAL_AXES_FRAMES = frozenset({"MOON_PA", "MOON_PA_DE440"})


def _spice_axis_rotation(axis: int, angle_rad: float) -> NDArray[np.float64]:
    """SPICE ``ROTATE`` matrix ``[angle]_axis`` (a frame rotation, not a vector one)."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    if axis == 1:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]], dtype=np.float64)
    if axis == 2:
        return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64)
    if axis == 3:
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"axis must be 1, 2 or 3; got {axis!r}")


def moon_pa_de440_to_me_matrix() -> NDArray[np.float64]:
    """Rotation matrix taking MOON_PA (DE440) coordinates to MOON_ME coordinates.

    SPICE builds a TK frame's ``ANGLES`` matrix as ``[a1]_3 [a2]_2 [a3]_1``; that
    product maps MOON_ME vectors to MOON_PA, so the PA->ME matrix returned here
    is its transpose.
    """
    a1, a2, a3 = (angle * _ARCSEC_RAD for angle in MOON_PA_DE440_TO_ME_ANGLES_ARCSEC)
    me_to_pa = (
        _spice_axis_rotation(3, a1) @ _spice_axis_rotation(2, a2) @ _spice_axis_rotation(1, a3)
    )
    return np.ascontiguousarray(me_to_pa.T)


def surface_map_rotation(fixed_frame: str | None) -> NDArray[np.float64] | None:
    """Body-fixed -> surface-map rotation for a propagation fixed frame, or ``None``.

    ``None`` means the body-fixed coordinates are used as map coordinates: the
    frame already is the map frame (``MOON_ME``) or its offset is not modelled.
    """
    if fixed_frame is None:
        return None
    if str(fixed_frame).strip().upper() in _PRINCIPAL_AXES_FRAMES:
        return moon_pa_de440_to_me_matrix()
    return None


def ephemeris_fixed_frame(ephem: Any) -> str | None:
    """Name of the fixed frame an ephemeris' ``q_i2f`` table rotates into, if recorded.

    Reads ``ephem.tables.fixed_frame`` (the frame SPICE sampled the attitude in),
    then ``ephem.fixed_frame``; returns ``None`` when neither is present.
    """
    if ephem is None:
        return None
    for owner in (getattr(ephem, "tables", None), ephem):
        frame = getattr(owner, "fixed_frame", None) if owner is not None else None
        if isinstance(frame, str) and frame.strip():
            return frame.strip()
    return None


__all__ = [
    "MOON_PA_DE440_TO_ME_ANGLES_ARCSEC",
    "ephemeris_fixed_frame",
    "moon_pa_de440_to_me_matrix",
    "surface_map_rotation",
]
