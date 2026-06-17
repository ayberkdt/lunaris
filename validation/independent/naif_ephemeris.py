"""Independent NAIF/SPICE ephemeris cross-check.

SPICE *is* the trusted ephemeris reference, so "independent" here means
independent of the Lunaris ephemeris *code path*: this module issues its own
minimal :func:`spiceypy.spkpos` queries and compares them to what
:class:`lunaris.physics.ephemeris.EphemerisManager` returns. It catches a wrong
target/observer/frame, a km↔m unit slip, or an unexpectedly large linear
table-interpolation error — none of which the in-tree code can catch by
comparing against itself.

Two error families are reported:

* **node error** — at a table grid node the Lunaris interpolation is the identity,
  so this isolates the table-build path (units, target, frame, observer). It
  should be at the metre level or below.
* **interpolation error** — at an off-node epoch the direct SPICE query is the
  truth; the gap quantifies the error introduced by Lunaris' Catmull-Rom table
  interpolation for the configured ``dt``. Only *interior* intervals are probed:
  Catmull-Rom needs two neighbours on each side, so the first and last intervals
  use clamped endpoint tangents and degrade by design — the production solver
  operates in the interior, which is what this metric reflects.

All positions are in metres in the table's inertial frame.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "furnished_kernels",
    "direct_position_m",
    "EphemerisCrossCheckResult",
    "crosscheck_ephemeris_manager",
]


@contextlib.contextmanager
def furnished_kernels(kernel_paths: Sequence[str]) -> Iterator[None]:
    """Temporarily furnish SPICE kernels, unloading them again on exit."""
    import spiceypy as spice

    loaded: list[str] = []
    try:
        for kernel in kernel_paths:
            spice.furnsh(str(kernel))
            loaded.append(str(kernel))
        yield
    finally:
        for kernel in reversed(loaded):
            with contextlib.suppress(Exception):
                spice.unload(kernel)


def direct_position_m(
    target: str,
    et: float,
    frame: str,
    observer: str,
    *,
    abcorr: str = "NONE",
) -> np.ndarray:
    """Direct ``spkpos`` query returning the position in METRES.

    Kernels must already be furnished (see :func:`furnished_kernels`).
    """
    import spiceypy as spice

    pos_km, _light_time = spice.spkpos(target, float(et), frame, abcorr, observer)
    return np.asarray(pos_km, dtype=np.float64) * 1000.0


@dataclass(frozen=True)
class EphemerisCrossCheckResult:
    """Per-target summary of the manager-vs-direct-SPICE comparison."""

    target: str
    n_nodes_checked: int
    n_offnode_checked: int
    max_node_error_m: float
    max_interp_error_m: float

    def node_path_ok(self, *, tol_m: float = 1.0) -> bool:
        """True when the table-build path matches direct SPICE to ``tol_m``."""
        return bool(self.max_node_error_m <= tol_m)


# Position getter names on EphemerisManager keyed by SPICE target.
_TARGET_GETTERS = {
    "EARTH": "get_earth_position",
    "SUN": "get_sun_position",
}


def crosscheck_ephemeris_manager(
    manager: Any,
    kernel_paths: Sequence[str],
    *,
    targets: Sequence[str] = ("EARTH", "SUN"),
    abcorr: str = "NONE",
    max_nodes: int = 16,
) -> dict[str, EphemerisCrossCheckResult]:
    """Cross-check a built :class:`EphemerisManager` against direct SPICE.

    ``manager`` is the object under test (its ``get_*_position`` interpolate the
    stored tables). ``kernel_paths`` are furnished here for the independent
    reference queries — the manager may already have cleared its own kernels.
    """
    provider = manager.get_data_provider()
    et0 = float(provider["et0"])
    t_tab = np.asarray(provider["t_tab_s"], dtype=np.float64)
    frame = str(provider["inertial_frame"])
    observer = str(provider["observer"])
    n = int(t_tab.shape[0])

    # Evenly subsample grid nodes; off-node probes sit at node midpoints.
    node_idx = np.unique(np.linspace(0, n - 1, num=min(max_nodes, n), dtype=int))

    results: dict[str, EphemerisCrossCheckResult] = {}
    with furnished_kernels(kernel_paths):
        for target in targets:
            getter_name = _TARGET_GETTERS.get(target.upper())
            if getter_name is None:
                raise ValueError(f"unsupported target {target!r}; expected one of {sorted(_TARGET_GETTERS)}")
            getter = getattr(manager, getter_name)

            node_err = 0.0
            for i in node_idx:
                t_s = float(t_tab[i])
                lunaris = np.asarray(getter(t_s), dtype=np.float64)
                ref = direct_position_m(target, et0 + t_s, frame, observer, abcorr=abcorr)
                node_err = max(node_err, float(np.linalg.norm(lunaris - ref)))

            interp_err = 0.0
            offnode_count = 0
            for i in node_idx:
                # Skip the first/last intervals: Catmull-Rom degrades there by
                # design (clamped boundary tangents), which would otherwise
                # dominate the interior interpolation accuracy we want to report.
                if i <= 0 or i + 1 >= n - 1:
                    continue
                t_s = 0.5 * (float(t_tab[i]) + float(t_tab[i + 1]))
                lunaris = np.asarray(getter(t_s), dtype=np.float64)
                ref = direct_position_m(target, et0 + t_s, frame, observer, abcorr=abcorr)
                interp_err = max(interp_err, float(np.linalg.norm(lunaris - ref)))
                offnode_count += 1

            results[target] = EphemerisCrossCheckResult(
                target=target,
                n_nodes_checked=len(node_idx),
                n_offnode_checked=offnode_count,
                max_node_error_m=node_err,
                max_interp_error_m=interp_err,
            )
    return results
