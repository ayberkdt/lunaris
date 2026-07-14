"""Linestyle-aware CVD audit of the multi-series plot cycle (W4.3).

Verifies the invariant behind ``VisualizationTokens.series_cycle`` /
``series_dash_cycle``: within one full combination period, two series that
receive the SAME dash pattern must keep their colors distinguishable for
normal vision *and* under protanopia / deuteranopia / tritanopia simulation
(Machado et al. 2009, severity 1.0). Pairs whose dash differs pass regardless
of color distance - the line style is the non-color identity channel, which is
exactly why the cycles are consumed together.

Usage:
    python tools/ui/color_audit.py            # report + exit 1 on violation

The check is also importable (``audit_series_cycle``) so a pytest can gate it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:  # direct script execution
    sys.path.insert(0, str(_REPO_SRC))

from lunaris.ui_foundation.tokens import DESIGN_TOKENS  # noqa: E402

#: Redmean distance below which two colors are treated as confusable. The
#: prior UI color audits used the same bar (see docs/UI_AUDIT.md).
CONFUSION_THRESHOLD = 60.0

# Machado, Oliveira, Fernandes (2009) severity-1.0 simulation matrices,
# applied in linear RGB.
_CVD_MATRICES: dict[str, tuple[tuple[float, float, float], ...]] = {
    "protanopia": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deuteranopia": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritanopia": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    value = color.strip().lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _to_linear(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _to_srgb(channel: float) -> float:
    channel = min(1.0, max(0.0, channel))
    return 12.92 * channel if channel <= 0.0031308 else 1.055 * channel ** (1 / 2.4) - 0.055


def simulate(color: str, deficiency: str) -> tuple[float, float, float]:
    """Simulate *color* (hex) under a CVD *deficiency*; returns sRGB in 0..1."""
    rgb_lin = tuple(_to_linear(c) for c in _hex_to_rgb(color))
    matrix = _CVD_MATRICES[deficiency]
    out_lin = tuple(sum(row[k] * rgb_lin[k] for k in range(3)) for row in matrix)
    return tuple(_to_srgb(c) for c in out_lin)  # type: ignore[return-value]


def redmean(rgb_a: tuple[float, float, float], rgb_b: tuple[float, float, float]) -> float:
    """Redmean color distance on 0-255 components."""
    r1, g1, b1 = (c * 255.0 for c in rgb_a)
    r2, g2, b2 = (c * 255.0 for c in rgb_b)
    rmean = (r1 + r2) / 2.0
    dr, dg, db = r1 - r2, g1 - g2, b1 - b2
    return (
        ((2.0 + rmean / 256.0) * dr * dr)
        + 4.0 * dg * dg
        + ((2.0 + (255.0 - rmean) / 256.0) * db * db)
    ) ** 0.5


def audit_series_cycle() -> list[str]:
    """Return violation messages (empty list = the cycle invariant holds)."""
    viz = DESIGN_TOKENS.visualization
    colors = viz.series_cycle
    dashes = viz.series_dash_cycle
    # The (color, dash) pattern repeats with period lcm(|colors|, |dashes|);
    # the adapter (lunaris.ui.core.plot_style) wraps at the same period.
    period = math.lcm(len(colors), len(dashes))

    violations: list[str] = []

    # Role dashes must be pairwise distinct (truth vs surrogate vs comparison).
    roles = {
        "truth": viz.truth_dash,
        "surrogate": viz.surrogate_dash,
        "comparison": viz.comparison_dash,
    }
    if len(set(roles.values())) != len(roles):
        violations.append(f"role dash styles are not distinct: {roles}")
    unknown = [f"{k}={v}" for k, v in roles.items() if v not in dashes]
    if unknown:
        violations.append(f"role dash not in series_dash_cycle: {unknown}")

    # Same-dash series pairs within one period must survive every vision type.
    for i in range(period):
        for j in range(i + 1, period):
            if i % len(dashes) != j % len(dashes):
                continue  # different line style carries the identity: pass
            color_i = colors[i % len(colors)]
            color_j = colors[j % len(colors)]
            if color_i == color_j:
                violations.append(
                    f"series {i} and {j} share color {color_i} AND dash "
                    f"{dashes[i % len(dashes)]} within one cycle period"
                )
                continue
            for vision in ("normal", *sorted(_CVD_MATRICES)):
                if vision == "normal":
                    pair = (_hex_to_rgb(color_i), _hex_to_rgb(color_j))
                else:
                    pair = (simulate(color_i, vision), simulate(color_j, vision))
                distance = redmean(*pair)
                if distance < CONFUSION_THRESHOLD:
                    violations.append(
                        f"series {i} ({color_i}) vs {j} ({color_j}): same dash "
                        f"'{dashes[i % len(dashes)]}' but redmean {distance:.1f} "
                        f"< {CONFUSION_THRESHOLD:.0f} under {vision}"
                    )
    return violations


def main() -> int:
    violations = audit_series_cycle()
    viz = DESIGN_TOKENS.visualization
    print(
        f"[color-audit] series_cycle={len(viz.series_cycle)} colors, "
        f"series_dash_cycle={len(viz.series_dash_cycle)} dashes, "
        f"period={math.lcm(len(viz.series_cycle), len(viz.series_dash_cycle))}"
    )
    if violations:
        for message in violations:
            print(f"[VIOLATION] {message}")
        return 1
    print("[color-audit] OK - every same-dash pair is distinguishable in all vision types.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
