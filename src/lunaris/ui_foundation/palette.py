"""Binding-neutral Lunar Graphite palettes and color helpers."""

from __future__ import annotations

from lunaris.ui_foundation.tokens import DESIGN_TOKENS

THEME = DESIGN_TOKENS.colors.as_dict()

LOG_COLORS = {
    "error": DESIGN_TOKENS.colors.error,
    "warning": DESIGN_TOKENS.colors.warning,
    "success": DESIGN_TOKENS.colors.success,
    "system": DESIGN_TOKENS.colors.fg_soft,
    "info": DESIGN_TOKENS.colors.accent_hov,
    "debug": DESIGN_TOKENS.colors.fg_muted,
    "timestamp": DESIGN_TOKENS.colors.inactive,
    "default": DESIGN_TOKENS.colors.fg_main,
}

ORBIT_THEME = {
    "space_bg": DESIGN_TOKENS.visualization.space_bg,
    "moon_dark": DESIGN_TOKENS.visualization.moon_dark,
    "moon_mid": DESIGN_TOKENS.visualization.moon_mid,
    "moon_light": DESIGN_TOKENS.visualization.moon_light,
    "orbit_line": DESIGN_TOKENS.visualization.orbit_line,
    "orbit_glow": DESIGN_TOKENS.visualization.orbit_glow,
    "spacecraft": DESIGN_TOKENS.visualization.spacecraft,
    "periapsis": DESIGN_TOKENS.visualization.periapsis,
    "apoapsis": DESIGN_TOKENS.visualization.apoapsis,
    "orbit_node": DESIGN_TOKENS.visualization.orbit_node,
    "orbit_plane": DESIGN_TOKENS.visualization.orbit_plane,
    "axis_x": DESIGN_TOKENS.visualization.axis_x,
    "axis_y": DESIGN_TOKENS.visualization.axis_y,
    "axis_z": DESIGN_TOKENS.visualization.axis_z,
}


def hex_to_rgba_float(color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Convert a hex color to an RGBA tuple with components in ``0..1``."""

    value = color.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        raise ValueError(f"Invalid hex color: {color!r}")
    red = int(value[0:2], 16) / 255.0
    green = int(value[2:4], 16) / 255.0
    blue = int(value[4:6], 16) / 255.0
    opacity = max(0.0, min(1.0, float(alpha)))
    return (red, green, blue, opacity)


def rgba_css_to_tuple(color: str) -> tuple[float, float, float, float]:
    """Convert a CSS hex or rgb/rgba token to an RGBA float tuple."""

    value = color.strip()
    if value.startswith("#"):
        return hex_to_rgba_float(value)
    if value.lower().startswith("rgb"):
        inner = value[value.find("(") + 1 : value.rfind(")")]
        parts = [part.strip() for part in inner.split(",") if part.strip()]
        if len(parts) not in (3, 4):
            raise ValueError(f"Invalid rgb/rgba color: {color!r}")
        red = float(parts[0]) / 255.0
        green = float(parts[1]) / 255.0
        blue = float(parts[2]) / 255.0
        alpha = float(parts[3]) if len(parts) == 4 else 1.0
        return (red, green, blue, max(0.0, min(1.0, alpha)))
    raise ValueError(f"Unrecognized color token: {color!r}")


def with_alpha(color: str, alpha: float) -> str:
    """Return a CSS rgba token derived from a hex or rgb/rgba color."""

    red, green, blue, _ = rgba_css_to_tuple(color)
    opacity = max(0.0, min(1.0, float(alpha)))
    return (
        f"rgba({round(red * 255)}, {round(green * 255)}, "
        f"{round(blue * 255)}, {opacity:g})"
    )


__all__ = [
    "THEME",
    "LOG_COLORS",
    "ORBIT_THEME",
    "hex_to_rgba_float",
    "rgba_css_to_tuple",
    "with_alpha",
]
