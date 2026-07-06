"""Binding-neutral design tokens shared by the Lunaris desktop interfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ColorTokens:
    """Widget colors plus semantic aliases used by QSS consumers.

    Names follow the single ``bg_*`` / ``fg_*`` / role (accent, border, success,
    ...) scheme. There is one name per color; consumers read these names
    directly (via :meth:`as_dict` -> ``THEME``) with no parallel alias surface.
    """

    bg_space: str = "#090C12"
    bg_shell: str = "#0F141C"
    bg_card: str = "#171E29"
    bg_card_alt: str = "#212B38"
    bg_inset: str = "#0C1016"
    bg_entry: str = "#0E131B"
    bg_log: str = "#06090E"
    bg_hover: str = "#28333F"
    bg_overlay: str = "rgba(7,10,15,0.92)"

    fg_main: str = "#EEF2F8"
    fg_soft: str = "#BCC7D6"
    fg_muted: str = "#8C99AA"
    fg_disabled: str = "#79848F"
    fg_inverse: str = "#04070C"
    fg_link: str = "#6FA8FF"

    # Restrained orbital-blue accent. Lighter/calmer than the prior #3B86FF: on
    # a cool-graphite canvas a fully saturated blue vibrates at edges over long
    # sessions, and primary buttons carry dark fg_inverse text, so a lighter
    # accent both calms the chrome and lifts contrast (6.96:1 as text on bg_card,
    # 8.39:1 for dark text on the accent fill, vs 4.82/5.80). accent_hov stays
    # lighter for a perceptible hover; accent_deep is the darker pressed shade.
    accent: str = "#6AA9FF"
    accent_hov: str = "#86BBFF"
    accent_dim: str = "rgba(106,169,255,0.22)"
    accent_deep: str = "#2E6AD6"
    secondary: str = "#15D6A6"
    secondary_hov: str = "#4DE8C4"
    secondary_dim: str = "rgba(21,214,166,0.16)"

    # A third widget-level accent (violet) used to give navigation sections
    # distinct identities without leaning on the semantic status colors. It is
    # deliberately high-value on the graphite canvas (6.1:1 as text on bg_shell)
    # so a section label reads clearly; ``tertiary_dim`` is the tinted active fill.
    tertiary: str = "#AE9BFF"
    tertiary_hov: str = "#C6B8FF"
    tertiary_dim: str = "rgba(174,155,255,0.18)"

    # ``success`` is a true green, deliberately separated in hue from the
    # ``secondary`` teal (#15D6A6) above. They previously sat ~1-8 per channel
    # apart (#14D49E vs #15D6A6) so a "completed" status and a comparison data
    # series were perceptually the same color; value/hue must carry the meaning.
    success: str = "#3DD17E"
    warning: str = "#F5B43C"
    error: str = "#FF5D6C"
    # ``critical`` is a more intense red than ``error`` so the severity ladder
    # (warning < error < critical) is perceptible by value, not just by label.
    critical: str = "#FF3355"
    info: str = "#4A9DFF"
    inactive: str = "#6C7787"

    # ``border`` carries load-bearing component boundaries (inputs, lists, key
    # frames). Surface fills differ from their containers by only ~1.1:1, so the
    # border alone identifies the control; it must meet WCAG 1.4.11 non-text
    # 3:1. #3A4756 was 1.77:1 on bg_card; #6A7686 is >=3:1 on every surface
    # (3.63 card / 4.00 shell / 4.03 entry / 3.10 elevated). ``border_soft``
    # stays intentionally subtle for decorative rules.
    border: str = "#6A7686"
    border_soft: str = "#1F2833"
    border_strong: str = "#4A5765"
    panel_shadow: str = "rgba(0,0,0,0.45)"
    grid_color: str = "rgba(100,116,139,0.42)"

    def as_dict(self) -> dict[str, str]:
        """Return all color tokens as a flat ``name -> value`` mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VisualizationTokens:
    """OpenGL and scientific-plot colors kept separate from widget chrome."""

    space_bg: str = "#020408"
    moon_dark: str = "#5E6268"
    moon_mid: str = "#8D9299"
    moon_light: str = "#B7BCC4"
    orbit_line: str = "#7DB7FF"
    orbit_glow: str = "#3B82F6"
    spacecraft: str = "#F8FAFC"
    periapsis: str = "#E7B86A"
    apoapsis: str = "#22D3B6"
    orbit_plane: str = "rgba(79,140,255,0.10)"
    terrain: str = "#A0A7B0"
    gravity: str = "#A78BFA"
    srp: str = "#F59E0B"
    sun: str = "#FACC15"
    earth: str = "#3B86FF"
    selected_plot: str = "#3B86FF"
    comparison_plot: str = "#15D6A6"
    grid: str = "rgba(100,116,139,0.42)"
    axis_text: str = "#94A3B8"
    axis_x: str = "rgba(248,113,113,0.78)"
    axis_y: str = "rgba(34,211,182,0.78)"
    axis_z: str = "rgba(125,183,255,0.78)"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TypographyTokens:
    # Modular type scale (~1.2 ratio) with perceptible steps so hierarchy holds
    # in grayscale. ``section`` was 11 pt — only 1 pt over ``body`` (10 pt), an
    # imperceptible step that forced section headings to lean entirely on weight;
    # 12 pt (= body x 1.2) makes the heading tier read by size as well.
    # Qt style sheets accept a single family name here, not a CSS fallback
    # stack. The app-level font loader still handles platform fallbacks before
    # the stylesheet is applied.
    family_ui: str = '"Segoe UI"'
    family_mono: str = '"Consolas"'
    size_caption_pt: float = 9.0
    size_body_pt: float = 10.0
    size_section_pt: float = 12.0
    size_app_title_pt: float = 15.0
    size_page_title_pt: float = 20.0
    weight_regular: int = 400
    weight_medium: int = 500
    weight_semibold: int = 600
    weight_bold: int = 700


@dataclass(frozen=True, slots=True)
class SpacingTokens:
    xxs: int = 4
    xs: int = 6
    sm: int = 8
    md: int = 12
    lg: int = 16
    xl: int = 20
    xxl: int = 24
    xxxl: int = 32


@dataclass(frozen=True, slots=True)
class RadiusTokens:
    compact: int = 6
    control: int = 8
    section: int = 10
    shell: int = 12
    pill: int = 999


@dataclass(frozen=True, slots=True)
class ControlMetrics:
    compact_height: int = 30
    minimum_height: int = 34
    primary_height: int = 38
    icon_button_size: int = 30
    status_badge_height: int = 24
    form_label_width: int = 156


@dataclass(frozen=True, slots=True)
class LayoutTokens:
    shell_margin: int = 16
    shell_gap: int = 12
    page_gap: int = 16
    section_padding: int = 16
    page_max_width: int = 1440
    nav_width: int = 216
    nav_compact_width: int = 188
    nested_nav_width: int = 176
    console_collapsed_height: int = 42
    console_expanded_min_height: int = 210
    header_height: int = 64
    toolbar_height: int = 34


@dataclass(frozen=True, slots=True)
class DesignTokens:
    colors: ColorTokens = ColorTokens()
    visualization: VisualizationTokens = VisualizationTokens()
    typography: TypographyTokens = TypographyTokens()
    spacing: SpacingTokens = SpacingTokens()
    radii: RadiusTokens = RadiusTokens()
    controls: ControlMetrics = ControlMetrics()
    layout: LayoutTokens = LayoutTokens()


DESIGN_TOKENS = DesignTokens()


__all__ = [
    "ColorTokens",
    "VisualizationTokens",
    "TypographyTokens",
    "SpacingTokens",
    "RadiusTokens",
    "ControlMetrics",
    "LayoutTokens",
    "DesignTokens",
    "DESIGN_TOKENS",
]
