"""Binding-neutral design tokens shared by the Lunaris desktop interfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ColorTokens:
    """Widget colors plus semantic aliases used by QSS consumers.

    The compact historical names remain the compatibility surface. Properties
    below expose the explicit Phase 2 roles without storing duplicate values
    that could drift apart.
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

    accent: str = "#3B86FF"
    accent_hov: str = "#5C9CFF"
    accent_dim: str = "rgba(59,134,255,0.22)"
    accent_deep: str = "#2E6AD6"
    secondary: str = "#15D6A6"
    secondary_hov: str = "#4DE8C4"
    secondary_dim: str = "rgba(21,214,166,0.16)"

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

    border: str = "#3A4756"
    border_soft: str = "#1F2833"
    border_strong: str = "#4A5765"
    panel_shadow: str = "rgba(0,0,0,0.45)"
    grid_color: str = "rgba(100,116,139,0.42)"

    def as_legacy_dict(self) -> dict[str, str]:
        values = asdict(self)
        values.update(
            primary=self.accent,
            primary_hover=self.accent_hov,
            selected_bg=self.accent_dim,
            plot_bg=self.bg_log,
            text_disabled=self.fg_disabled,
            surface_app=self.surface_app,
            surface_shell=self.surface_shell,
            surface_card=self.surface_card,
            surface_elevated=self.surface_elevated,
            surface_inset=self.surface_inset,
            surface_input=self.surface_input,
            surface_terminal=self.surface_terminal,
            surface_selected=self.surface_selected,
            surface_hover=self.surface_hover,
            surface_overlay=self.surface_overlay,
            text_primary=self.text_primary,
            text_secondary=self.text_secondary,
            text_muted=self.text_muted,
            text_inverse=self.text_inverse,
            text_link=self.text_link,
            border_quiet=self.border_quiet,
            border_standard=self.border_standard,
            border_focus=self.border_focus,
            border_selected=self.border_selected,
            divider=self.divider,
        )
        return values

    @property
    def surface_app(self) -> str:
        return self.bg_space

    @property
    def surface_shell(self) -> str:
        return self.bg_shell

    @property
    def surface_card(self) -> str:
        return self.bg_card

    @property
    def surface_elevated(self) -> str:
        return self.bg_card_alt

    @property
    def surface_inset(self) -> str:
        return self.bg_inset

    @property
    def surface_input(self) -> str:
        return self.bg_entry

    @property
    def surface_terminal(self) -> str:
        return self.bg_log

    @property
    def surface_selected(self) -> str:
        return self.accent_dim

    @property
    def surface_hover(self) -> str:
        return self.bg_hover

    @property
    def surface_overlay(self) -> str:
        return self.bg_overlay

    @property
    def text_primary(self) -> str:
        return self.fg_main

    @property
    def text_secondary(self) -> str:
        return self.fg_soft

    @property
    def text_muted(self) -> str:
        return self.fg_muted

    @property
    def text_disabled(self) -> str:
        return self.fg_disabled

    @property
    def text_inverse(self) -> str:
        return self.fg_inverse

    @property
    def text_link(self) -> str:
        return self.fg_link

    @property
    def primary(self) -> str:
        return self.accent

    @property
    def border_quiet(self) -> str:
        return self.border_soft

    @property
    def border_standard(self) -> str:
        return self.border

    @property
    def border_focus(self) -> str:
        return self.accent

    @property
    def border_selected(self) -> str:
        return self.accent

    @property
    def divider(self) -> str:
        return self.border_soft


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
    family_ui: str = '"Segoe UI", "Inter", "Noto Sans", sans-serif'
    family_mono: str = '"Cascadia Mono", "Consolas", "Courier New", monospace'
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
    page_max_width: int = 1180
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
