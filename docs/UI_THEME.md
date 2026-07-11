# UI Theme - Lunar Graphite

Lunar Graphite is the shared desktop theme for Lunaris Mission Studio and the
ST-LRPS Studio. It favors calm graphite surfaces, strong text hierarchy, one
orbital-blue accent, and semantic colors used only for state.

## Ownership

`lunaris.ui_foundation` is the single source of truth (SSOT) for the entire UI
design system. It is **Qt-binding-neutral** (imports no `PySide6`) so the mission
UI and the ST-LRPS Studio share exactly one palette, token set, and stylesheet
generator. Everything else is a thin compatibility re-export — do not define or
edit color literals, tokens, or QSS anywhere else.

| Concern | SSOT (edit here) | Compatibility re-exports (do not add tokens) |
| --- | --- | --- |
| Typed `DESIGN_TOKENS` (`ColorTokens`, `TypographyTokens`, `SpacingTokens`, `RadiusTokens`, `ControlMetrics`, `LayoutTokens`, `VisualizationTokens`) | `lunaris.ui_foundation.tokens` | `lunaris.ui.theme.tokens`, `lunaris.ui.theme.__init__` |
| `THEME` widget palette, `LOG_COLORS`, `ORBIT_THEME`, color helpers (`with_alpha`, `hex_to_rgba_float`, `rgba_css_to_tuple`) | `lunaris.ui_foundation.palette` | `lunaris.ui.core.ui_commons` (dict facade for existing pages and plot adapters) |
| Global QSS generation (`build_app_stylesheet`) | `lunaris.ui_foundation.stylesheet` | `lunaris.ui.theme.stylesheet`, `lunaris.ui.theme.__init__` |

`THEME` / `LOG_COLORS` remain read-only-compatible dictionary facades over the
typed tokens. `ORBIT_THEME` is a facade over `VisualizationTokens`; OpenGL
widgets still receive their specialized lunar/material roles separately from
application chrome. The `lunaris.ui_foundation` boundary is enforced by an
import-linter contract (see [pyproject.toml] `[tool.importlinter]`): the
foundation must never import either desktop application.

Page-local QSS should be limited to runtime plot series, OpenGL materials, or a
third-party widget that cannot be styled through a property or palette.

## Palette

The single source of truth is `lunaris.ui_foundation.tokens.ColorTokens`; this
table mirrors it and is guarded by `tests/test_ui_theme.py`
(`test_ui_theme_doc_palette_matches_tokens`). Edit the tokens first, then the
table.

| Token | Value | Role |
| --- | --- | --- |
| `bg_space` | `#090C12` | application canvas |
| `bg_shell` | `#0F141C` | top bar, navigation, console strip |
| `bg_card` | `#171E29` | sections and primary surfaces |
| `bg_card_alt` | `#212B38` | elevated surfaces |
| `bg_inset` | `#0C1016` | read-only and disabled surfaces |
| `bg_entry` | `#0E131B` | input controls |
| `bg_log` | `#06090E` | console canvas |
| `bg_hover` | `#28333F` | hover fill on flat surfaces |
| `fg_main` | `#EEF2F8` | primary text |
| `fg_soft` | `#BCC7D6` | secondary text |
| `fg_muted` | `#8C99AA` | metadata and hints |
| `fg_disabled` | `#79848F` | inactive control text |
| `fg_inverse` | `#04070C` | dark text on accent/semantic fills |
| `fg_link` | `#6FA8FF` | hyperlinks |
| `accent` | `#6AA9FF` | selection, focus, primary action |
| `accent_hov` | `#86BBFF` | hover state of accented controls |
| `accent_deep` | `#2E6AD6` | pressed/active accent shade |
| `secondary` | `#15D6A6` | comparison series, secondary accent |
| `secondary_hov` | `#4DE8C4` | hover state of secondary controls |
| `tertiary` | `#AE9BFF` | navigation section identity (third accent) |
| `tertiary_hov` | `#C6B8FF` | hover state of tertiary elements |
| `success` | `#3DD17E` | success and completed state |
| `warning` | `#F5B43C` | warning state |
| `error` | `#FF5D6C` | failure and destructive action |
| `critical` | `#FF3355` | critical state (more intense than `error`) |
| `info` | `#4A9DFF` | informational state |
| `inactive` | `#6C7787` | idle/neutral state marks |
| `border` | `#6A7686` | control and standard surface border (WCAG 1.4.11, >=3:1) |
| `border_soft` | `#1F2833` | separators and quiet surfaces |
| `border_strong` | `#4A5765` | focus-adjacent and elevated borders |

Decorative gradients are not used. Translucent variants are derived from tokens
with `with_alpha(...)`.

## Shared Components

The page hierarchy is `PageShell > PageHeader > Section/Subsection`. Forms,
metrics, notices, actions, toolbars, segmented controls, empty states, search,
overflow actions, and the execution console use components from
`lunaris.ui.components`.

The execution console is collapsed by default. Its compact strip keeps the last
message and warning/error counts visible. Expanded mode provides search,
severity filtering, follow output, copy, clear, overflow actions, and collapse.

## Screenshot

The offscreen capture utility writes to the ignored `outputs/ui` directory:

```bash
python tools/ui/capture_main_window.py
python tools/ui/capture_main_window.py --page Forces --state validating
python tools/ui/capture_main_window.py --target st-lrps --page "Training Monitor" --state running
python tools/ui/capture_main_window.py --dialog gravity
```

The utility requires the project UI dependencies, including PySide6.
