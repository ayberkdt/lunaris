# UI Theme - Lunar Graphite

Lunar Graphite is the shared desktop theme for Lunaris Mission Studio and the
ST-LRPS Studio. It favors calm graphite surfaces, strong text hierarchy, one
orbital-blue accent, and semantic colors used only for state.

## Ownership

- `lunaris.ui.theme.tokens` contains the typed `DESIGN_TOKENS` source of truth:
  `ColorTokens`, `TypographyTokens`, `SpacingTokens`, `RadiusTokens`,
  `ControlMetrics`, `LayoutTokens`, and `VisualizationTokens`.
- `lunaris.ui.core.ui_commons.THEME` is the backward-compatible dictionary
  facade used by existing pages and plot adapters.
- `lunaris.ui.theme.stylesheet.build_app_stylesheet` owns global QSS for both
  desktop studios.
- `ORBIT_THEME` is a compatibility facade over `VisualizationTokens`; OpenGL
  widgets still receive their specialized lunar/material roles separately from
  application chrome.

Page-local QSS should be limited to runtime plot series, OpenGL materials, or a
third-party widget that cannot be styled through a property or palette.

## Palette

| Token | Value | Role |
| --- | --- | --- |
| `bg_space` | `#070B11` | application canvas |
| `bg_shell` | `#0C121B` | top bar, navigation, console strip |
| `bg_card` | `#121A25` | sections and primary surfaces |
| `bg_card_alt` | `#192433` | elevated surfaces |
| `bg_inset` | `#0D151F` | read-only and disabled surfaces |
| `bg_entry` | `#0A121C` | input controls |
| `bg_log` | `#05080D` | console canvas |
| `fg_main` | `#F3F6FB` | primary text |
| `fg_soft` | `#C5D0DE` | secondary text |
| `fg_muted` | `#94A3B8` | metadata and hints |
| `fg_disabled` | `#728298` | inactive control text |
| `accent` | `#4F8CFF` | selection, focus, primary action |
| `success` | `#22D3B6` | success and completed state |
| `warning` | `#E7B86A` | warning state |
| `error` | `#F87171` | failure and destructive action |
| `critical` | `#FB7185` | critical state |
| `info` | `#60A5FA` | informational state |
| `border` | `#465A73` | control and standard surface border |
| `border_soft` | `#253246` | separators and quiet surfaces |
| `border_strong` | `#60758F` | focus-adjacent and elevated borders |

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
