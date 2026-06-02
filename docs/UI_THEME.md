# UI Theme — *Lunar Graphite*

This document describes the desktop UI theme for the Lunaris PySide6 application.

## Design goal

**Lunar Graphite** is a calm, professional, engineering-oriented dark theme. The
desktop UI should feel like serious scientific mission-control / orbital-analysis
software — readable for long sessions, dark but not neon, modern but not toy-like.

Concretely, the redesign favours:

- **flat surfaces** over gradients (gradients are reserved for the primary *Run*
  action and the progress-bar chunk);
- a **single restrained accent** — orbital blue (`#6AA9FF`) — instead of a
  dominant ion-cyan / violet glow;
- **lunar telemetry teal** (`#6EE7C8`) as the secondary highlight / success color;
- **amber** (`#E7B86A`) used *only* for warning states and the periapsis marker —
  never as a general brand accent;
- subtle borders and a clear elevation hierarchy
  (`bg_space` → `bg_shell` → `bg_card` → `bg_card_alt`).

## Two palettes, two jobs

| Palette | Where it lives | Used for |
| --- | --- | --- |
| `THEME` | `lunaris.ui.core.ui_commons` | **Qt widgets** — every QSS color, badge, button, input, card. |
| `ORBIT_THEME` | `lunaris.ui.core.ui_commons` | **OpenGL orbit preview** — Moon, orbit line, markers, axes. |
| `LOG_COLORS` | `lunaris.ui.core.ui_commons` | Rich-text colors for the execution log. |

`THEME` / `LOG_COLORS` are the single source of truth for Qt styling.
`ORBIT_THEME` is kept separate because the 3D preview needs deliberately
different values (a true space-black backdrop, regolith greys, marker hues) and
is consumed as float-RGBA tuples rather than CSS strings.

> **Avoid page-local hard-coded colors.** Route every color through `THEME` or
> `ORBIT_THEME`. For translucent variants in QSS, derive them from a token with
> `with_alpha(THEME["accent"], 0.2)` rather than writing a raw `rgba(...)` literal.

## Architecture

- **`lunaris.ui.core.ui_commons`** — defines `THEME`, `LOG_COLORS`,
  `ORBIT_THEME`, the shared widget primitives, and the color helpers
  (`hex_to_rgba_float`, `rgba_css_to_tuple`, `with_alpha`).
- **`lunaris.ui.theme.stylesheet`** — `build_app_stylesheet(THEME, LOG_COLORS)`
  generates the entire application QSS. This keeps the large stylesheet out of
  `lunaris.ui.app`, so the main window stays an orchestration layer rather than a
  design-token dump.
- **`lunaris.ui.app.MainWindow._apply_theme`** — sets the base `QPalette` and
  applies `build_app_stylesheet(THEME, LOG_COLORS)`.

```python
from lunaris.ui.core.ui_commons import THEME, LOG_COLORS
from lunaris.ui.theme import build_app_stylesheet

self.setStyleSheet(build_app_stylesheet(THEME, LOG_COLORS))
```

## Color helpers

```python
hex_to_rgba_float("#6AA9FF")            # -> (0.416, 0.663, 1.0, 1.0)   (OpenGL)
hex_to_rgba_float("#6AA9FF", 0.25)      # -> (0.416, 0.663, 1.0, 0.25)
rgba_css_to_tuple("rgba(106,169,255,0.12)")  # -> (0.416, 0.663, 1.0, 0.12)
with_alpha("#F87171", 0.3)              # -> "rgba(248, 113, 113, 0.3)"  (QSS)
```

## Token reference

### `THEME` (Qt widgets)

| Token | Value | Role |
| --- | --- | --- |
| `bg_space` | `#05070A` | app background (deepest) |
| `bg_shell` | `#0A0F16` | header / nav / log-header shell |
| `bg_card` | `#111821` | primary cards |
| `bg_card_alt` | `#17202B` | elevated / hover surfaces |
| `bg_entry` | `#0D141D` | inputs, selectors |
| `bg_log` | `#030507` | terminal / log (`plot_bg` alias) |
| `fg_main` / `fg_soft` / `fg_muted` | `#E7ECF2` / `#B8C3D0` / `#7D8997` | text tiers |
| `accent` / `accent_hov` / `accent_dim` | `#6AA9FF` / `#9AC4FF` / `rgba(106,169,255,0.12)` | primary orbital blue |
| `secondary` / `secondary_hov` / `secondary_dim` | `#6EE7C8` / `#9FF3DD` / `rgba(110,231,200,0.10)` | telemetry teal |
| `success` / `warning` / `error` / `info` | `#6EE7C8` / `#E7B86A` / `#F87171` / `#8AB4F8` | semantic |
| `border` / `border_soft` | `#263241` / `#1B2530` | borders / separators |
| `accent_deep` | `#315F99` | calm hover edge |

Aliases kept for backward compatibility: `primary`, `primary_hover`,
`selected_bg`, `panel_shadow`, `plot_bg`, `grid_color`, `text_disabled`.

### `ORBIT_THEME` (OpenGL orbit preview)

| Token | Value | Role |
| --- | --- | --- |
| `space_bg` | `#020408` | GL background |
| `moon_dark` / `moon_mid` / `moon_light` | `#5E6268` / `#8D9299` / `#B7BCC4` | regolith greys |
| `orbit_line` / `orbit_glow` | `#7DB7FF` / `#3B82F6` | trajectory + faint glow |
| `spacecraft` | `#F8FAFC` | current true-anomaly marker |
| `periapsis` / `apoapsis` | `#E7B86A` / `#6EE7C8` | amber / teal markers |
| `orbit_plane` | `rgba(106,169,255,0.08)` | optional orbit-plane fill |
| `axis_x` / `axis_y` / `axis_z` | muted red / teal / blue | ECI orientation axes |

## App identity

The visible desktop app is **Lunaris Mission Studio** (`APP_NAME` in
`ui_commons.py`); the window title, header, and saved-profile metadata all track
it. The Python package, console entry points (`lunaris-ui`, `lunaris-studio`),
and the ST-LRPS surrogate feature names are unchanged. Session data now lives
under a `LunarisMissionStudio` app-data folder, but the previous `STLRPSStudio`
folder is read once so saved mission profiles survive the rename.

## Execution Console

The bottom log panel is the **Execution Console**
(`lunaris.ui.widgets.log_panel.ExecutionLogPanel`). It replaces the previous
HTML-append log with a robust, buffered design:

- an internal `LogEntry` model plus a small pending queue flushed to the widget
  on an ~80 ms timer, so high-volume subprocess output is batched;
- a `QPlainTextEdit` console with controlled, colored `QTextCursor` insertion
  (stable spacing, clean copy, no HTML layout bugs);
- compact, scannable severity prefixes — `[HH:MM:SS] [INFO] message` — with a
  muted timestamp (`LOG_COLORS["timestamp"]`);
- retained output bounded to `MAX_LOG_LINES` (10000);
- toolbar controls: **Pause / Copy / Clear / Save / Collapse** plus
  **Auto-scroll / Wrap / Show-timestamps** toggles.

`MainWindow` keeps thin `_log_message` / `_log_separator` / `_clear_log`
delegates so existing call sites are unchanged.

## Capturing a screenshot

A lightweight, offscreen-capable utility renders the main window and saves a PNG:

```bash
python tools/ui/capture_main_window.py
# -> outputs/ui/main_window_lunar_graphite.png
```

It forces Qt's `offscreen` platform by default, so it runs without a display.
