# Lunaris Desktop UI Audit

Date: 2026-06-13 (baseline counts updated 2026-07-11)

## Scope

This audit covers the PySide6 mission studio in `lunaris.ui` and the PyQt6
ST-LRPS studio in `lunaris.surrogate.st_lrps.ui`. It focuses on hierarchy,
theme ownership, reusable components, accessibility, navigation, and the
execution workflow. Scientific behavior and command builders are out of scope.

## Baseline

- 310 Python `setStyleSheet(...)` call sites under `src/lunaris`
  (as of 2026-07-11: 50 in the mission studio — mostly `ui_commons`
  shared infrastructure — and 1 in the ST-LRPS UI, the global-stylesheet
  application in `qt_common.py`).
- 408 Python color literals under `src/lunaris`.
- 121 inline stylesheet call sites under the ST-LRPS UI alone
  (burned down to 1 across five passes ending 2026-07-11; page styling now
  resolves through `ui_foundation.stylesheet.build_app_stylesheet` roles).
- The main mission studio already has a centralized Lunar Graphite stylesheet,
  but page-local styles still override it frequently.
- ST-LRPS carries a second cyan/violet visual language and a duplicated global
  stylesheet.

These counts are diagnostic, not a zero-inline mandate. Dynamic plot colors,
runtime severity colors, and isolated third-party widget fixes may remain local
when a property or palette cannot express the same behavior safely.

## Findings

### P0 - Theme ownership is split

`lunaris.ui.core.ui_commons`, `lunaris.ui.theme.stylesheet`,
`st_lrps.ui.studio_parts.qt_common`, and many pages all define parts of the
visual system. The ST-LRPS stylesheet is a parallel theme implementation rather
than a consumer of the application theme.

Impact: theme changes are expensive, visual regressions are easy, and the two
studios feel like separate products.

Action: introduce typed tokens, preserve the `THEME` mapping as a compatibility
facade, and make both studios consume one QSS builder.

### P0 - Pages do not share a stable hierarchy

Page headings, descriptions, action placement, card margins, and form layouts
are constructed independently. Several pages use large nested group boxes;
others use flat widgets or custom cards. Primary actions can appear in the
global header, inside a page, or at the bottom of a form.

Impact: users must relearn scanning and action placement on each page.

Action: standardize on `PageShell > PageHeader > Section/Subsection` with an
optional page action area and a restrained content width.

### P1 - The execution console competes with primary work

The console opens expanded by default and consumes roughly one third of the
window. Its header exposes many equal-weight controls. The collapsed state hides
useful run feedback, and there is no message search or severity filter.

Impact: configuration pages lose vertical space and runtime feedback is either
too dominant or disappears.

Action: default to a compact status strip, preserve splitter/collapse state, and
provide search, severity filter, follow-output, copy, clear, overflow, and
collapse controls in expanded mode.

### P1 - Navigation and top chrome are visually heavy

The mission studio uses a 246 px fixed navigation rail, a title/action header,
and a separate status summary row. ST-LRPS has another fixed sidebar and header
implementation.

Impact: shell chrome consumes substantial space before page content begins.

Action: use a compact application bar, 220-224 px navigation, subtle selected
state, and one shared shell vocabulary.

### P1 - Inline styling blocks accessibility consistency

Text tiers, borders, focus states, disabled states, and minimum control heights
vary by page. Some ST-LRPS strings contain escaped token expressions such as
`{{THEME['fg_soft']}}`, which render as literal invalid QSS values.

Impact: focus visibility and contrast are not reliably inherited; malformed
styles fail silently.

Action: move structural visuals to QSS object names/properties, expose semantic
notice and badge variants, and keep minimum interactive heights in metrics.

### P2 - Forms and metrics are overly card-heavy

Repeated group boxes and nested panels create visual noise. Labels, units, and
field widths are not aligned consistently. Metric displays use multiple
independent card implementations.

Impact: dense engineering data is harder to compare than necessary.

Action: use `FormGrid`, `LabeledField`, `UnitField`, `KeyValueList`, and
`MetricRow`; reserve elevated cards for real grouping or comparison.

### P2 - Responsive behavior is mostly fixed-width

Sidebars, some buttons, and page-local panels rely on fixed widths. Long paths
and localized labels can crowd controls. Scroll ownership differs by page.

Action: make the shell own scrolling, use expanding content columns, cap only
the readable content width, and avoid fixed button widths except compact icon
controls.

## Accessibility Checklist

- Primary text and semantic foregrounds must maintain readable contrast on all
  base surfaces.
- Every interactive control must expose a visible keyboard focus state.
- Controls use a 34 px compact minimum height, with larger primary actions.
- Status is conveyed by text plus color; color is never the only signal.
- Navigation remains usable by keyboard and announces meaningful accessible
  names/tooltips.
- Dense pages preserve logical tab order and do not trap focus in plots or logs.
- Reduced visual motion is the default; no decorative animation is required.

## Validation Targets

- Token and stylesheet contract tests.
- Widget smoke tests for all shared primitives.
- Console buffering, filtering, collapse, and counter tests.
- Offscreen screenshots at 1280x860 and 1024x768.
- Existing launcher, page, session, command-builder, and batch propagation tests remain
  green.

## Phase 2 Follow-up

The visual redesign now routes Mission Studio and ST-LRPS Studio through the
same typed tokens and global QSS. The shell uses responsive 216/188 px
navigation widths, idle headers hide run-only controls, settings dialogs share
one title/description/action hierarchy, and visualization colors have a
separate typed role group. Focus, read-only, disabled, status, warning, error,
success, and primary-action states are represented by semantic selectors rather
than page-specific accent colors.

## Color-Theory / Best-Practice Pass (2026-07-11)

A measurement-driven pass (WCAG contrast matrix, grayscale-luminance ordering,
protan/deutan/tritan simulation over every token pair) produced these durable
outcomes:

- **Visualization token fixes.** `selected_plot` duplicated `earth` exactly
  (#3B86FF, indistinguishable in any vision) — moved to the accent family
  (#6AA9FF). `apoapsis` (#22D3B6) was iso-luminant with `periapsis` and
  `comparison_plot`, identical to `axis_y`, and collapsed into
  `comparison_plot` under CVD simulation — lifted to #7DEEDD (same cool hue
  family, value-separated). Node geometry got its own `orbit_node` role
  (#9F7CFF) instead of borrowing the apoapsis color for three meanings.
- **Verified-clean results.** The severity ladder is luminance-ordered
  (warning 0.52 > error 0.30 > critical 0.24) and survives grayscale. Text
  contrast passes AA on every surface pairing except intentionally-muted
  `fg_disabled` (WCAG-exempt). Segoe UI digits measured tabular-by-default on
  Windows, so no numeric font role is needed. Known latent CVD confusions
  (`selected_plot`~`gravity` deutan, `periapsis`~`sun` tritan) require
  linestyle/marker redundancy if those series ever co-plot.
- **Doc drift guard.** `UI_THEME.md`'s palette table is now enforced against
  the live tokens by `tests/test_ui_theme.py::test_ui_theme_doc_palette_matches_tokens`.
- **Motion.** `MotionTokens` (150/200 ms) joined the foundation; the execution
  console collapse/expand is now an interruptible 200 ms OutCubic drawer slide
  that honors the reduced-motion preference and stays instant pre-show.
- **Glyph discipline.** Color-emoji status icons (job queue, preset combo) were
  replaced with theme-colorable monochrome glyphs or explicit text suffixes;
  redundant "⚠" prefixes were dropped where the component kind already carries
  the state.
