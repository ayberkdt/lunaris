# Lunaris UI — Follow-up Plan (post Phase 0–5)

Continuation of [UI_INDUSTRIAL_ROADMAP.md](UI_INDUSTRIAL_ROADMAP.md). Phases 0–5
of that roadmap are **complete and verified**; this document captures the
remaining gaps found in the closing system review and the next work, prioritized
by leverage with explicit acceptance checks.

## Current state (verified baseline)

- All seven user-visible pages (Orbit, Forces, Propagation, Results & Export,
  Telemetry, Data, Batch Propagation) migrated onto the component system
  (`PageShell` / `Section` / `InlineNotice` / `EmptyState` / `DataTable`).
  Page- and widget-level inline `setStyleSheet` = 0 (only `__main__` test blocks
  and justified token-based exceptions remain).
- Keyboard: `ToggleSwitch` focusable + Space/Enter + contrast-checked focus ring;
  global page-switch (Ctrl+1..N) and console-focus shortcuts.
- State/feedback: validation/empty/error via `InlineNotice` / `EmptyState`;
  reduced-motion preference honored by progress widgets.
- Density (comfortable/compact) + readable page max-width + window-geometry
  persistence (QSettings).
- Verification baseline (re-runnable): per-file `pytest` sweep with timeouts
  (~387 UI tests green), `scan_hardcoded_colors.py` = 0 over 30 files,
  `contrast_check.py` for new pairings, offscreen screenshots via
  `tools/ui/capture_main_window.py`.

## Verification (applies to every step below)

- `python -m pytest tests/test_ui_*.py tests/test_monte_carlo_page.py -q` green
  (run per-file with a timeout so one hang never masks results).
- `python .claude/skills/lunaris-pyside6-ui/scripts/scan_hardcoded_colors.py src/lunaris/ui` → 0.
- `python .claude/skills/accessibility-audit/scripts/contrast_check.py <fg> <bg>`
  for every new text/control pairing (focus indicators need ≥ 3:1, text ≥ 4.5:1).
- Before/after offscreen capture of any changed surface.

---

## Step A — Migrate modal dialogs to the component system (highest leverage)

Problem: the pages are clean, but the settings **dialogs** still bypass the
component system with inline QSS, so visual drift now lives in the dialogs.

Evidence (inline `setStyleSheet` remaining, page-level = 0):

- `force_models_page.py`: `GravitySettingsDialog` (tab QSS, group-box borders,
  hint labels, adaptive-table QSS), `AdaptiveDegreeDialog` (form frame, table
  QSS, row labels), `AlbedoSettingsDialog` (form frame, checkbox/note labels).
- `mission_propagation_page.py`: `SolverSettingsDialog` (tolerance frame),
  `SpacecraftBusDialog` (properties frame).

Work:

1. Replace bespoke group-box/frame QSS with `Section` (or the global
   `QFrame#section` / `QFrame[studioSurface="true"]`) and object-name labels
   (`sectionTitle`, `fieldHint`, `fieldLabel`).
2. Replace per-dialog `QTabWidget` QSS with the global tab styling.
3. Replace ad-hoc data tables (e.g. adaptive altitude→degree editor) with
   `DataTable` where read-only, keeping the editable editor as a table but
   routing its styling through the global `#dataTable` rule.
4. Keep the shared dialog contract intact (`objectName == "settingsDialog"`,
   `dialogTitle` / `dialogDescription`, exactly one `kind="primary"` button) —
   `tests/test_ui_phase2_layout.py::test_settings_dialogs_share_*` must stay green.

Acceptance: inline `setStyleSheet` in `force_models_page.py` /
`mission_propagation_page.py` ≈ 0 (outside `__main__`); dialog-contract tests
green; before/after captures at visual parity.

## Step B — Keyboard & focus completeness (Phase 1 tail)

1. Explicit `setTabOrder` across each page where creation order ≠ visual order
   (audit each page; most are already correct by construction).
2. Finish `setAccessibleName` / `setAccessibleDescription` coverage: every
   interactive control (inputs, combos, spinboxes, icon-only buttons) — not just
   toggles — has an accessible name; tooltips add help without being the *only*
   carrier of essential info.
3. Verify a full keyboard-only walk of the Run workflow (or note no display).

Acceptance: keyboard-only walk reaches every action; `accessibleName()`
non-empty on all interactive controls in a scripted offscreen check.

## Step C — Scientific-honesty status strip (Phase 2 tail)

The mission status strip shows requested gravity only. Make scientific context
always visible:

1. **Requested-vs-actual backend** — after a run, surface the resolved backend
   (and any fallback) from the run/ensemble metadata next to the requested one.
2. **Reference frame** — show the working frame for the active page; resolve the
   label from a real source (constants / config), do not hard-code a guess.
3. **Demo-vs-real** — a clear flag when a surface is illustrative (e.g. the
   two-body orbit preview already labels itself; mirror that in the strip).

Acceptance: backend/frame come from real metadata (no invented strings); a run
that falls back to CPU visibly shows requested≠actual.

## Step D — Data-table rollout + copy/CSV everywhere (Phase 3 tail)

`DataTable` exists and is applied to the MC backend-compare table only.

1. Adopt `DataTable` for the **artifact browser** (or add Ctrl+C / "Copy CSV"
   to the existing tree) and the **MC metrics** panel (Metric | Value, monospace
   values, CSV export).
2. Ensure every physical value carries unit-bearing headers (`UnitField` /
   `(label, unit)` headers) and monospace numeric alignment.

Acceptance: results/ephemeris/event surfaces support sort + copy + CSV; numeric
columns are right-aligned monospace with unit-bearing headers.

## Step E — Discoverability & polish tail (Phase 5 tail)

1. Tooltip/help coverage sweep on remaining controls and an onboarding empty
   state where a page can be empty on first run.
2. Icon consistency pass (one icon family, consistent sizes/weights via
   `get_icon`).
3. Unit audit: replace any remaining `px` font sizes with `pt` tokens
   (one fixed in telemetry; sweep for others).
4. Closing accessibility re-audit with the `accessibility-audit` skill after
   Steps A–D land.

Acceptance: `accessibility-audit` closing pass has no blocker/serious findings;
no raw `px` font sizes outside intentional cases.

---

## Suggested order

A (dialogs) → C (honesty strip) → D (tables) → B (keyboard/focus) → E (polish),
with the standard verification gate after each step. Steps are independent enough
to ship one at a time.

## Out of scope / intentional exceptions

- Token-based semantic markers (e.g. the colored artifact bullets in
  Results & Export) — allowed local color per the design system's QSS rules.
- `setStyleSheet("")` ghost-field clearing in the orbit page (no color literal).
- `__main__` standalone-test blocks in page modules.
