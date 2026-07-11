# Lunaris UI — Industrial-Grade Roadmap

Phased plan to take the Lunaris desktop UI from "polished" to "industrial tool"
quality (the bar set by STK / GMAT / FreeFlyer / Ansys-class mission-analysis
software). Each step is grounded in concrete code evidence, prioritized by
leverage, and has an explicit acceptance check.

Scope: `src/lunaris/ui/**`, `src/lunaris/ui_foundation/**`. Theme/token work
(Lunar Graphite v4) is considered done and is the baseline for this roadmap.

## Current state (evidence)

Solid foundation already in place:

- Single source of truth for tokens (`ui_foundation/tokens.py` → `palette.py` →
  `stylesheet.py`), 0 hard-coded color literals.
- A real design-system component layer exists in
  `ui/components/primitives.py`: `PageHeader`, `PageShell`, `Section`,
  `Subsection`, `FormGrid`, `LabeledField`, `UnitField`, `KeyValueList`,
  `MetricRow`, `InlineNotice`, `ActionBar`, `Toolbar`, `SegmentedControl`,
  `EmptyState`, `CompactSearchField`, `OverflowMenuButton`.
- Global QSS covers component states (focus / disabled / tables / scrollbars /
  tooltips / tabs).

Core industrial gaps (re-measured 2026-07-11; the original counts below were a
generation stale):

- **Inline styling in pages is largely retired.** Per-page `setStyleSheet`
  counts are now 0–3 (`orbit_config_page` 3, `data_files_page` 2,
  `result_exports_page` 2, `batch_propagation_page` 1, `force_models_page` 1);
  the ~50 remaining call sites are shared infrastructure in `ui_commons`. The
  original Phase 0 driver (36/23/15/12/11 per page) is effectively done; page
  code routes through `primitives.py` + object-name selectors. Residual work is
  the shared-infra sites, not the pages.
- **Keyboard / accessibility is the real gap.** `setTabOrder` is used **0**
  times across `ui/` — tab order is entirely Qt-default. `setAccessibleName`
  is broader than first measured (76 call sites across 13 files) but uneven;
  focus rings on the custom `ToggleSwitch` / `SegmentedControl` are unverified.
- **States/feedback uneven**, custom widgets (`ToggleSwitch`, `CostIndicator`)
  are paint-only (no keyboard/focus), and data tables appear on only 2 pages.
- **Spacing rhythm** is now normalized to the 4px scale and gated by
  `tools/ui/spacing_scan.py`; a snapshot-diff harness
  (`tools/ui/snapshot_suite.py`) guards further visual changes.

## Verification (applies to every phase)

- Snapshot-diff regression check via `tools/ui/snapshot_suite.py`
  (`--baseline` on a known-good tree, then `--compare`); before/after single
  captures still available through `tools/ui/capture_main_window.py`.
- `python tools/ui/spacing_scan.py` → clean (on-scale or allow-listed).
- `python .claude/skills/lunaris-pyside6-ui/scripts/scan_hardcoded_colors.py` → 0.
- `python .claude/skills/accessibility-audit/scripts/contrast_check.py` for any
  new text/control pairing.
- `pytest tests/test_ui_*` green.

---

## Phase 0 — Consistency lock (DONE 2026-07-11)

Problem: two competing "right ways" (component system vs inline styling) cause
drift.

Outcome: page-level inline `setStyleSheet` is down to 0–3 per page (from
36/23/15/12/11); pages route through `primitives.py` + object-names;
`scan_hardcoded_colors` = 0. Remaining inline sites are shared `ui_commons`
infrastructure, tracked separately. Spacing is normalized to the 4px scale and
gated. The next-highest leverage has shifted to **Phase 1 (keyboard)**, where
`setTabOrder` coverage is still zero.

## Phase 1 — Keyboard & interaction (daily professional use)

3. Global shortcuts + command surface: Run/Stop, page switching (Ctrl+1..n),
   focus console / search, save/open (today only a few in `ui/app.py`).
4. Full keyboard navigation: `setTabOrder` across every page, visible focus
   ring, `setAccessibleName/Description` on every interactive control.
5. Make custom widgets keyboard-accessible: `ToggleSwitch` / `SegmentedControl`
   handle Space/Enter, draw a focus ring, verified on HiDPI (currently
   paint-only, no focus/keyboard).

## Phase 2 — State & feedback robustness

6. Standardize loading / busy / empty / error / validation states through
   `InlineNotice` / `EmptyState`; every long-running op is cancelable; the UI
   thread must never freeze (threading audit).
7. Keep scientific honesty always visible: requested-vs-actual backend, units,
   reference frame, and a clear demo-vs-real data distinction in the status
   strip.

## Phase 3 — Data presentation (mission-analysis core)

8. Strong data tables: sort / filter / copy / CSV export, monospace numeric
   alignment, unit-bearing headers — for results / ephemeris / event logs
   (tables exist on only 2 pages today).
9. Units, frames, and provenance mandatory on every physical value (via
   `UnitField`).

## Phase 4 — Density & layout discipline

10. Responsive audit: min sizes, splitters, no overflow/clipping, HiDPI;
    consistent spacing scale; page max-width.
11. Density mode (comfortable / compact) for long sessions.

## Phase 5 — Discoverability & polish

12. Tooltip/help coverage, onboarding empty-states, settings persistence.
13. Icon consistency, reduced-motion support, closing accessibility audit
    (`accessibility-audit` skill).

---

Recommended start: Phase 0 pilot on `force_models_page` — migrate to the
component system and prove visual parity with before/after captures; if the
pilot is clean, apply the same pattern to the remaining pages.
