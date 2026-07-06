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

Core industrial gaps:

- **Pages bypass the component system with ad-hoc inline styling.** Inline
  `setStyleSheet` counts per page: `force_models_page` 36, `batch_propagation_page`
  23, `orbit_config_page` 15, `data_files_page` 12, `result_exports_page` 11.
  `docs/UI_DESIGN_SYSTEM.md` mandates object-names/properties over inline QSS;
  pages violate this heavily → visual drift + maintenance burden.
- **Keyboard / accessibility is thin.** Shortcuts only in `ui/app.py`;
  `setTabOrder` effectively absent; `setAccessibleName/Description` in ~6 files.
- **States/feedback uneven**, custom widgets (`ToggleSwitch`, `CostIndicator`)
  are paint-only (no keyboard/focus), and data tables appear on only 2 pages.

## Verification (applies to every phase)

- Before/after screenshots via `tools/ui/capture_main_window.py`.
- `python .claude/skills/lunaris-pyside6-ui/scripts/scan_hardcoded_colors.py` → 0.
- `python .claude/skills/accessibility-audit/scripts/contrast_check.py` for any
  new text/control pairing.
- `pytest tests/test_ui_*` green.

---

## Phase 0 — Consistency lock (highest leverage; start here)

Problem: two competing "right ways" (component system vs inline styling) cause
drift.

1. Migrate pages onto the `primitives.py` components + object-names, busiest
   first: `force_models_page` (36) → `batch_propagation_page` (23) →
   `orbit_config_page` (15) → `data_files_page` (12) → `result_exports_page`
   (11). Replace inline `setStyleSheet` with `Section` / `FormGrid` /
   `InlineNotice` / object-name selectors.
2. Acceptance: inline `setStyleSheet` in pages ≈ 0; `scan_hardcoded_colors` = 0;
   before/after captures at visual parity.

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
