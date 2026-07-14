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

Current state (re-measured 2026-07-14):

- **Inline styling in pages is retired.** Per-page `setStyleSheet` counts are
  0–3; the remaining call sites are shared infrastructure in `ui_commons`.
  Page code routes through `primitives.py` + object-name selectors.
- **Keyboard layer landed.** `FormGrid` wires deterministic tab order
  automatically on every `add_row` (`apply_tab_order`, `tests/test_ui_keyboard.py`);
  `ToggleSwitch` paints a state-aware focus ring and `SegmentedControl`
  segments inherit `QPushButton:focus` from the global QSS; all menu and
  window-level shortcuts come from one inventory
  (`ui/core/shortcuts.py`) with a headless uniqueness/conflict test.
- **Form validation contract live.** `FormGrid`/`LabeledField` expose
  `set_error` / `focus_first_invalid` (`tests/test_ui_forms.py`); the
  propagation page pilots the blur-display/instant-clear behaviour and gates
  F5/preflight on `validate_inputs()`.
- **Results zone P1a shipped.** `ui/core/results_index.py` (read-only,
  depth-bounded, symlink-skipping) + the Run History card on Results & Export:
  run selector, KPI summary with explicit units, provenance (config sha256),
  figure/report gallery, DEMO badge for demonstration output.
- **Multi-series plots carry a non-color channel.** `VisualizationTokens`
  defines paired color+dash cycles consumed together
  (`ui/core/plot_style.py`); `tools/ui/color_audit.py` gates the invariant
  under protan/deutan/tritan simulation (also run from
  `tests/test_plot_series_style.py`). Series roles: truth=solid,
  surrogate=dashed, comparison=dotted.
- **Spacing rhythm** is normalized to the 4px scale and gated by
  `tools/ui/spacing_scan.py`; a snapshot-diff harness
  (`tools/ui/snapshot_suite.py`) guards further visual changes.
- **Data tables**: the shared `DataTable` (sort/copy/CSV/unit headers/numeric
  alignment) is used on `force_models_page` and `batch_propagation_page`; the
  artifact browser tree has copy/CSV parity.

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
gated.

## Phase 1 — Keyboard & interaction (DONE 2026-07-14)

3. ~~Global shortcuts + command surface~~ — single inventory in
   `ui/core/shortcuts.py` (menu labels + key sequences consumed from it;
   conflicts caught by test).
4. ~~Full keyboard navigation~~ — `FormGrid.apply_tab_order()` re-wires the
   chain on every added row; keyboard round-trip tests in
   `tests/test_ui_keyboard.py`. `setAccessibleName` coverage remains uneven on
   older pages (open follow-up).
5. ~~Custom-widget keyboard access~~ — `ToggleSwitch` (QAbstractButton: Space
   works, state-aware focus ring painted) and `SegmentedControl` (radio-group
   arrow navigation, QSS focus) verified by tests.

## Phase 2 — State & feedback robustness (PARTIAL)

6. Validation states standardized at the primitive level
   (`set_error`/`focus_first_invalid`, blur-display contract, propagation-page
   pilot + preflight gate). Remaining: roll the pilot out to
   `force_models_page` and `batch_propagation_page`; a fresh threading audit
   found long operations (propagation, frozen search, batch) already run in
   QProcess/workers with cancellation.
7. Keep scientific honesty always visible: requested-vs-actual backend, units,
   reference frame, and a clear demo-vs-real data distinction in the status
   strip. The Run History card now badges DEMO output and shows provenance.

## Phase 3 — Data presentation (mission-analysis core) (PARTIAL)

8. ~~Strong data tables~~ — shared `DataTable` primitive (sort / copy / CSV /
   unit headers / right-aligned monospace numerics) used by the force-model
   preview and the batch backend-comparison table.
9. Units, frames, and provenance mandatory on every physical value (via
   `UnitField`); the Results-zone KPI path formats units through one helper.

## Phase 4 — Density & layout discipline

10. Responsive audit: min sizes, splitters, no overflow/clipping, HiDPI;
    consistent spacing scale; page max-width.
11. Density mode (comfortable / compact) for long sessions.

## Phase 5 — Discoverability & polish

12. Tooltip/help coverage, onboarding empty-states, settings persistence.
13. Icon consistency, reduced-motion support, closing accessibility audit
    (`accessibility-audit` skill).

---

Next highest-leverage work (2026-07-14): finish the Phase 2 form-contract
rollout (`force_models_page`, then `batch_propagation_page` — largest page,
migrate last with snapshot cover), close the `setAccessibleName` gaps on older
pages, then Phase 4 (density/responsive audit) and Phase 5 (discoverability,
closing accessibility audit).
