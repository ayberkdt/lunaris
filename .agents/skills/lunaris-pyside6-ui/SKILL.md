---
name: lunaris-pyside6-ui
description: >-
  Implement or fix Lunaris PySide6 desktop UI code the Lunaris way — reusing the
  Lunar Graphite theme and shared widgets, with correct threading, HiDPI, focus,
  and accessibility. Use when adding or editing a page/widget under
  src/lunaris/ui/, wiring signals/slots, running work off the UI thread,
  handling long-running tasks/cancellation, fixing layout/overflow/resize/HiDPI
  issues, validation/disabled/loading/error states, or capturing UI screenshots.
  Trigger on "add a settings field", "the UI freezes during a run", "this page
  doesn't resize", "wire this button", "lunaris-ui". NOT for visual/UX decisions
  (use lunaris-ux-design), accessibility auditing (use accessibility-audit), or
  the Next.js/Three.js web preview (use lunaris-web-3d).
---

# Lunaris PySide6 UI Engineering

Implement against the existing app, not a fresh Qt project. The recurring bugs
are: hard-coded colors, work on the UI thread, and broken focus/resize. Reuse
before you abstract.

## Invocation

Auto-trigger; inline. Take visual/UX direction from `lunaris-ux-design`; hand
contrast/keyboard sign-off to `accessibility-audit`.

## Canonical sources

- `docs/UI_THEME.md`; tokens + shared primitives in `lunaris.ui.core.ui_commons`
  (`THEME`, `LOG_COLORS`, `ORBIT_THEME`, `with_alpha`, color helpers).
- Stylesheet generator: `lunaris.ui.theme.stylesheet.build_app_stylesheet`.
- App + pages: `lunaris.ui.app` (`MainWindow`), `src/lunaris/ui/pages/*`,
  `src/lunaris/ui/widgets/`, `src/lunaris/ui/components/`.
- Patterns reference: `references/pyside6-patterns.md`.
- Entry points: `lunaris-ui`, `lunaris-launcher`, `lunaris-studio`.

## Rules

1. **Theme, never literals.** Every color routes through `THEME`/`ORBIT_THEME`;
   translucency via `with_alpha(THEME["accent"], 0.2)`, never a raw `rgba(...)`.
   Run `scripts/scan_hardcoded_colors.py` before finishing.
2. **Reuse shared widgets.** Use existing primitives in `ui_commons`/`widgets/`
   and the page patterns in `pages/` before creating a new component.
3. **Threading.** Long work (propagation, MC, training, benchmarks) runs off the
   GUI thread (QThread/worker + signals). Never block the event loop; never touch
   widgets from a worker thread — marshal via signals. Provide progress and a
   working **cancel**; preserve user input on failure.
4. **State coverage.** Implement disabled, loading, validation-error, and empty/
   first-run states — not just the happy path. Invalid config combinations are
   blocked or explained, never silently accepted.
5. **HiDPI / resize / focus.** Layouts (not fixed geometry) so panels resize and
   text doesn't clip; verify focus order and keyboard operation; set accessible
   names on icon-only controls.

## Procedure

1. Find the existing page/widget and the token(s) involved; make the smallest
   change consistent with neighbors.
2. Wire signals/slots; move any heavy call to a worker with progress + cancel.
3. Add the missing interaction states.
4. Verify theme/threading/resize/focus; capture a screenshot.

## Verification

- `python -m pytest tests/test_monte_carlo_page.py -q` and other UI-page tests;
  broaden when shared widgets/theme change.
- `python .Codex/skills/lunaris-pyside6-ui/scripts/scan_hardcoded_colors.py src/lunaris/ui`
  → no new raw hex/rgba outside the token modules.
- Launch `lunaris-ui` (needs a display + `ui` extra) and screenshot the page;
  if no display is available, say so rather than claiming a visual check.

## Stop conditions

- A fix needs a literal color or a blocking call on the UI thread → stop and use a
  token / move to a worker.

## Output

A change summary: files touched, tokens used, threading/cancellation handling,
the interaction states covered, and the verification (tests + screenshot, or an
explicit note that no display was available).

## Acceptance

No hard-coded colors; heavy work off the UI thread with cancel; disabled/loading/
error/empty states present; resize/HiDPI/focus verified; UI tests pass; screenshot
captured or its absence explained.
