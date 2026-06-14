---
name: accessibility-audit
description: >-
  Audit Lunaris UI accessibility against WCAG-derived criteria for a dark desktop
  scientific app. Use when asked to "check accessibility", "is the contrast ok",
  "can this be used with the keyboard", "are these colors colorblind-safe",
  "review focus order", or before shipping a UI change. Produces findings with
  severity, location, evidence, impact, and remediation — covering contrast,
  keyboard/focus, color-independent meaning, labels, text scaling/HiDPI, reduced
  motion, tooltip/icon-only misuse, error identification, and table/chart
  alternatives. NOT for visual design choices (use lunaris-ux-design) or writing
  the widget code (use lunaris-pyside6-ui).
---

# Accessibility Audit

A focused audit that turns "looks fine" into specific, located, fixable findings.
Default to WCAG-derived thresholds, adapted to a desktop dark theme.

## Invocation

Auto-trigger; inline. Read-only audit — report findings; the fixes belong to
`lunaris-pyside6-ui` / `lunaris-web-3d`.

## Thresholds (WCAG 2.2, verified 2026-06-13; see `docs/UI_UX_RESEARCH.md`)

- **Text contrast (1.4.3):** ≥ 4.5:1 normal; ≥ 3:1 large (≥ 18pt / 14pt bold,
  ≈ 24px / 18.5px). AAA (1.4.6) = 7:1.
- **Non-text contrast (1.4.11):** ≥ 3:1 for UI component boundaries, focus
  indicators, and meaningful graphical objects (chart lines, markers).
- **Color is not the only channel (1.4.1):** status/series must also differ by
  shape, label, icon, or position.

## Checklist (full list: `checklists/a11y-checklist.md`)

1. **Contrast.** Every text/background and control/background pair meets the bar.
   Use `scripts/contrast_check.py <hex_fg> <hex_bg>`; resolve colors from `THEME`.
2. **Keyboard.** Every action reachable and operable by keyboard; logical focus
   order; **visible** focus indicator (≥ 3:1).
3. **Color independence.** Warning vs error vs success distinguished beyond hue;
   plot series carry markers/labels, not color alone.
4. **Labels & names.** Inputs have associated labels; icon-only buttons have
   accessible names; tooltips are not the *only* carrier of essential info.
5. **Scaling / HiDPI.** Text and layout survive OS scaling without clipping.
6. **Reduced motion.** Animations (incl. the 3D preview) honor reduced-motion.
7. **Errors.** Errors are identified in text (not color alone), located, and
   recoverable; disabled controls remain legible.
8. **Tables/charts.** Dense tables are scannable; charts have a text/numeric
   alternative or accessible summary.

## Procedure

1. Enumerate the surfaces in scope (page, dialog, chart, 3D view).
2. Walk each checklist item; gather concrete evidence (token pair + measured
   ratio, focus path, screenshot region).
3. Rank findings by severity (blocker / serious / moderate / minor).

## Verification

- `python .Codex/skills/accessibility-audit/scripts/contrast_check.py 6AA9FF 0E1116`
  (and every flagged pair) — exit code 0 means it meets the requested level.
- Keyboard-only walk of the workflow (or note that no display was available).

## Output (required per finding)

`[SEVERITY] location — criterion — evidence (measured ratio / focus gap) —
impact — remediation (token/widget change)`. End with a prioritized summary.

## Stop conditions

- A color pair cannot be resolved to tokens → report it as "untraceable color"
  (itself a finding), don't guess the hex.

## Acceptance

Every checklist area covered; each finding has severity, location, measured
evidence, impact, and a concrete remediation; contrast numbers come from the
script, not estimation.
