---
name: lunaris-ux-design
description: >-
  Design or revise the Lunaris desktop UX with evidence, not taste — information
  architecture, workflows, color, typography, layout, and interaction for a dark
  scientific mission-analysis app. Use when asked to "redesign", "improve the
  layout/UX", "make this page clearer", "rework the navigation/workspace", "the
  settings page is confusing", "choose colors/spacing/typography", or "research
  how pro tools do X". Works from the existing Lunar Graphite system and real
  user tasks; produces problem→principle→token-change→verification proposals, not
  "make it modern". NOT for writing the PySide6 widget code (use lunaris-pyside6-ui),
  contrast/keyboard auditing (use accessibility-audit), or chart design (use
  scientific-figures).
---

# Lunaris UX Design (evidence-based)

Lunaris is professional scientific software used for long sessions on dense data.
Design decisions must be justified by a perceptual/interaction principle and the
existing design system — never by "it looks more modern". Reject decorative
glassmorphism, neon glow, gratuitous gradients, and "card soup".

## Invocation

Auto-trigger; inline for proposals. For broad "research how pro tools do X",
consider an isolated research pass and summarize findings, not raw dumps.

## Operate from the existing system first

- **Canonical:** `docs/UI_THEME.md` (Lunar Graphite) and the live tokens in
  `lunaris.ui.core.ui_commons` (`THEME` for Qt widgets, `ORBIT_THEME` for the
  OpenGL preview, `LOG_COLORS` for the log). Restrained accent = orbital blue
  `#6AA9FF`; teal `#6EE7C8` success; amber `#E7B86A` **warning only**; elevation
  `bg_space → bg_shell → bg_card → bg_card_alt`.
- Before proposing any color/spacing, decide: **reuse** an existing token,
  **extend** it, or **revise** it — and say which. Never introduce page-local
  hard-coded colors.
- The principle references live in: `references/color-and-contrast.md`,
  `references/typography-and-density.md`, `references/layout-and-interaction.md`,
  and the research synthesis in `docs/UI_UX_RESEARCH.md`. Read the relevant one;
  don't restate it from memory.

## Procedure

1. **Inspect first.** Identify the page/workflow, the user (mission analyst / ML
   researcher), the data density, and the task: configuration → execution →
   results → comparison. Look at the real page under `src/lunaris/ui/pages/`.
2. **Frame the problem** in user terms (a scan-path, a mode error, a hierarchy
   failure), not "ugly".
3. **Apply a principle.** Cite the specific one (value-based hierarchy over hue;
   proximity/common-region grouping; Hick–Hyman for choice count; Fitts for target
   size; progressive disclosure for expert/novice). Hierarchy must survive
   grayscale — test by desaturating.
4. **Separate the three zones:** configuration, execution, and results must be
   visually distinct; never let demo/synthetic data look like solver output (see
   scientific-UX rules in `docs/UI_UX_RESEARCH.md`).
5. **Propose token-level changes**, mapped to `THEME`/`ORBIT_THEME`, with the
   accessibility and consistency impact stated.
6. **Show units, frames, and provenance.** Every physical value exposes its unit;
   reference frame and requested-vs-actual backend stay visible; scientific
   warnings are blocking/inline, not decorative toasts.

## Verification

- Desaturate the proposal mentally/with a tool — hierarchy must hold.
- Contrast every new text/control pairing (hand off to `accessibility-audit` /
  `accessibility-audit/scripts/contrast_check.py`).
- Confirm tokens resolve in `ui_commons` and no raw hex was introduced
  (`lunaris-pyside6-ui/scripts/scan_hardcoded_colors.py`).
- Capture before/after screenshots when a change lands (via `lunaris-pyside6-ui`).

## Stop conditions

- A proposal relies on a new accent color, gradient, or glow without a principle
  and a token decision → stop and re-ground in Lunar Graphite.
- A change would make synthetic/demo data visually indistinguishable from solver
  output → reject.

## Output

A design proposal: problem (user terms) → principle (cited) → token/component
change (mapped to THEME/ORBIT_THEME) → accessibility & consistency impact →
verification method. Keep recommendations few and defensible.

## Acceptance

Each recommendation has a problem, a cited principle, a concrete token/component
change, an accessibility note, and a verification step; the existing system is
reused/extended before anything new; hierarchy survives grayscale.
