---
name: scientific-figures
description: >-
  Create or review Lunaris scientific figures (matplotlib/pyqtgraph) with correct
  units, frames, scales, and honest encodings — not generic plotting. Use when
  asked to "plot/figure/chart" an orbit, altitude history, orbital elements,
  position/velocity error, RIC decomposition, acceleration residuals,
  altitude/lat/lon-binned error, runtime comparison, ablation, training curve,
  Monte Carlo envelope, covariance, perturbation budget, gravity-degree
  sensitivity, or truth-vs-surrogate comparison, or to "write a figure caption".
  Every figure declares its analytical question, source data, units, frame, and
  scale. NOT for UI widget styling (use lunaris-ux-design / lunaris-pyside6-ui) or
  the 3D web preview (use lunaris-web-3d).
---

# Lunaris Scientific Figures

A figure is an argument. Generic plotting advice produces misleading axes,
decorative 3D, and unitless labels. This skill ties each figure to a question and
to Lunaris data semantics.

## Invocation

Auto-trigger; inline. Use matplotlib (a core dependency) / pyqtgraph; do **not**
pull in seaborn/plotly or a journal-template meta-skill. For colormaps, prefer
perceptually-uniform sequential maps (e.g. viridis) and avoid jet/rainbow
(`docs/UI_UX_RESEARCH.md`, color section).

## Canonical sources

- `src/lunaris/analysis/`, `src/lunaris/visualization/` (existing plotting +
  RIC/error/report helpers — reuse before writing new).
- `docs/PERTURBATION_BUDGET.md`, `docs/BENCHMARK_RESULTS.md`.
- Figure catalog: `references/figure-catalog.md` (per-figure question, data,
  units, frame, scale, legend, caption, pitfalls — read the entry for your figure).

## Rules for every figure

1. **State the question** the figure answers before plotting.
2. **Units on every axis/label** (m, m/s, m/s², s, deg, km-altitude). No bare
   numbers.
3. **Frame explicit.** Position/velocity errors declare the frame (inertial vs
   Moon-fixed) and, for RIC, the radial/in-track/cross-track convention.
4. **Honest scale.** Don't truncate axes to exaggerate; log scale only when the
   data spans decades and is labeled as log; never hide normalization.
5. **Don't conflate regimes.** Interpolation vs spatial generalization vs altitude
   extrapolation results get distinct series/panels and labels (ties to
   `st-lrps-evidence-audit`). Truth vs surrogate must be visually distinguished;
   demo/synthetic data is labeled as such.
6. **Uncertainty where meaningful.** MC envelopes/covariance get bands; do **not**
   add error bars to deterministic single-trajectory comparisons.
7. **2D for quantitative comparison.** No decorative 3D for what is a 2D
   comparison; reserve 3D for genuine trajectory geometry.
8. **Self-contained caption.** Frame, units, model kind, degree, scenario/seed,
   and what to conclude — readable without the body text.

## Procedure

1. Read the catalog entry; identify the source artifact and its columns/units.
2. Reuse an existing plotting helper if one fits; else write a minimal one under
   `analysis`/`visualization` or a scratch script (outputs under `outputs/`).
3. Apply the rules; export PNG/PDF/SVG as the context needs.
4. Write the caption and a one-line validation of the encoding.

## Verification

- Re-read axes: every one has a unit and an honest range.
- Confirm the data source matches the claim (no truth mislabeled as surrogate).
- Colorblind check: distinguishable in grayscale / with CVD simulation.

## Stop conditions

- The figure would imply generalization from an interpolation split, or conflate
  truth and surrogate → stop and fix the framing.

## Output

The figure (saved under `outputs/`), a self-contained caption, and a short note on
data source, units, frame, scale choice, and uncertainty treatment.

## Acceptance

Question stated; units + frame on everything; honest scale; regimes/truth-vs-
surrogate distinct; uncertainty only where meaningful; caption self-contained;
perceptually-sound colors.
