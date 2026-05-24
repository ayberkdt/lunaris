# Revision Notes -- v1 to v2

## General
- Switched to AIAA journal figure style: 8--10 pt fonts, white background,
  thin spines, no oversized bold titles.
- Adopted colorblind-safe palette (Wong 2011): vermilion (SH20), blue (ST-LRPS),
  bluish-green (SH60).  Distinct marker shapes and line styles used throughout.
- All figures exported as vector PDF + 300 DPI PNG with `pdf.fonttype = 42`.
- Figure sizes chosen so that text is 8--10 pt when inserted at 0.9\linewidth.

## Figure 1 -- Speed--Accuracy Tradeoff
- Removed the artificial SH200 reference point (was plotted at 1e-4 m).
- Removed the decorative Pareto arrow.
- Removed heavy annotation boxes; replaced with small direct labels.
- Added a subtle 1 m horizontal guide line.
- Reduced figure size to single-column width (3.5 x 2.8 in).

## Figure 2 -- RMS Error Distribution
- Replaced colored x-tick labels with standard black labels.
- Removed large median-value annotations placed directly on violins.
- Added semi-transparent jittered strip plot for individual scenario points.
- Replaced histogram+twin-axis ECDF with a clean ECDF-only panel (b)
  showing all three models on a log x-axis.
- Numerical values (median, P95, max) moved to caption / LaTeX table.

## Figure 3 -- Error Sensitivity
- Changed to logarithmic color normalisation (LogNorm) per panel.
- Reduced marker size and outline weight.
- Replaced inline text labels for selected cases with a compact legend
  on the ST-LRPS panel only.
- Made "individual color scales" explicit in the caption.
- Ensured consistent x/y limits across all three panels.

## Figure 4 -- RIC Decomposition
- Converted from grouped vertical bar chart to horizontal dot/lollipop.
- Removed numeric value labels above every bar.
- Key insight (along-track dominance) is visible from layout; noted in caption.

## Figure 5 -- Time Histories
- Removed embedded PNG screenshots (`plt.imread` / `imshow`).
- Confirmed that raw surrogate trajectories are not saved to disk
  (only `truth/sh200_dop853_trajectories.npz` exists).
- Figure skipped in this version; revision notes explain how to regenerate.

## Figure 6 -- Summary Table
- Replaced matplotlib table image with a LaTeX `booktabs` table
  (`table_validation_summary.tex`).
