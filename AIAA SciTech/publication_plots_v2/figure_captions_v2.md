# Figure Captions -- ST-LRPS Validation (v2)

## Figure 1
Speed--accuracy tradeoff for 100 one-day lunar orbit propagation cases.
Markers denote SH20 RK4, ST-LRPS RK4, and SH60 RK4 models.
All position errors are computed relative to the SH200 DOP853 reference solution.
The dashed horizontal line indicates the 1 m threshold.

## Figure 2
Distribution of trajectory-level RMS position error over the 100-case validation ensemble.
(a) Violin and box plots with individual scenario points (jittered).
(b) Empirical cumulative distribution functions on a logarithmic abscissa.
Median RMS values: SH20 = 98.510 m, ST-LRPS = 0.158 m, SH60 = 0.058 m.

## Figure 3
Dependence of RMS position error on initial periapsis altitude and inclination.
Each panel uses an individual logarithmic color scale.
Selected ST-LRPS best, representative, and worst cases are marked in panel (b).

## Figure 4
RIC-frame decomposition of RMS position error.
(a) Median and (b) 95th-percentile component errors for the three models.
Along-track error dominates across all models and percentile levels.

## Figure 5
_Not generated._ Raw surrogate time-history data were not saved to disk
during the original batch run. To produce this figure, re-run the three
selected scenarios (best #55, representative #81, worst #36) with
trajectory output enabled.
