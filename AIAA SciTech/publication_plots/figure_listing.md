# Publication Figures - ST-LRPS Validation (AIAA SciTech)

Generated from 100-case one-day lunar orbit validation batch.
Three-model comparison: GPU SH20 RK4, GPU ST-LRPS RK4, GPU SH60 RK4
vs SH200 DOP853 reference.

---

## Data Sources

| Source | Description |
|--------|-------------|
| `stlrps_ders_cikisi_100/` | ST-LRPS per-scenario metrics (correct model checkpoint) |
| `gpu_sh60_sh20_stlrps_100/` | SH20 & SH60 per-scenario metrics (same 100 scenarios, seed=42) |

---

## Figure 1: Speed-Accuracy Tradeoff
`fig1_speed_accuracy_tradeoff.{png,pdf}`

Four models compared on total runtime vs median RMS position error.
ST-LRPS achieves sub-meter accuracy (0.158 m median) at runtime
comparable to the lowest-fidelity SH20 model.

---

## Figure 2: RMS Error Distribution
`fig2_rms_error_distribution.{png,pdf}`

(a) Violin+box plot comparing SH20, ST-LRPS, SH60 error distributions.
(b) ST-LRPS detailed histogram + ECDF.
Median = 0.158 m, P95 = 0.689 m, Max = 0.960 m.

---

## Figure 3: Error Sensitivity
`fig3_error_sensitivity_alt_inc.{png,pdf}`

Three-panel scatter: periapsis altitude vs inclination colored by RMS error.
Shows SH20 errors ~100 m, ST-LRPS ~0.16 m, SH60 ~0.06 m.

---

## Figure 4: RIC Error Decomposition
`fig4_ric_error_decomposition.{png,pdf}`

Grouped bar chart comparing Radial/Along-track/Cross-track error
for all three models. Along-track dominates for all models.

---

## Figure 5: Selected Case Time Histories
`fig5_selected_case_time_history.{png,pdf}`

Three panels from actual simulation data:
- (a) Best: Scenario #55
- (b) Representative: Scenario #81
- (c) Worst: Scenario #36

---

## Figure 6: Summary Comparison Table
`fig6_summary_table.{png,pdf}`

Publication-ready table with key metrics for all three models.

---

## Notes

- All position errors in meters (converted from km).
- 300 DPI, TrueType fonts (PDF type 42).
- SH20 and SH60 data from `gpu_sh60_sh20_stlrps_100` run
  (same 100 scenarios, seed=42, same SH200 DOP853 reference).
