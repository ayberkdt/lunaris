# orbit_benchmarks/

Paper-safe orbit benchmark outputs, one subfolder per benchmark config, written
by `--stage orbit-benchmark`: `orbit_benchmark_metrics.csv`,
`orbit_benchmark_scenario_results.csv`, `orbit_benchmark_runtime.csv`,
`orbit_benchmark_summary.md`, plus `benchmark_manifest.json` and
`validation_report.json`.

Paper-safe mode forbids synthetic data, contract-free artifacts, and target-mode
mismatches. It requires a strict domain. The high-degree SH DOP853 model is a
numerical reference, not truth.
