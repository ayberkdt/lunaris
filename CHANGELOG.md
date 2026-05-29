# Changelog

## Unreleased

- Standardized all generated artifacts under a single, git-ignored `outputs/` root with typed subdirectories (`simulations/`, `monte_carlo/`, `missions/`, `gravity_benchmark/`, `training/`, `evaluations/`, `runtime/`, `dataset_reports/`, `datasets/`, `visualization/`). Updated every default output path to match.
- Reworked the ST-LRPS Studio "Gravity Plots" page into a folder-first workflow: pick a results folder, then choose from the models discovered inside it.
- Aligned documentation and configuration (`README.md`, `docs/`, `validation/` docs, `pyproject.toml`, `environment.yml`, `CITATION.cff`) with the unified `lunaris` naming and removed legacy output-directory references.

## 0.1.0

- Introduced the `lunaris` package namespace and `src/` layout.
- Preserved ST-LRPS as `lunaris.surrogate.st_lrps`.
- Added console entry points for CLI, GUI, Studio, Monte Carlo, training, evaluation, and orbit benchmark workflows.
