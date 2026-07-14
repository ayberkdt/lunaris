# Lunaris Documentation

This directory is the working documentation index for Lunaris. The project is a
lunar orbit propagation framework first, with ST-LRPS kept as an optional
advanced subsystem.

## Start Here

| Document | Use it for |
| --- | --- |
| [Getting Started In 10 Minutes](GETTING_STARTED_10_MINUTES.md) | Install, download data, verify it, run one orbit, and inspect a plot |
| [Public API](PUBLIC_API.md) | Which Python modules and console commands are stable enough to use |
| [Data Presets](DATA_PRESETS.md) | Named `lunaris-data` bundles such as `minimal`, `surface`, and `st-lrps-dev` |
| [Architecture](ARCHITECTURE.md) | Layering, dependency directions, config flow, force flags, ST-LRPS boundary |

## Core Framework

| Document | Use it for |
| --- | --- |
| [Algorithm Catalogue](ALGORITHM_CATALOG.md) | Traceability for every implemented algorithm/model: canonical name, verified source, symbols, tests (generated from the registry) |
| [Algorithm Traceability Policy](ALGORITHM_TRACEABILITY_POLICY.md) | How to name, cite, classify and register a new algorithm or model |
| [Force Model Validation](FORCE_MODEL_VALIDATION.md) | Physics and limiting-case validation notes |
| [Gravity Engine External Validation](GRAVITY_ENGINE_EXTERNAL_VALIDATION.md) | Independent gravity/orbit validation harness |
| [Perturbation Budget](PERTURBATION_BUDGET.md) | Acceleration budgets and interpretation |
| [Reproducible Benchmarks](REPRODUCIBLE_BENCHMARKS.md) | Benchmark manifests, provenance, and CI smoke mode |
| [Benchmark Results](BENCHMARK_RESULTS.md) | Accepted benchmark table contract and Sprint 6 metric definitions |
| [Paper-Safe Fail Policy](PAPER_SAFE_POLICY.md) | Which conditions hard-fail under `paper_safe=true` vs warn-and-record in research mode |
| [HPC](HPC.md) | Headless and cluster setup |

## ST-LRPS

| Document | Use it for |
| --- | --- |
| [Config And Artifact Contracts](CONFIG_AND_ARTIFACT_CONTRACTS.md) | Dataset, training, checkpoint, runtime, and benchmark contracts |
| [Capability Matrix](ST_LRPS_CAPABILITY_MATRIX.md) | Which surrogate kind / baseline / training path implements which feature (generated SSOT) |
| [Dataset Pipeline](DATASET_PIPELINE.md) | Dataset generation, validation, quality reports, and split manifests |
| [Validation Hygiene](ST_LRPS_VALIDATION_HYGIENE.md) | Train-only scalers, spatial/OOD splits, artifact compatibility, paper-safe rules |
| [Modularity Audit](ST_LRPS_MODULARITY_AUDIT.md) | ST-LRPS package/UI modularity notes |

## UI And Design

| Document | Use it for |
| --- | --- |
| [UI Theme](UI_THEME.md) | Lunar Graphite tokens and theme contract |
| [UI Design System](UI_DESIGN_SYSTEM.md) | Desktop UI design rules |
| [UI Page Architecture](UI_PAGE_ARCHITECTURE.md) | Page/component layout and ownership |

## Development Notes

| Document | Use it for |
| --- | --- |
| [Frame Handling And Physics Refactor](development/FRAME_HANDLING_AND_PHYSICS_REFACTOR.md) | GPU smoke checklist, optional force-probe boundary, and future third-body generalization notes |

Generated outputs belong under `outputs/` or an explicit runtime output
directory, never under `src/`.
