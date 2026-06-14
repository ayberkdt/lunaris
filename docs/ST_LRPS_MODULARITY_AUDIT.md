# ST-LRPS Internal Modularity Audit

Phase 2.5 measured the two largest ST-LRPS areas before deciding whether a
behavior-preserving split would reduce coupling.

## Evidence

Largest Python modules at the audit point:

| Module | Approximate lines | Observation |
|---|---:|---|
| `ui/studio_parts/training_pages.py` | 3,900 | Large page/controller, tightly coupled to Qt widgets |
| `evaluation/cli.py` | 3,060 | Large evaluation workflow; gravity benchmarking is already split into `_gravity_benchmark/` |
| `ui/studio_parts/data_pages.py` | 2,520 | Widget-heavy data workspace plus one reusable HDF5 metadata helper |
| `ui/studio_parts/common_widgets.py` | 2,410 | High fan-in shared Qt widget surface |

The useful low-risk boundary was HDF5 metadata inspection. Training, runtime,
evaluation, and main-window modules imported the entire `data_pages` module,
including wildcard imports, while only the page modules needed
`_introspect_h5`.

## Decision And Change

`dataset_introspection.py` is now the small Qt-independent metadata surface.
Training, runtime, and evaluation pages import it directly; the main window no
longer imports `data_pages` through a wildcard. This removes four unnecessary
edges to the 2.5K-line data workspace without changing widget behavior.

No further split is made in this phase:

- The gravity benchmark already has explicit `types`, `compute`, `results_io`,
  `plotting`, `metrics`, and `modes` modules.
- The remaining large UI files own cohesive Qt page/controller lifecycles.
  Splitting classes merely to reduce line counts would create more cross-file
  state and signal wiring during the feature freeze.
- `common_widgets.py` still has high fan-in and wildcard consumers. Its cleanup
  should be paired with the planned legacy/alias consolidation so one canonical
  public widget surface is chosen once rather than introducing another interim
  compatibility layer.

The repository test `test_st_lrps_ui_modularity.py` prevents the removed
`data_pages` wildcard coupling from returning.
