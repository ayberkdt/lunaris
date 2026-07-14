# Contributing to Lunaris

Thanks for your interest in improving Lunaris. This project is a lunar
orbit-propagation and ST-LRPS surrogate-gravity research codebase, so changes are
held to a scientific-defensibility bar: results must be reproducible and claims
must be traceable to real artifacts.

## Development setup

Lunaris targets Python 3.10–3.12.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"      # runtime + GPU/ML + UI + dev tooling
pre-commit install            # run the linters automatically on commit
```

If you only need the core library and tooling (no GPU/UI), `pip install -e ".[dev]"`
is enough for most physics/contract work; the heavier `torch`/`PySide6` extras are
optional and their tests skip cleanly when absent.

## Quality gates

Every change must keep these green. They run in CI and in the pre-commit hook.

```bash
ruff check .                 # lint (also `ruff format` for formatting)
mypy                         # type-check the core scope: common, core, physics
lint-imports                 # import-linter: enforce the layered architecture
pytest -m "not slow and not requires_data and not requires_cuda"
```

Notes:

- **mypy scope** is intentionally `src/lunaris/{common,core,physics}` (see
  `[tool.mypy]`). The ST-LRPS package is excluded from the type gate; do not
  chase its pre-existing out-of-scope errors.
- **Architecture** is enforced by `import-linter` contracts in
  `[tool.importlinter]` (e.g. `physics` never imports `core`/`ui`; `common`
  stays dependency-light). New cross-layer imports will fail `lint-imports`.
- **Test markers** (`pytest.ini_options`): `slow`, `integration`,
  `requires_data` (SPICE kernels / gravity files), `requires_cuda`,
  `requires_torch`, `requires_pyshtools`. Tests must skip cleanly — never raise —
  when their prerequisite is unavailable, so an unfiltered run on a bare machine
  stays green.
- The suite is shuffled by `pytest-randomly`; do not rely on test ordering.

## Scientific-correctness expectations

- Keep float64 reference paths exact; document any tolerance you assert and why.
- State units, frames (inertial vs Moon-fixed), and time systems explicitly.
- Never present synthetic/quick-mode output as a benchmark result. Paper-grade
  runs use the `paper` run-preset and a generalization split.
- Add or extend tests with each change; a new perturbation, frame/time
  transform, integrator, or CPU/GPU kernel needs a correctness check, not just a
  smoke test.
- Register new algorithms and physical models in the traceability system. A new
  named algorithm, model, integrator, interpolation/sampling method, neural
  architecture, or scientific data product needs an entry in
  `docs/algorithms/algorithm_registry.yaml` with a verified primary source. Run
  `python tools/algorithm_registry.py validate` and `... generate`, then commit
  the registry, `references/references.bib`, and the generated
  `docs/ALGORITHM_CATALOG.md` together. See
  [Algorithm Traceability Policy](docs/ALGORITHM_TRACEABILITY_POLICY.md).

## Dependency locks

Pinned environments live in `locks/*.lock.txt` (the canonical lockfiles). If you
change `pyproject.toml` dependencies, regenerate the affected lock and mention it
in your PR. CI runs a lock-freshness check.

## Pull requests

- Branch off `main`; keep the change focused.
- Describe what changed, why, and how you verified it (commands + results).
- Make sure `ruff`, `mypy`, `lint-imports`, and the default `pytest` selection
  pass locally before opening the PR.

## Reporting issues

Open a GitHub issue at <https://github.com/ayberkdt/lunaris/issues> with a minimal
reproduction, the command you ran, and the observed vs expected behavior. For
security-sensitive reports, follow [SECURITY.md](SECURITY.md) instead.
