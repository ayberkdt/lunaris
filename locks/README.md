# Reproducible dependency locks

`pyproject.toml` remains the **single source of truth** for version *ranges*.
The files in this directory are fully pinned, hash-verified resolutions of those
ranges, intended for **reproducible Paper-evidence and HPC runs** where the exact
transitive dependency set must be identical across machines.

| Lock file | Equivalent to | Use for |
| --- | --- | --- |
| `requirements-hpc-linux-py311.lock.txt` | `pip install .[hpc]` | headless HPC / GPU nodes (torch + h5py + engine core, no Qt UI) |
| `requirements-all-linux-py311.lock.txt` | `pip install .[all]` | full workstation stack (ML + Qt UI + reports + dev/test tooling) |

## Canonical target

The locks are resolved for **Linux (`x86_64-unknown-linux-gnu`) + CPython 3.11** —
the environment used by CI guardrails, the `lunaris-st-lrps-paper-evidence`
benchmark, and Linux HPC nodes. Day-to-day development on other platforms should
keep using `pip install -e ".[all]"`; the locks exist for *reproducibility*, not
for replacing the editable install.

## Install from a lock

```bash
# exact, hash-verified environment (fails if any artifact hash differs)
pip install --require-hashes -r locks/requirements-hpc-linux-py311.lock.txt

# the project itself is intentionally NOT in the lock; add it without deps:
pip install --no-deps -e .
```

### CUDA note

On Linux the default PyPI `torch` wheel pinned here is CUDA-enabled, so the lock
installs directly on CUDA nodes. To target a *specific* CUDA build instead,
install the matching `torch` from the PyTorch index first, then install the lock
with `--no-deps` for the remaining packages.

## Regenerating

Locks are generated with [`uv`](https://docs.astral.sh/uv/) so they can be
resolved for the Linux target from any host OS. Re-run after changing
dependencies in `pyproject.toml`:

```bash
uv pip compile pyproject.toml --extra hpc \
  --python-version 3.11 --python-platform linux --generate-hashes \
  -o locks/requirements-hpc-linux-py311.lock.txt

uv pip compile pyproject.toml --extra all \
  --python-version 3.11 --python-platform linux --generate-hashes \
  -o locks/requirements-all-linux-py311.lock.txt
```

The exact command is also recorded in each lock file's header.
