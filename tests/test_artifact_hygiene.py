"""Repository hygiene: no generated artifacts or data files committed.

Complements ``test_repo_hygiene.py`` (which guards stale project identity and a
fixed set of generated report filenames). This module enforces the *binary
artifact* and *source-tree purity* contracts that keep the repo lightweight and
reproducible for research use:

- no model/data binaries (``.pt .pth .ckpt .h5 .hdf5 .npz .npy``) are committed
  outside the explicitly allowed fixture locations,
- no data files are committed under ``src/`` (the package tree is code-only),
- no generated output directories (``outputs/ runs/ checkpoints/ evals/
  reports/ mc_results/``) are committed.

Exceptions are explicit and minimal: ``tests/fixtures/`` and
``examples/st_lrps_minimal_artifact/``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Minimal, explicit exceptions. Tiny deterministic fixtures may live here.
ALLOWED_PREFIXES = (
    "tests/fixtures/",
    "examples/st_lrps_minimal_artifact/",
)

BINARY_ARTIFACT_SUFFIXES = (".pt", ".pth", ".ckpt", ".h5", ".hdf5", ".npz", ".npy")

# Data-like payloads that must never live inside the importable package tree.
SRC_DATA_SUFFIXES = BINARY_ARTIFACT_SUFFIXES + (".csv", ".tab", ".bsp", ".tpc", ".tls", ".bpc")

# Directory names that only ever hold generated run output.
GENERATED_DIRS = ("outputs", "runs", "checkpoints", "evals", "reports", "mc_results")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _committed_files(root: Path) -> list[str]:
    """Return committed paths as posix strings relative to the repo root.

    Uses ``git ls-files`` so untracked local scratch files never trip the guard;
    falls back to a filesystem walk if git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:  # pragma: no cover - git nearly always present in CI
        out: list[str] = []
        for path in root.rglob("*"):
            if path.is_file() and ".git" not in path.parts:
                out.append(path.relative_to(root).as_posix())
        return out


def _is_allowed(rel_posix: str) -> bool:
    return any(rel_posix.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def test_no_committed_binary_model_artifacts():
    root = _project_root()
    offenders = [
        rel for rel in _committed_files(root)
        if rel.lower().endswith(BINARY_ARTIFACT_SUFFIXES) and not _is_allowed(rel)
    ]
    assert not offenders, (
        "Binary model/data artifacts must not be committed outside "
        f"{ALLOWED_PREFIXES}:\n  " + "\n  ".join(sorted(offenders))
    )


def test_no_data_files_under_src():
    root = _project_root()
    offenders = [
        rel for rel in _committed_files(root)
        if rel.startswith("src/") and rel.lower().endswith(SRC_DATA_SUFFIXES)
    ]
    assert not offenders, (
        "The src/ package tree must contain code only, not data payloads:\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_no_committed_generated_output_directories():
    root = _project_root()
    offenders = []
    for rel in _committed_files(root):
        if _is_allowed(rel):
            continue
        parts = rel.split("/")
        if any(d in parts for d in GENERATED_DIRS):
            offenders.append(rel)
    assert not offenders, (
        "Generated run-output directories must not be committed "
        f"({GENERATED_DIRS}):\n  " + "\n  ".join(sorted(offenders))
    )
