"""Local recurring GPU qualification runner.

The `cuda-nightly.yml` workflow runs the `requires_cuda` suite on CI, but only if
a self-hosted GPU runner is registered. Until one is, this script provides an
equivalent *local* qualification: it runs the `requires_cuda` suite and writes a
timestamped, provenance-stamped log so the CPU/GPU parity, float32/float64 drift
bound, and (data-permitting) real-artifact validation are exercised on a real
device on a recurring cadence.

Schedule it weekly with Windows Task Scheduler (or cron on Linux); see
`docs/backend_matrix.md` ("GPU qualification"). Each run appends a log under
``outputs/gpu_qualification/`` and exits non-zero if the suite fails, so a
scheduler can surface regressions.

Run: ``python tools/gpu_qualification.py``
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "gpu_qualification"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _device_banner() -> str:
    """Best-effort CUDA/torch identity for the log header (never raises)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "torch present but CUDA NOT available -> requires_cuda tests will SKIP"
        name = torch.cuda.get_device_name(0)
        return f"torch {torch.__version__}; CUDA device: {name}"
    except Exception as exc:  # noqa: BLE001 - diagnostic banner only
        return f"torch/CUDA unavailable: {type(exc).__name__}: {exc}"


def count_passed(junit_xml: Path) -> int | None:
    """Passed test count from a pytest JUnit XML report (None if unreadable)."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(junit_xml).getroot()
    except (OSError, ET.ParseError):
        return None
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    passed = 0
    for suite in suites:
        total = int(suite.get("tests", 0))
        not_passed = sum(int(suite.get(key, 0)) for key in ("failures", "errors", "skipped"))
        passed += total - not_passed
    return passed


def qualification_verdict(returncode: int, passed: int | None) -> tuple[int, str]:
    """Exit code and verdict line.

    pytest exits 0 when every ``requires_cuda`` test is skipped (no CUDA
    visible), so a zero exit code alone is not a qualification: at least one
    test must have passed.
    """
    if returncode != 0:
        return returncode, f"pytest exit code: {returncode} (FAIL)"
    if not passed:
        return 1, "pytest exit code: 0 but no requires_cuda test passed (FAIL: nothing was validated on a GPU)"
    return 0, f"pytest exit code: 0, {passed} requires_cuda test(s) passed (PASS)"


def run_qualification() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    log_path = OUT_DIR / f"qualification_{stamp}.log"
    junit_path = OUT_DIR / f"qualification_{stamp}.junit.xml"

    banner = _device_banner()
    header = [
        f"Lunaris GPU qualification — {stamp}",
        f"commit: {_git_commit()}",
        f"device: {banner}",
        "-" * 72,
    ]
    print("\n".join(header))

    # -rs reports skip reasons: a green run that actually SKIPPED everything (no
    # CUDA visible) is distinguishable from one that truly validated the GPU.
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests/", "-m", "requires_cuda", "-q", "-rs",
            f"--junitxml={junit_path}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    body = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")
    exit_code, verdict = qualification_verdict(int(proc.returncode), count_passed(junit_path))

    log_path.write_text("\n".join(header) + "\n" + body + "\n" + verdict + "\n", encoding="utf-8")
    print(body)
    print(verdict)
    print(f"Log written to: {log_path}")
    return exit_code


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(
        description="Run the requires_cuda suite and write a provenance-stamped log "
        "under outputs/gpu_qualification/. Fails unless at least one test passed."
    ).parse_args()
    raise SystemExit(run_qualification())
