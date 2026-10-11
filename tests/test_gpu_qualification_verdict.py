"""``tools/gpu_qualification.py`` must not report PASS when nothing ran on a GPU.

Regression test: the verdict was ``pytest exit code == 0``, and pytest exits
0 when every ``requires_cuda`` test is skipped (no CUDA visible), so a CPU-only
machine reported a PASS.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "gpu_qualification.py"


def _load():
    spec = importlib.util.spec_from_file_location("gpu_qualification_tool", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="{e}" failures="{f}" skipped="{s}" tests="{t}"/></testsuites>
"""


@pytest.mark.parametrize(
    ("t", "f", "e", "s", "expected"),
    [(5, 0, 0, 5, 0), (5, 0, 0, 2, 3), (0, 0, 0, 0, 0)],
)
def test_count_passed_reads_the_junit_report(tmp_path: Path, t, f, e, s, expected) -> None:
    tool = _load()
    report = tmp_path / "r.xml"
    report.write_text(_JUNIT.format(t=t, f=f, e=e, s=s), encoding="utf-8")
    assert tool.count_passed(report) == expected


def test_all_skipped_run_is_a_failure() -> None:
    tool = _load()
    code, verdict = tool.qualification_verdict(0, 0)
    assert code != 0
    assert "FAIL" in verdict


def test_unreadable_report_is_a_failure(tmp_path: Path) -> None:
    tool = _load()
    assert tool.count_passed(tmp_path / "missing.xml") is None
    code, _ = tool.qualification_verdict(0, None)
    assert code != 0


def test_passing_run_and_failing_run() -> None:
    tool = _load()
    assert tool.qualification_verdict(0, 3)[0] == 0
    assert tool.qualification_verdict(1, 3)[0] == 1


def test_help_does_not_run_the_suite() -> None:
    proc = subprocess.run(
        [sys.executable, str(_TOOL), "--help"], capture_output=True, text=True, timeout=60, check=False
    )
    assert proc.returncode == 0
    assert "requires_cuda" in proc.stdout
    assert "commit:" not in proc.stdout
