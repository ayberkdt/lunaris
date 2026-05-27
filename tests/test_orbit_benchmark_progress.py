"""Tests for orbit-level benchmark progress reporting (logging/UI only).

These cover the dependency-light progress module
(:mod:`st_lrps.evaluation.progress`) and the Studio status-strip parser. No real
propagation is run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from st_lrps.evaluation import progress as P  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Progress line parsing
# ---------------------------------------------------------------------------

def test_parse_phase_line_extracts_all_fields():
    line = ("[progress] phase=gpu_model model=sh20 current_step=4320 "
            "total_steps=43200 percent=10.0 elapsed_s=744 eta_s=6696 "
            "steps_per_s=5.81")
    info = P.parse_progress_line(line)
    assert info is not None
    assert info["kind"] == "progress"
    assert info["phase"] == "gpu_model"
    assert info["model"] == "sh20"
    assert info["current_step"] == 4320
    assert info["total_steps"] == 43200
    assert info["percent"] == pytest.approx(10.0)
    assert info["eta_s"] == pytest.approx(6696.0)
    assert info["steps_per_s"] == pytest.approx(5.81)


def test_parse_total_line_extracts_phase_model_eta():
    line = "[progress_total] percent=63.4 phase=gpu_model model=sh30 elapsed_s=8123 eta_s=5400"
    info = P.parse_progress_line(line)
    assert info is not None
    assert info["kind"] == "progress_total"
    assert info["percent"] == pytest.approx(63.4)
    assert info["phase"] == "gpu_model"
    assert info["model"] == "sh30"
    assert info["eta_s"] == pytest.approx(5400.0)


def test_parse_truth_line_extracts_current_total():
    line = ('[progress] phase=truth current=44 total=100 percent=44.0 '
            'elapsed_s=2718 eta_s=3472 message="SH200 DOP853 truth"')
    info = P.parse_progress_line(line)
    assert info["phase"] == "truth"
    assert info["current"] == 44
    assert info["total"] == 100
    assert info["message"] == "SH200 DOP853 truth"


def test_parse_message_with_spaces_and_quotes():
    line = P.format_progress("report", message='plots: "best" & worst')
    info = P.parse_progress_line(line)
    assert info["message"] == 'plots: "best" & worst'


def test_parse_non_progress_returns_none():
    for line in (
        "[gpu-batch][sh20] step 4320/43200 | 10.0% | elapsed 12.4 min",
        "[truth] Scenario 044/100 done | id=12",
        "ordinary stdout text",
        "",
    ):
        assert P.parse_progress_line(line) is None


def test_format_parse_roundtrip():
    line = P.format_progress(
        "gpu_model", model="sh60", current_step=10, total_steps=20,
        percent=50.0, steps_per_s=8.62,
    )
    info = P.parse_progress_line(line)
    assert info["model"] == "sh60"
    assert info["current_step"] == 10
    assert info["percent"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# 2. GPU step progress math
# ---------------------------------------------------------------------------

def test_compute_step_stats_basic():
    stats = P.compute_step_stats(4320, 43200, 744.0)
    assert stats["percent"] == pytest.approx(10.0)
    assert stats["steps_per_s"] == pytest.approx(4320 / 744.0)
    assert stats["eta_s"] == pytest.approx((43200 - 4320) / (4320 / 744.0))


def test_compute_step_stats_clamps_and_handles_zero_elapsed():
    # Over-count is clamped to total; zero elapsed yields no rate / no ETA.
    stats = P.compute_step_stats(999, 100, 0.0)
    assert stats["current_step"] == 100
    assert stats["percent"] == pytest.approx(100.0)
    assert stats["steps_per_s"] == 0.0
    assert stats["eta_s"] is None


def test_gpu_total_steps_formula():
    # 1 day at 60 s output cadence, 10 s RK4 step:
    #   n_snaps = 86400/60 = 1440 ; steps_per_snap = 60/10 = 6 ; total = 8640
    assert P.gpu_total_steps(86400.0, 60.0, 10.0) == 8640


def test_gpu_total_steps_clamps_large_rk4_step():
    # rk4 step larger than output cadence is clamped to the output cadence.
    assert P.gpu_total_steps(600.0, 60.0, 120.0) == P.gpu_total_steps(600.0, 60.0, 60.0)


def test_gpu_total_steps_rejects_nonpositive():
    with pytest.raises(ValueError):
        P.gpu_total_steps(100.0, 0.0, 10.0)


def test_compute_eta_s_endpoints():
    assert P.compute_eta_s(100.0, 0.0) is None
    assert P.compute_eta_s(100.0, 100.0) is None
    assert P.compute_eta_s(100.0, 50.0) == pytest.approx(100.0)


def test_step_throttle_respects_step_and_time_gates():
    # Gates are measured from the last armed/emitted point.
    th = P.StepThrottle(1000, min_interval_s=5.0)  # step_interval = 10
    assert th.update(0, now=0.0) is False          # first call only arms (0, 0.0)
    assert th.update(5, now=3.0) is False          # step gate: 5 < 10
    assert th.update(12, now=3.5) is False         # time gate: 3.5 s < 5 s
    assert th.update(15, now=6.0) is True          # both gates met -> re-arm (15, 6.0)
    assert th.update(20, now=7.0) is False         # step gate: 5 < 10
    assert th.update(30, now=12.0) is True         # both gates met again


def test_step_throttle_interval_floored_at_one():
    th = P.StepThrottle(5)
    assert th.step_interval == 1


# ---------------------------------------------------------------------------
# 3. Overall (weighted, monotonic) progress
# ---------------------------------------------------------------------------

def test_overall_progress_is_monotonic():
    ov = P.OverallProgress({"truth": 0.40, "gpu": 0.50, "report": 0.10})
    seq = []
    # Include a deliberate regression (0.3 after 0.6) which must NOT lower total.
    seq.append(ov.update("truth", 0.0))
    seq.append(ov.update("truth", 0.6))
    seq.append(ov.update("truth", 0.3))   # regression within phase
    seq.append(ov.update("gpu", 0.5))
    seq.append(ov.update("gpu", 1.0))
    seq.append(ov.update("report", 0.5))
    seq.append(ov.update("report", 1.0))
    assert all(b >= a - 1e-9 for a, b in zip(seq, seq[1:])), seq
    assert seq[-1] == pytest.approx(100.0)


def test_overall_progress_advancing_phase_completes_earlier_ones():
    ov = P.OverallProgress({"truth": 0.40, "gpu": 0.50, "report": 0.10})
    # Jumping straight to gpu implicitly completes truth (40%).
    pct = ov.update("gpu", 0.0)
    assert pct == pytest.approx(40.0)


def test_overall_progress_truth_cache_collapse():
    # When truth is cached the caller omits the truth phase; gpu dominates.
    ov = P.OverallProgress({"gpu": 0.50, "report": 0.10})
    assert ov.update("truth", 1.0) == 0.0          # unknown phase ignored
    pct = ov.update("gpu", 0.5)
    # gpu weight renormalises to 0.5/0.6 ; half-done => ~41.7%
    assert pct == pytest.approx(100.0 * (0.5 / 0.6) * 0.5)
    assert ov.update("gpu", 1.0) == pytest.approx(100.0 * 0.5 / 0.6)


def test_overall_progress_requires_a_weight():
    with pytest.raises(ValueError):
        P.OverallProgress({"truth": 0.0})


# ---------------------------------------------------------------------------
# 4. Human formatting helpers
# ---------------------------------------------------------------------------

def test_format_eta_hours_minutes_seconds():
    assert P.format_eta(3600 + 51 * 60) == "1h 51m"
    assert P.format_eta(7 * 60 + 30) == "7m 30s"
    assert P.format_eta(45) == "45s"
    assert P.format_eta(None) == "—"


def test_format_duration():
    assert P.format_duration(45) == "45s"
    assert P.format_duration(744).endswith("min")
    assert P.format_duration(None) == "—"


# ---------------------------------------------------------------------------
# 5. Studio status-strip parser (Qt; skipped if no binding available)
# ---------------------------------------------------------------------------

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    try:
        from st_lrps.ui.studio_parts.qt_common import QApplication
    except Exception:  # pragma: no cover - no Qt binding present
        pytest.skip("No Qt binding available")
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def tab(qapp):
    from st_lrps.ui.studio_parts.orbit_benchmark_pages import OrbitBenchmarkTab
    t = OrbitBenchmarkTab()
    yield t
    t.deleteLater()


def test_studio_strip_updates_from_progress_lines(tab):
    lines = [
        '[progress] phase=scenario current=100 total=100 percent=100.0 message="Scenarios ready"',
        '[progress] phase=truth current=44 total=100 percent=44.0 elapsed_s=2718 eta_s=3472 message="SH200 DOP853 truth"',
        '[progress_total] percent=17.6 phase=truth elapsed_s=2718 eta_s=12700',
        '[gpu-batch][sh20] step 4320/43200 | 10.0% | elapsed 12.4 min | ETA 1h 51m | 5.8 steps/s',
        '[progress] phase=gpu_model model=sh20 current_step=4320 total_steps=43200 percent=10.0 elapsed_s=744 eta_s=6696 steps_per_s=5.81',
        '[progress_total] percent=63.4 phase=gpu_model model=sh20 elapsed_s=8123 eta_s=5400',
    ]
    # Drive through the real ProcessPane append path (same as live stdout).
    for ln in lines:
        tab.runner.append(ln)

    assert tab._st_phase.text() == "GPU model"   # humanized phase label
    assert tab._st_model.text() == "SH20"        # humanized model label
    assert tab._st_phase_pct.text() == "10.0%"
    assert tab._st_overall_pct.text() == "63.4%"
    assert tab._st_steps.text() == "5.8"
    # progress_total flips the overall bar to determinate 0..100.
    assert tab.overall_bar.maximum() == 100
    assert tab.overall_bar.value() == 63


def test_studio_strip_ignores_plain_logs(tab):
    # Establish a known overall value, then feed non-progress lines.
    tab.runner.append('[progress_total] percent=42.0 phase=truth elapsed_s=10 eta_s=10')
    assert tab._st_overall_pct.text() == "42.0%"
    for ln in (
        "[truth] Scenario 044/100 done",
        "[gpu-batch] Model 01/06 | SH20 RK4 starting ...",
        "ordinary stdout line",
        "",
    ):
        tab.runner.append(ln)  # must not raise or change overall %
    assert tab._st_overall_pct.text() == "42.0%"


def test_studio_parse_progress_never_raises(tab):
    # Malformed-ish lines must be swallowed by the parser hook.
    for ln in ("[progress]", "[progress_total]", "[progress] phase=", None and ""):
        if ln is None:
            continue
        tab._parse_progress(ln)
