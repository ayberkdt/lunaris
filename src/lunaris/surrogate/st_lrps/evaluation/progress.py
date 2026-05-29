# st_lrps/evaluation/progress.py
# -*- coding: utf-8 -*-
"""
Structured progress reporting for the orbit-level gravity benchmark.

This module is intentionally dependency-light (standard library only) so it can
be imported both by the heavy CLI harness
(``st_lrps.evaluation.compare_gravity_models``) and by the PySide6/PyQt6 Studio
UI without dragging in torch / scipy / SPICE.

Two machine-parseable line kinds are emitted on stdout:

    [progress] phase=truth current=44 total=100 percent=44.0 elapsed_s=2718 eta_s=3472 message="SH200 DOP853 truth"
    [progress] phase=gpu_model model=sh20 current_step=4320 total_steps=43200 percent=10.0 elapsed_s=744 eta_s=6696 steps_per_s=5.81
    [progress_total] percent=63.4 phase=gpu_model model=sh30 elapsed_s=8123 eta_s=5400

``[progress]`` carries per-phase progress; ``[progress_total]`` carries the
weighted estimate of overall run progress. Both are consumed by
:func:`parse_progress_line`, used by the UI and the tests.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional

__all__ = [
    "PROGRESS_PREFIX",
    "PROGRESS_TOTAL_PREFIX",
    "format_progress",
    "format_progress_total",
    "emit_progress",
    "emit_progress_total",
    "parse_progress_line",
    "compute_eta_s",
    "compute_step_stats",
    "windowed_rate",
    "eta_from_rate",
    "gpu_total_steps",
    "format_duration",
    "format_eta",
    "OverallProgress",
    "StepThrottle",
]

PROGRESS_PREFIX = "[progress]"
PROGRESS_TOTAL_PREFIX = "[progress_total]"

# Keys whose values are rendered as plain integers.
_INT_KEYS = frozenset({"current", "total", "current_step", "total_steps"})
# Keys whose values are rendered as rounded-integer seconds.
_SECONDS_KEYS = frozenset({"elapsed_s", "eta_s"})


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_value(key: str, value: Any) -> str:
    """Render one ``key=value`` token value for a progress line."""
    if key == "message":
        text = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{text}"'
    if key == "percent":
        return f"{float(value):.1f}"
    if key == "steps_per_s":
        return f"{float(value):.2f}"
    if key in _SECONDS_KEYS:
        return f"{int(round(float(value)))}"
    if key in _INT_KEYS:
        return f"{int(round(float(value)))}"
    if isinstance(value, float):
        # Integer-valued floats print without a trailing ".0".
        return f"{int(value)}" if value.is_integer() else f"{value:g}"
    return str(value)


def _format_line(prefix: str, fields: Dict[str, Any]) -> str:
    tokens = [prefix]
    for key, value in fields.items():
        if value is None:
            continue
        tokens.append(f"{key}={_fmt_value(key, value)}")
    return " ".join(tokens)


def format_progress(phase: str, **fields: Any) -> str:
    """Build a ``[progress]`` line. ``message`` (if present) is quoted.

    Field order follows kwargs insertion order, with ``phase`` first.
    """
    ordered: Dict[str, Any] = {"phase": phase}
    ordered.update(fields)
    return _format_line(PROGRESS_PREFIX, ordered)


def format_progress_total(
    percent: float,
    phase: str,
    *,
    model: Optional[str] = None,
    elapsed_s: Optional[float] = None,
    eta_s: Optional[float] = None,
) -> str:
    """Build a ``[progress_total]`` line for the weighted overall estimate."""
    return _format_line(
        PROGRESS_TOTAL_PREFIX,
        {
            "percent": percent,
            "phase": phase,
            "model": model,
            "elapsed_s": elapsed_s,
            "eta_s": eta_s,
        },
    )


def emit_progress(phase: str, **fields: Any) -> None:
    """Print a ``[progress]`` line (flushed)."""
    print(format_progress(phase, **fields), flush=True)


def emit_progress_total(
    percent: float,
    phase: str,
    *,
    model: Optional[str] = None,
    elapsed_s: Optional[float] = None,
    eta_s: Optional[float] = None,
) -> None:
    """Print a ``[progress_total]`` line (flushed)."""
    print(
        format_progress_total(
            percent, phase, model=model, elapsed_s=elapsed_s, eta_s=eta_s
        ),
        flush=True,
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Matches key=value where value is either a double-quoted string (with escapes)
# or a run of non-whitespace characters.
_TOKEN_RE = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')


def _coerce(value: str) -> Any:
    """Turn a raw token value into a float when it looks numeric, else str."""
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    try:
        return float(value)
    except ValueError:
        return value


def parse_progress_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a ``[progress]`` / ``[progress_total]`` line.

    Returns a dict with a ``kind`` key (``"progress"`` or ``"progress_total"``)
    plus every parsed field (numeric values as floats, ``message`` as str), or
    ``None`` if the line is not a progress line.
    """
    if not line:
        return None
    text = line.strip()
    if text.startswith(PROGRESS_TOTAL_PREFIX):
        kind = "progress_total"
        body = text[len(PROGRESS_TOTAL_PREFIX):]
    elif text.startswith(PROGRESS_PREFIX):
        kind = "progress"
        body = text[len(PROGRESS_PREFIX):]
    else:
        return None

    result: Dict[str, Any] = {"kind": kind}
    for key, raw in _TOKEN_RE.findall(body):
        result[key] = _coerce(raw)
    return result


# ---------------------------------------------------------------------------
# Numeric helpers (pure; unit-testable without CUDA)
# ---------------------------------------------------------------------------

def compute_eta_s(elapsed_s: float, percent: float) -> Optional[float]:
    """Linear ETA from elapsed wall-time and percent-complete (0..100)."""
    pct = float(percent)
    if pct <= 1e-9 or pct >= 100.0:
        return None
    return max(0.0, float(elapsed_s)) * (100.0 - pct) / pct


def compute_step_stats(
    current_step: int, total_steps: int, elapsed_s: float
) -> Dict[str, Any]:
    """Percent / steps-per-second / ETA for a fixed-step propagation.

    ``current_step`` is clamped to ``[0, total_steps]`` and ``total_steps`` is
    floored at 1 so the percentage is always well-defined.
    """
    tot = max(1, int(total_steps))
    cur = min(max(0, int(current_step)), tot)
    percent = 100.0 * cur / tot
    elapsed = max(0.0, float(elapsed_s))
    steps_per_s = (cur / elapsed) if elapsed > 0.0 else 0.0
    eta_s: Optional[float] = (
        (tot - cur) / steps_per_s if steps_per_s > 0.0 else None
    )
    return {
        "current_step": cur,
        "total_steps": tot,
        "percent": percent,
        "steps_per_s": steps_per_s,
        "eta_s": eta_s,
    }


def windowed_rate(
    d_step: int, d_t: float, *, fallback_cur: int = 0, fallback_elapsed: float = 0.0
) -> float:
    """Steps/second over a recent window, falling back to the cumulative rate.

    Using the rate over the most recent reporting window (rather than the
    cumulative rate from step 0) keeps the figure — and any ETA derived from it
    — from being dragged down by one-off start-up costs such as JIT compilation
    or the first CUDA kernel launch.
    """
    if d_t > 0.0 and d_step > 0:
        return float(d_step) / float(d_t)
    if fallback_elapsed > 0.0 and fallback_cur > 0:
        return float(fallback_cur) / float(fallback_elapsed)
    return 0.0


def eta_from_rate(remaining_steps: float, steps_per_s: float) -> Optional[float]:
    """Seconds remaining for ``remaining_steps`` at ``steps_per_s`` (None if unknown)."""
    if steps_per_s <= 0.0:
        return None
    return max(0.0, float(remaining_steps)) / float(steps_per_s)


def gpu_total_steps(duration_s: float, output_dt_s: float, rk4_dt_s: float) -> int:
    """Total fixed-step RHS-step count for a GPU batch propagation.

    Mirrors the step plan in ``propagate_gpu_batch_model``: an ``rk4_dt_s``
    larger than ``output_dt_s`` is clamped to the output cadence.
    """
    if output_dt_s <= 0.0 or rk4_dt_s <= 0.0:
        raise ValueError("output_dt_s and rk4_dt_s must be positive")
    eff_rk4 = min(float(rk4_dt_s), float(output_dt_s))
    steps_per_snap = max(1, round(float(output_dt_s) / eff_rk4))
    n_snaps = max(1, round(float(duration_s) / float(output_dt_s)))
    return int(n_snaps * steps_per_snap)


def format_duration(seconds: Optional[float]) -> str:
    """Human elapsed string, e.g. ``"45s"``, ``"12.4 min"`` or ``"1h 56m"``."""
    if seconds is None:
        return "—"
    s = max(0.0, float(seconds))
    if s < 60.0:
        return f"{s:.0f}s"
    if s < 3600.0:
        return f"{s / 60.0:.1f} min"
    h, rem = divmod(int(round(s)), 3600)
    m = rem // 60
    return f"{h}h {m:02d}m"


def format_eta(seconds: Optional[float]) -> str:
    """Human ETA string, e.g. ``"1h 51m"``, ``"7m 30s"`` or ``"45s"``."""
    if seconds is None:
        return "—"
    s = int(round(max(0.0, float(seconds))))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


# ---------------------------------------------------------------------------
# Overall (weighted, monotonic) progress
# ---------------------------------------------------------------------------

class OverallProgress:
    """Weighted estimate of total run progress across ordered phases.

    Phases are weighted (default: ``truth`` 40%, ``gpu`` 50%, ``report`` 10%).
    Calling :meth:`update` for a phase implicitly marks all earlier phases
    complete, so the reported percentage is monotonically non-decreasing as the
    run advances in phase order. When truth is reused from cache the caller
    constructs the object without a ``truth`` phase, which collapses that weight
    and lets the GPU phase dominate the bar.
    """

    DEFAULT_WEIGHTS = {"truth": 0.40, "gpu": 0.50, "report": 0.10}

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        *,
        start_time: Optional[float] = None,
    ) -> None:
        raw = dict(weights) if weights else dict(self.DEFAULT_WEIGHTS)
        positive = {k: float(v) for k, v in raw.items() if float(v) > 0.0}
        if not positive:
            raise ValueError("OverallProgress needs at least one positive weight")
        total = sum(positive.values())
        self._weights = {k: v / total for k, v in positive.items()}
        self._order = list(self._weights.keys())
        self._fraction = {k: 0.0 for k in self._order}
        self._last_percent = 0.0
        self._t0 = start_time if start_time is not None else time.perf_counter()

    @property
    def phases(self) -> list:
        return list(self._order)

    def update(self, phase: str, fraction: float) -> float:
        """Set ``phase`` progress (0..1), mark earlier phases done, return %.

        Unknown phases (e.g. ``truth`` when collapsed) are ignored and the last
        percentage is returned unchanged.
        """
        if phase not in self._weights:
            return self._last_percent
        frac = min(1.0, max(0.0, float(fraction)))
        idx = self._order.index(phase)
        for i, name in enumerate(self._order):
            if i < idx:
                self._fraction[name] = 1.0
            elif i == idx:
                self._fraction[name] = max(self._fraction[name], frac)
        pct = 100.0 * sum(
            self._weights[name] * self._fraction[name] for name in self._order
        )
        self._last_percent = max(self._last_percent, pct)
        return self._last_percent

    @property
    def percent(self) -> float:
        return self._last_percent

    def elapsed_s(self, now: Optional[float] = None) -> float:
        now = now if now is not None else time.perf_counter()
        return max(0.0, now - self._t0)

    def eta_s(self, now: Optional[float] = None) -> Optional[float]:
        return compute_eta_s(self.elapsed_s(now), self._last_percent)


# ---------------------------------------------------------------------------
# Step throttle for the GPU inner loop
# ---------------------------------------------------------------------------

class StepThrottle:
    """Decide when to emit progress inside a tight fixed-step loop.

    Emits at most ~100 times over the whole run (``total_steps // 100`` step
    gate) and no more often than ``min_interval_s`` wall-seconds, so neither
    fast nor slow runs flood the log.
    """

    def __init__(
        self,
        total_steps: int,
        *,
        min_interval_s: float = 5.0,
    ) -> None:
        self.total_steps = max(1, int(total_steps))
        self.min_interval_s = float(min_interval_s)
        self.step_interval = max(1, self.total_steps // 100)
        self._last_step = 0
        self._last_t: Optional[float] = None

    def needs_time_check(self, current_step: int) -> bool:
        """Cheap step-only gate: True when a wall-clock check may be due.

        Lets a tight fixed-step loop skip ``time.perf_counter()`` (and the
        :meth:`update` call) on the vast majority of steps, consulting the clock
        only once the step-count interval has been reached. Returns True on the
        very first call so a subsequent :meth:`update` still arms its baseline
        exactly as before. This method never emits and never mutates throttle
        state; :meth:`update` remains the sole authority on whether to emit and
        still enforces both the step and minimum wall-clock interval gates.
        """
        if self._last_t is None:
            return True
        return (current_step - self._last_step) >= self.step_interval

    def update(self, current_step: int, now: float) -> bool:
        """Return True (and arm the next window) when a progress emit is due."""
        if self._last_t is None:
            self._last_t = now
            self._last_step = current_step
            return False
        if (current_step - self._last_step) < self.step_interval:
            return False
        if (now - self._last_t) < self.min_interval_s:
            return False
        self._last_step = current_step
        self._last_t = now
        return True
