"""
Pure-Python training metrics utilities for the ST-LRPS dashboard.

No Qt or GPU dependencies — all functions are testable in isolation.

Provides:
- compute_auto_log_interval  — adaptive batch logging frequency
- ETAEstimator               — robust remaining-time estimation
- TrainingLogParser           — regex-based structured log parsing
- TrainingMetricsStore        — accumulator for parsed training records
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


# ── Phase 7: Auto log interval ────────────────────────────────────────────

def compute_auto_log_interval(
    total_batches: int,
    target_updates: int = 10,
) -> int:
    """Return a log-every-N-batches value that yields ~target_updates per epoch.

    Rules
    -----
    * Always log the first and last batch (caller responsibility).
    * Returns ``max(1, ceil(total_batches / target_updates))``.

    Examples
    --------
    >>> compute_auto_log_interval(1000)
    100
    >>> compute_auto_log_interval(87)
    9
    >>> compute_auto_log_interval(5)
    1
    """
    if total_batches <= 0:
        return 1
    if target_updates <= 0:
        target_updates = 10
    return max(1, math.ceil(total_batches / target_updates))


# ── Phase 8: Epoch guard (debounces repeated "Epoch X/Y" lines) ────────────

class EpochGuard:
    """Debounce epoch start/end transitions parsed from log lines.

    The training log prints ``Epoch X/Y`` on many lines within the same epoch
    (epoch banner, every batch row, validation header, summaries). Feeding each
    occurrence to :class:`ETAEstimator.on_epoch_start` would restart the epoch
    timer repeatedly and corrupt the ETA. This guard returns ``True`` only on
    the first transition into a given epoch number.
    """

    def __init__(self) -> None:
        self._started: Optional[int] = None
        self._ended: Optional[int] = None

    def reset(self) -> None:
        self._started = None
        self._ended = None

    def should_start(self, epoch: int) -> bool:
        """True the first time ``epoch`` is seen as a start; False on repeats."""
        if epoch != self._started:
            self._started = epoch
            return True
        return False

    def should_end(self, epoch: int) -> bool:
        """True the first time ``epoch`` is seen as an end; False on repeats."""
        if epoch != self._ended:
            self._ended = epoch
            return True
        return False


# ── Phase 9: ETA Estimator ────────────────────────────────────────────────

class ETAEstimator:
    """Robust ETA estimator for multi-epoch training.

    Tracks epoch start/end wall-clock times and uses an exponential
    moving average (EMA) to smooth epoch durations.

    Parameters
    ----------
    ema_alpha : float
        Smoothing factor for exponential moving average (0 < alpha ≤ 1).
        Smaller values give more weight to history, larger values react
        faster to recent epochs.  Default 0.3.
    """

    def __init__(self, ema_alpha: float = 0.3):
        self._alpha = max(0.01, min(1.0, ema_alpha))

        self._total_epochs: int = 0
        self._current_epoch: int = 0
        self._batch_progress: float = 0.0  # 0..1 within current epoch

        self._epoch_start: Optional[float] = None
        self._ema_epoch_s: Optional[float] = None
        self._completed_epochs: int = 0
        self._training_start: Optional[float] = None

    # ── Public API ───────────────────────────────────────────────────

    def set_total_epochs(self, n: int) -> None:
        self._total_epochs = max(0, n)

    def on_training_start(self) -> None:
        self._training_start = time.monotonic()
        self._epoch_start = time.monotonic()
        self._completed_epochs = 0
        self._ema_epoch_s = None

    def on_epoch_start(self, epoch: int) -> None:
        self._current_epoch = epoch
        self._epoch_start = time.monotonic()
        self._batch_progress = 0.0

    def on_batch_progress(self, batch: int, total_batches: int) -> None:
        if total_batches > 0:
            self._batch_progress = min(1.0, batch / total_batches)

    def on_epoch_end(self, epoch: int) -> None:
        self._current_epoch = epoch
        self._batch_progress = 1.0
        if self._epoch_start is not None:
            dur = time.monotonic() - self._epoch_start
            if dur > 0:
                if self._ema_epoch_s is None:
                    self._ema_epoch_s = dur
                else:
                    self._ema_epoch_s = (
                        self._alpha * dur + (1 - self._alpha) * self._ema_epoch_s
                    )
            self._completed_epochs += 1

    def elapsed_seconds(self) -> Optional[float]:
        if self._training_start is None:
            return None
        return time.monotonic() - self._training_start

    def remaining_seconds(self) -> Optional[float]:
        """Estimated seconds remaining.  Returns None if insufficient data."""
        if self._total_epochs <= 0:
            return None

        remaining_epochs = self._total_epochs - self._current_epoch
        if remaining_epochs <= 0:
            return 0.0

        # After at least 1 completed epoch, use EMA
        if self._ema_epoch_s is not None and self._completed_epochs >= 1:
            # Subtract progress within the current epoch
            current_remaining = (1.0 - self._batch_progress) * self._ema_epoch_s
            future_full = max(0, remaining_epochs - 1) * self._ema_epoch_s
            total = current_remaining + future_full
            return max(0.0, total)

        # During first epoch, estimate from batch progress
        if self._epoch_start is not None and self._batch_progress > 0.05:
            elapsed_in_epoch = time.monotonic() - self._epoch_start
            est_epoch_dur = elapsed_in_epoch / self._batch_progress
            current_remaining = (1.0 - self._batch_progress) * est_epoch_dur
            future_full = max(0, remaining_epochs - 1) * est_epoch_dur
            total = current_remaining + future_full
            return max(0.0, total)

        return None  # Insufficient data

    def estimated_finish_time(self) -> Optional[datetime]:
        """Wall-clock datetime when training is expected to finish."""
        rem = self.remaining_seconds()
        if rem is None:
            return None
        return datetime.now() + timedelta(seconds=rem)

    def current_epoch_seconds(self) -> Optional[float]:
        """Wall-clock seconds spent in the current (in-progress) epoch."""
        if self._epoch_start is None:
            return None
        return max(0.0, time.monotonic() - self._epoch_start)

    def average_epoch_seconds(self) -> Optional[float]:
        """Smoothed average epoch duration (EMA), or None before any epoch ends."""
        return self._ema_epoch_s

    def format_elapsed(self) -> str:
        e = self.elapsed_seconds()
        if e is None:
            return "--:--:--"
        return _fmt_duration(e)

    def format_remaining(self) -> str:
        r = self.remaining_seconds()
        if r is None:
            return "Estimating…"
        return _fmt_duration(r)

    def format_current_epoch(self) -> str:
        c = self.current_epoch_seconds()
        if c is None:
            return "--:--:--"
        return _fmt_duration(c)

    def format_avg_epoch(self) -> str:
        a = self.average_epoch_seconds()
        if a is None:
            return "Estimating…"
        return _fmt_duration(a)

    def format_finish(self) -> str:
        ft = self.estimated_finish_time()
        if ft is None:
            return "Estimating…"
        return format_finish_time(ft)


def format_finish_time(ft: datetime, now: Optional[datetime] = None) -> str:
    """Format an estimated finish datetime.

    HH:MM when the finish is today, otherwise YYYY-MM-DD HH:MM.
    """
    now = now or datetime.now()
    if ft.date() == now.date():
        return ft.strftime("%H:%M")
    return ft.strftime("%Y-%m-%d %H:%M")


def _fmt_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Phase 8: Training Log Parser ──────────────────────────────────────────

@dataclass
class TrainingRecord:
    """A single normalized training event."""
    timestamp: str = ""
    epoch: int = 0
    phase: str = ""        # "train", "val", "checkpoint", "system"
    batch: int = 0
    total_batches: int = 0
    progress_pct: Optional[float] = None
    loss_opt: Optional[float] = None
    loss_ref: Optional[float] = None
    loss_u: Optional[float] = None
    loss_a: Optional[float] = None
    direction_loss: Optional[float] = None
    cos_sim: Optional[float] = None
    angular_deg: Optional[float] = None
    lr: Optional[float] = None
    samples_per_s: Optional[float] = None
    eta_s: Optional[float] = None
    memory: str = ""
    event: str = ""        # "batch", "epoch_end", "val_summary", "checkpoint_saved",
                           # "best_updated", "warning", "error", "info"
    message: str = ""
    severity: str = "info"  # "info", "success", "warning", "error"
    score: Optional[float] = None
    best_score: Optional[float] = None
    best_epoch: Optional[int] = None


class TrainingLogParser:
    """Parse structured log lines into TrainingRecord objects.

    Designed to run in parallel with the existing LiveLossPlot.parse_line()
    regex pipeline — this parser feeds the structured progress table and
    KPI cards, while LiveLossPlot continues to feed chart data.
    """

    # ── Compiled regexes ──────────────────────────────────────────────

    _RE_EPOCH = re.compile(
        r"Epoch\s*(?:\[\s*)?(\d+)\s*/\s*(\d+)", re.IGNORECASE
    )
    # The training engine prints the current epoch in key=value form
    # (e.g. "[train] epoch=5 batch=12/100"); capture it as a fallback.
    _RE_EPOCH_KV = re.compile(r"\bepoch\s*=\s*(\d+)", re.IGNORECASE)
    _RE_BATCH = re.compile(
        r"(?:batch|step)\s*[=:]?\s*(?:\[\s*)?(\d+)\s*/\s*(\d+)", re.IGNORECASE
    )
    _RE_TRAIN_LOSSES = re.compile(
        r"\[train\].*?opt[=:]?\s*([\d.eE+-]+).*?ref[=:]?\s*([\d.eE+-]+)",
        re.IGNORECASE,
    )
    _RE_LOSS_OPT = re.compile(r"loss_opt[=:]\s*([\d.eE+-]+)", re.IGNORECASE)
    _RE_LOSS_REF = re.compile(r"(?:loss_)?ref[=:]\s*([\d.eE+-]+)", re.IGNORECASE)
    _RE_LOSS_U = re.compile(r"(?:U|loss_u)[=:]\s*([\d.eE+-]+)", re.IGNORECASE)
    _RE_LOSS_A = re.compile(
        r"(?:\ba\b|loss_a|accel)[=:]\s*([\d.eE+-]+)", re.IGNORECASE
    )
    _RE_DIR_LOSS = re.compile(r"dir[=:]\s*([\d.eE+-]+)", re.IGNORECASE)
    _RE_COS_SIM = re.compile(r"cos(?:sim|_sim)?[=:]\s*([\d.eE+-]+)", re.IGNORECASE)
    _RE_ANGULAR = re.compile(r"ang[=:]\s*([\d.eE+-]+)\s*deg", re.IGNORECASE)
    _RE_LR = re.compile(r"lr[=:]\s*([\d.eE+-]+)", re.IGNORECASE)
    _RE_SAMPLES = re.compile(r"([\d.,]+)\s*(?:pts|samples)/s", re.IGNORECASE)
    _RE_ETA = re.compile(r"eta[=:]\s*([\d.]+)\s*s", re.IGNORECASE)
    _RE_MEMORY = re.compile(r"(cuda_mem=[\d/]+\s*MiB)", re.IGNORECASE)
    _RE_SCORE = re.compile(r"score[=:]\s*([\d.eE+-]+)", re.IGNORECASE)
    _RE_BEST = re.compile(r"best[=:]\s*(YES|no)", re.IGNORECASE)
    _RE_BEST_SCORE = re.compile(
        r"best.*?score[=:]\s*([\d.eE+-]+)", re.IGNORECASE
    )
    _RE_CHECKPOINT_BEST = re.compile(
        r"\[checkpoint\].*best\s+updated", re.IGNORECASE
    )
    _RE_CHECKPOINT_LAST = re.compile(
        r"\[checkpoint\].*(?:last\s+saved|ckpt_last)", re.IGNORECASE
    )
    _RE_CHECKPOINT_TRACK = re.compile(
        r"\[checkpoint\].*tracking\s+starts?\s+at\s+epoch\s+(\d+)", re.IGNORECASE
    )
    # Warnings: bracketed level tags, the standard "warning:" prefix, and the
    # common Python *Warning class names. Deliberately avoids a bare "warn".
    _RE_WARNING = re.compile(
        r"\[(?:WARNING|WARN|UYARI)\]|warning:|\bUserWarning\b|\w*Warning\b|Deprecat",
        re.IGNORECASE,
    )
    # Errors: bracketed level tags ([ERROR], [HATA], [FATAL ERROR], [CRITICAL]),
    # Python failure markers, and numeric non-finite values matched with word
    # boundaries so the substring "Inf" inside "[INFO]" is NOT treated as an error.
    _RE_ERROR = re.compile(
        r"\[(?:ERROR|HATA|CRITICAL|FATAL)\b[^\]]*\]"
        r"|\bTraceback\b|\bException\b|\bFailed\b"
        r"|\b(?:NaN|Inf(?:inity)?)\b",
        re.IGNORECASE,
    )
    _RE_VAL_HEADER = re.compile(r"\[val\s*\]", re.IGNORECASE)
    _RE_TRAIN_HEADER = re.compile(r"\[train\s*\]", re.IGNORECASE)

    def __init__(self) -> None:
        self._current_epoch: int = 0
        self._total_epochs: int = 0

    def parse_line(self, line: str) -> Optional[TrainingRecord]:
        """Parse a single log line into a TrainingRecord, or None if not parseable."""
        line = line.strip()
        if not line:
            return None

        rec = TrainingRecord()
        rec.timestamp = datetime.now().strftime("%H:%M:%S")

        # ── Epoch detection ───────────────────────────────────────
        m_ep = self._RE_EPOCH.search(line)
        if m_ep:
            self._current_epoch = int(m_ep.group(1))
            self._total_epochs = int(m_ep.group(2))
        else:
            m_ep_kv = self._RE_EPOCH_KV.search(line)
            if m_ep_kv:
                self._current_epoch = int(m_ep_kv.group(1))
        rec.epoch = self._current_epoch

        # ── Error / warning detection (highest priority) ──────────
        if self._RE_ERROR.search(line):
            rec.severity = "error"
            rec.event = "error"
            rec.phase = "system"
            rec.message = line
            return rec

        if self._RE_WARNING.search(line):
            rec.severity = "warning"
            rec.event = "warning"
            rec.phase = "system"
            rec.message = line
            return rec

        # ── Checkpoint events ─────────────────────────────────────
        if self._RE_CHECKPOINT_BEST.search(line):
            rec.event = "best_updated"
            rec.severity = "success"
            rec.phase = "checkpoint"
            rec.message = line
            m_s = self._RE_SCORE.search(line)
            if m_s:
                rec.score = _safe_float(m_s.group(1))
            return rec

        if self._RE_CHECKPOINT_LAST.search(line):
            rec.event = "checkpoint_saved"
            rec.severity = "info"
            rec.phase = "checkpoint"
            rec.message = line
            return rec

        m_track = self._RE_CHECKPOINT_TRACK.search(line)
        if m_track:
            rec.event = "info"
            rec.phase = "checkpoint"
            rec.message = line
            return rec

        # ── Phase detection ───────────────────────────────────────
        is_val = bool(self._RE_VAL_HEADER.search(line))
        is_train = bool(self._RE_TRAIN_HEADER.search(line))

        # ── Batch progress ────────────────────────────────────────
        m_batch = self._RE_BATCH.search(line)
        if m_batch:
            rec.batch = int(m_batch.group(1))
            rec.total_batches = int(m_batch.group(2))
            if rec.total_batches > 0 and rec.batch > 0:
                rec.progress_pct = min(100.0, 100.0 * rec.batch / rec.total_batches)

        # ── Loss extraction ───────────────────────────────────────
        m_train_losses = self._RE_TRAIN_LOSSES.search(line)
        if m_train_losses:
            rec.loss_opt = _safe_float(m_train_losses.group(1))
            rec.loss_ref = _safe_float(m_train_losses.group(2))
        else:
            m_lo = self._RE_LOSS_OPT.search(line)
            if m_lo:
                rec.loss_opt = _safe_float(m_lo.group(1))
            m_lr = self._RE_LOSS_REF.search(line)
            if m_lr:
                rec.loss_ref = _safe_float(m_lr.group(1))

        m_u = self._RE_LOSS_U.search(line)
        if m_u:
            rec.loss_u = _safe_float(m_u.group(1))
        m_a = self._RE_LOSS_A.search(line)
        if m_a:
            rec.loss_a = _safe_float(m_a.group(1))
        m_d = self._RE_DIR_LOSS.search(line)
        if m_d:
            rec.direction_loss = _safe_float(m_d.group(1))
        m_cs = self._RE_COS_SIM.search(line)
        if m_cs:
            rec.cos_sim = _safe_float(m_cs.group(1))
        m_ang = self._RE_ANGULAR.search(line)
        if m_ang:
            rec.angular_deg = _safe_float(m_ang.group(1))

        # ── LR, throughput, ETA ───────────────────────────────────
        m_lr2 = self._RE_LR.search(line)
        if m_lr2:
            rec.lr = _safe_float(m_lr2.group(1))
        m_sp = self._RE_SAMPLES.search(line)
        if m_sp:
            rec.samples_per_s = _safe_float(m_sp.group(1).replace(",", ""))
        m_eta = self._RE_ETA.search(line)
        if m_eta:
            rec.eta_s = _safe_float(m_eta.group(1))
        m_mem = self._RE_MEMORY.search(line)
        if m_mem:
            rec.memory = m_mem.group(1).strip()

        # ── Score ─────────────────────────────────────────────────
        m_sc = self._RE_SCORE.search(line)
        if m_sc:
            rec.score = _safe_float(m_sc.group(1))

        # ── Classify ──────────────────────────────────────────────
        has_metrics = any([
            rec.loss_opt, rec.loss_ref, rec.loss_u, rec.loss_a,
            rec.direction_loss, rec.cos_sim, rec.lr,
        ])

        if is_val and has_metrics:
            rec.phase = "val"
            rec.event = "val_summary"
        elif is_train and has_metrics:
            rec.phase = "train"
            rec.event = "batch"
        elif has_metrics:
            rec.phase = "train"
            rec.event = "batch"
        else:
            # Non-metric line — skip for structured table
            return None

        rec.message = line
        return rec


def _safe_float(s: str) -> Optional[float]:
    """Convert string to float, returning None on failure."""
    try:
        v = float(s)
        if math.isfinite(v):
            return v
        return None
    except (ValueError, TypeError):
        return None


# ── Phase 8: Metrics Store ────────────────────────────────────────────────

class TrainingMetricsStore:
    """Accumulates parsed training records and provides latest metrics.

    Used by KPI cards and the structured progress table to display
    the current training state without depending on raw log text.
    """

    def __init__(self, max_records: int = 5000) -> None:
        self._records: List[TrainingRecord] = []
        self._max_records = max_records

        # Latest values cache for fast KPI lookups
        self._latest: Dict[str, Any] = {}

    @property
    def records(self) -> List[TrainingRecord]:
        return self._records

    def append(self, record: TrainingRecord) -> int:
        """Append a record and return its index."""
        idx = len(self._records)
        self._records.append(record)

        # Trim old records if over capacity
        if len(self._records) > self._max_records:
            excess = len(self._records) - self._max_records
            self._records = self._records[excess:]
            idx -= excess

        # Update latest cache
        self._update_latest(record)
        return max(0, idx)

    def clear(self) -> None:
        self._records.clear()
        self._latest.clear()

    def latest(self, key: str, default: Any = None) -> Any:
        return self._latest.get(key, default)

    def latest_train_loss(self) -> Optional[float]:
        return self._latest.get("train_loss_opt")

    def latest_val_loss(self) -> Optional[float]:
        return self._latest.get("val_loss_ref")

    def latest_lr(self) -> Optional[float]:
        return self._latest.get("lr")

    def latest_epoch(self) -> int:
        return self._latest.get("epoch", 0)

    def latest_best_score(self) -> Optional[float]:
        return self._latest.get("best_score")

    def latest_best_epoch(self) -> Optional[int]:
        return self._latest.get("best_epoch")

    def _update_latest(self, rec: TrainingRecord) -> None:
        if rec.epoch > 0:
            self._latest["epoch"] = rec.epoch

        if rec.phase == "train":
            if rec.loss_opt is not None:
                self._latest["train_loss_opt"] = rec.loss_opt
            if rec.loss_ref is not None:
                self._latest["train_loss_ref"] = rec.loss_ref
            if rec.loss_u is not None:
                self._latest["train_loss_u"] = rec.loss_u
            if rec.loss_a is not None:
                self._latest["train_loss_a"] = rec.loss_a
            if rec.cos_sim is not None:
                self._latest["train_cos_sim"] = rec.cos_sim
        elif rec.phase == "val":
            if rec.loss_ref is not None:
                self._latest["val_loss_ref"] = rec.loss_ref
            if rec.loss_opt is not None:
                self._latest["val_loss_opt"] = rec.loss_opt
            if rec.loss_u is not None:
                self._latest["val_loss_u"] = rec.loss_u
            if rec.loss_a is not None:
                self._latest["val_loss_a"] = rec.loss_a
            if rec.cos_sim is not None:
                self._latest["val_cos_sim"] = rec.cos_sim
            if rec.angular_deg is not None:
                self._latest["val_angular_deg"] = rec.angular_deg

        if rec.lr is not None:
            self._latest["lr"] = rec.lr
        if rec.score is not None:
            self._latest["score"] = rec.score
        if rec.samples_per_s is not None:
            self._latest["samples_per_s"] = rec.samples_per_s

        if rec.event == "best_updated":
            if rec.score is not None:
                self._latest["best_score"] = rec.score
            self._latest["best_epoch"] = rec.epoch
