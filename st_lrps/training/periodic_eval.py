# -*- coding: utf-8 -*-
"""
Periodic Evaluation During Training (monitoring only).

This module implements an OPTIONAL feature: at selected epochs, after the normal
epoch train/validation phase has finished and ``ckpt_last.pt`` has been safely
saved, the existing evaluation CLI (``st_lrps.evaluation.cli``) is invoked as a
SUBPROCESS on the current checkpoint. The purpose is purely observability — to
watch field-level diagnostics (parity, acceleration/potential/angular metrics,
altitude-binned diagnostics) evolve during training, so a run that looks fine on
the scalar train/val losses but is secretly learning poorly can be spotted.

Hard guarantees (by construction):
  * Evaluation runs in a separate process. It cannot touch the live optimizer,
    scheduler, GradNorm, gradients, RNG, model weights, or checkpoint selection.
  * It never runs before ``ckpt_last.pt`` is saved for the epoch.
  * A failed periodic evaluation does NOT abort training unless the user
    explicitly disables ``periodic_eval_continue_on_fail``.
  * With both schedule knobs disabled (the default) nothing here executes.

The schedule helper and the command builder are intentionally pure and testable;
the subprocess runner is the only side-effecting part.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Repository root: st_lrps/training/periodic_eval.py -> parents[2] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

EVAL_CLI_MODULE = "st_lrps.evaluation.cli"

_VALID_DATASETS = ("val", "test", "ood")
_VALID_PREFER = ("last", "best")


# ---------------------------------------------------------------------------
# Pure scheduling
# ---------------------------------------------------------------------------
def compute_periodic_eval_epochs(
    total_epochs: int,
    count: Optional[int],
    every_epochs: Optional[int],
    start_epoch: int = 1,
) -> List[int]:
    """Return the sorted, de-duplicated list of (1-based) periodic-eval epochs.

    Exactly one scheduling mode may be active:
      * ``count``        : run ``count`` evenly-spaced evaluations across the full
                           training horizon, e.g. count=10, total=400 ->
                           [40,80,...,400]. The final epoch is naturally included.
      * ``every_epochs`` : run every K epochs, e.g. every=25, total=100 ->
                           [25,50,75,100].

    Rules:
      * ``count`` and ``every_epochs`` are mutually exclusive (ValueError if both).
      * Neither set, or a non-positive value, yields an empty schedule.
      * Epochs strictly below ``start_epoch`` are dropped (used on resume so
        already-passed epochs are not scheduled). This filters the schedule; it
        does not change the underlying even spacing.
    """
    if count is not None and every_epochs is not None:
        raise ValueError(
            "periodic_eval_count and periodic_eval_every_epochs are mutually "
            "exclusive; set at most one."
        )

    total = int(total_epochs)
    if total <= 0:
        return []

    epochs: set[int] = set()
    if count is not None:
        c = int(count)
        if c <= 0:
            return []
        # Cannot evaluate more times than there are epochs.
        c = min(c, total)
        for i in range(1, c + 1):
            e = int(round(total * i / c))
            epochs.add(max(1, min(total, e)))
    elif every_epochs is not None:
        k = int(every_epochs)
        if k <= 0:
            return []
        e = k
        while e <= total:
            epochs.add(e)
            e += k
    else:
        return []

    start = int(start_epoch)
    return sorted(e for e in epochs if e >= start)


# ---------------------------------------------------------------------------
# Plan resolution
# ---------------------------------------------------------------------------
@dataclass
class PeriodicEvalPlan:
    """Resolved periodic-evaluation plan for a training run."""

    enabled: bool
    epochs: List[int] = field(default_factory=list)
    dataset: str = "val"
    prefer_checkpoint: str = "last"
    max_samples: int = 200_000
    batch_size: int = 8192
    device: str = "auto"
    timeout_sec: Optional[int] = None
    continue_on_fail: bool = True

    @property
    def epochs_set(self) -> set[int]:
        return set(self.epochs)


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    return getattr(cfg, name, default)


def resolve_periodic_eval_plan(cfg: Any, *, start_epoch: int = 1) -> PeriodicEvalPlan:
    """Build a :class:`PeriodicEvalPlan` from a ``TrainConfig``-like object.

    ``start_epoch`` is the first 1-based epoch that will actually run (so on a
    resume from epoch 287, pass 288 to drop already-passed scheduled epochs).
    With both schedule knobs disabled the returned plan has ``enabled=False`` and
    an empty epoch list.
    """
    count = _cfg_get(cfg, "periodic_eval_count", None)
    every = _cfg_get(cfg, "periodic_eval_every_epochs", None)
    count = int(count) if count not in (None, 0) else None
    every = int(every) if every not in (None, 0) else None

    total_epochs = int(_cfg_get(cfg, "epochs", 0) or 0)
    epochs = compute_periodic_eval_epochs(total_epochs, count, every, start_epoch=start_epoch)

    dataset = str(_cfg_get(cfg, "periodic_eval_dataset", "val") or "val").strip().lower()
    if dataset not in _VALID_DATASETS:
        dataset = "val"
    prefer = str(_cfg_get(cfg, "periodic_eval_prefer_checkpoint", "last") or "last").strip().lower()
    if prefer not in _VALID_PREFER:
        prefer = "last"

    # Batch size: fall back to the training batch size when not explicitly set.
    pe_bs = _cfg_get(cfg, "periodic_eval_batch_size", None)
    if pe_bs in (None, 0):
        pe_bs = int(_cfg_get(cfg, "batch_size", 8192) or 8192)
    else:
        pe_bs = int(pe_bs)

    timeout = _cfg_get(cfg, "periodic_eval_timeout_sec", None)
    timeout = int(timeout) if timeout not in (None, 0) else None

    return PeriodicEvalPlan(
        enabled=bool(epochs),
        epochs=epochs,
        dataset=dataset,
        prefer_checkpoint=prefer,
        max_samples=int(_cfg_get(cfg, "periodic_eval_max_samples", 200_000) or 200_000),
        batch_size=pe_bs,
        device=str(_cfg_get(cfg, "periodic_eval_device", "auto") or "auto").strip().lower(),
        timeout_sec=timeout,
        continue_on_fail=bool(_cfg_get(cfg, "periodic_eval_continue_on_fail", True)),
    )


def resolve_eval_dataset_path(cfg: Any, dataset: str) -> Optional[str]:
    """Resolve the dataset path for periodic evaluation.

    ``val``  -> ``cfg.val_data`` (falls back to ``cfg.data`` for single-dataset
                runs that keep validation as an internal split),
    ``test`` -> ``cfg.test_data``,
    ``ood``  -> ``cfg.ood_data``.

    Returns ``None`` when the selected dataset is not configured; the caller then
    logs a warning and skips that periodic evaluation rather than crashing.
    """
    dataset = str(dataset).strip().lower()
    if dataset == "val":
        return _cfg_get(cfg, "val_data", None) or _cfg_get(cfg, "data", None)
    if dataset == "test":
        return _cfg_get(cfg, "test_data", None)
    if dataset == "ood":
        return _cfg_get(cfg, "ood_data", None)
    return None


# ---------------------------------------------------------------------------
# Pure command builder
# ---------------------------------------------------------------------------
def build_periodic_eval_command(
    *,
    run_dir: str | Path,
    data_path: str | Path,
    out_dir: str | Path,
    prefer_checkpoint: str = "last",
    max_samples: int = 200_000,
    batch_size: int = 8192,
    device: str = "auto",
    dataset_name: str = "data",
    python_exe: Optional[str] = None,
) -> List[str]:
    """Build the argv list that invokes the evaluation CLI for one epoch.

    Uses only flags the evaluation CLI actually supports (``--model-dir``,
    ``--data``, ``--out``, ``--checkpoint-prefer``, ``--max-samples``,
    ``--batch-size``, ``--device``, ``--dataset-name``).
    """
    exe = python_exe or sys.executable
    cmd = [
        exe,
        "-u",
        "-m",
        EVAL_CLI_MODULE,
        "--model-dir", str(run_dir),
        "--data", str(data_path),
        "--out", str(out_dir),
        "--checkpoint-prefer", str(prefer_checkpoint),
        "--max-samples", str(int(max_samples)),
        "--batch-size", str(int(batch_size)),
        "--device", str(device),
    ]
    if dataset_name and str(dataset_name) != "data":
        cmd += ["--dataset-name", str(dataset_name)]
    return cmd


# ---------------------------------------------------------------------------
# History (resume support)
# ---------------------------------------------------------------------------
def periodic_evals_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "periodic_evals"


def history_path(run_dir: str | Path) -> Path:
    return periodic_evals_dir(run_dir) / "periodic_eval_history.jsonl"


def epoch_output_dir(run_dir: str | Path, epoch: int) -> Path:
    return periodic_evals_dir(run_dir) / f"epoch_{int(epoch):04d}"


def load_periodic_eval_history(run_dir: str | Path) -> Dict[int, str]:
    """Return ``{epoch: last_status}`` from the history jsonl (empty if absent)."""
    path = history_path(run_dir)
    result: Dict[int, str] = {}
    if not path.is_file():
        return result
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                ep = rec.get("epoch")
                if ep is None:
                    continue
                try:
                    result[int(ep)] = str(rec.get("status", "")).lower()
                except (TypeError, ValueError):
                    continue
    except OSError:
        return result
    return result


def completed_periodic_eval_epochs(run_dir: str | Path) -> set[int]:
    """Epochs whose most recent periodic-eval record is success or skipped.

    These are not re-run on resume. A prior *failure* on a past epoch does not
    block training and is not auto-rerun (the schedule simply moves on).
    """
    return {
        ep for ep, status in load_periodic_eval_history(run_dir).items()
        if status in ("success", "skipped")
    }


def _append_history_record(run_dir: str | Path, record: Dict[str, Any]) -> None:
    path = history_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


# ---------------------------------------------------------------------------
# Metrics extraction (best-effort, for the log line)
# ---------------------------------------------------------------------------
def _extract_summary_metrics(out_dir: Path) -> tuple[Optional[str], Dict[str, Any]]:
    """Best-effort read of headline metrics from the eval outputs.

    Returns ``(summary_path, metrics)``. Tries the flat ``summary_metrics.json``
    first (written when --out is set), then falls back to ``eval_report.json``.
    Never raises.
    """
    summary_json = out_dir / "summary_metrics.json"
    if summary_json.is_file():
        try:
            rows = json.loads(summary_json.read_text(encoding="utf-8"))
            if isinstance(rows, list) and rows:
                row = rows[0]
            elif isinstance(rows, dict):
                row = rows
            else:
                row = {}
            return str(summary_json), {
                "rmse_u": row.get("rmse_u"),
                "rmse_a": row.get("rmse_a_vec"),
                "mae_a": row.get("mae_a_vec"),
                "mean_ang_deg": row.get("angular_mean_deg"),
                "n_samples": row.get("n_samples"),
            }
        except Exception:
            pass

    report_json = out_dir / "eval_report.json"
    if report_json.is_file():
        try:
            report = json.loads(report_json.read_text(encoding="utf-8"))
            metrics = report.get("metrics") or {}
            u = metrics.get("U") or {}
            avec = metrics.get("residual_vector_metrics") or metrics.get("|a|") or {}
            ang = (metrics.get("angular_metrics") or {}).get("residual_all") or {}
            return str(report_json), {
                "rmse_u": u.get("rmse"),
                "rmse_a": avec.get("rmse"),
                "mae_a": avec.get("mae"),
                "mean_ang_deg": ang.get("mean_deg"),
                "mean_cos_sim": ang.get("mean_cos_sim"),
            }
        except Exception:
            pass
    return None, {}


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.3e}"
    except (TypeError, ValueError):
        return "n/a"


# ---------------------------------------------------------------------------
# Side-effecting runner
# ---------------------------------------------------------------------------
def run_periodic_eval(
    cfg: Any,
    run_dir: str | Path,
    epoch: int,
    plan: PeriodicEvalPlan,
    *,
    log: Optional[logging.Logger] = None,
    dataset_name: str = "data",
    python_exe: Optional[str] = None,
    _runner=None,
) -> bool:
    """Run one periodic evaluation for a 1-based ``epoch``. Returns success bool.

    A missing dataset is treated as a skip (returns ``True`` — not a failure).
    The subprocess stdout/stderr are captured to log files under the epoch dir.
    A history record is always written. This function never raises for an
    evaluation failure; it logs, records, and returns ``False`` so the caller can
    decide whether to continue based on ``plan.continue_on_fail``.

    ``_runner`` is an injection point for tests (defaults to ``subprocess.run``).
    """
    lg = log or logger
    run_dir = Path(run_dir)
    out_dir = epoch_output_dir(run_dir, epoch)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.perf_counter()

    data_path = resolve_eval_dataset_path(cfg, plan.dataset)
    if not data_path or not Path(str(data_path)).exists():
        lg.warning(
            f"[periodic-eval] epoch={epoch} skipped: dataset={plan.dataset!r} path "
            f"{'missing' if not data_path else 'not found: ' + str(data_path)}."
        )
        _append_history_record(run_dir, {
            "epoch": int(epoch),
            "status": "skipped",
            "dataset": plan.dataset,
            "started_at": started_at,
            "finished_at": started_at,
            "command": None,
            "output_dir": str(out_dir),
            "metrics_summary_path": None,
            "error": f"dataset path missing for {plan.dataset!r}",
        })
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_periodic_eval_command(
        run_dir=run_dir,
        data_path=data_path,
        out_dir=out_dir,
        prefer_checkpoint=plan.prefer_checkpoint,
        max_samples=plan.max_samples,
        batch_size=plan.batch_size,
        device=plan.device,
        dataset_name=dataset_name,
        python_exe=python_exe,
    )

    lg.info(
        f"[periodic-eval] epoch={epoch} starting dataset={plan.dataset} "
        f"checkpoint={plan.prefer_checkpoint} out={out_dir}"
    )

    runner = _runner or subprocess.run
    stdout_path = out_dir / "eval_stdout.log"
    stderr_path = out_dir / "eval_stderr.log"
    status = "failure"
    error_msg: Optional[str] = None
    rc: Optional[int] = None

    try:
        completed = runner(
            cmd,
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=plan.timeout_sec,
        )
        rc = int(getattr(completed, "returncode", 1))
        try:
            stdout_path.write_text(getattr(completed, "stdout", "") or "", encoding="utf-8")
            stderr_path.write_text(getattr(completed, "stderr", "") or "", encoding="utf-8")
        except OSError:
            pass
        if rc == 0:
            status = "success"
        else:
            tail = (getattr(completed, "stderr", "") or "").strip().splitlines()[-3:]
            error_msg = f"exit code {rc}: " + " | ".join(tail) if tail else f"exit code {rc}"
    except subprocess.TimeoutExpired:
        status = "failure"
        error_msg = f"timeout after {plan.timeout_sec}s"
    except Exception as exc:  # pragma: no cover - defensive
        status = "failure"
        error_msg = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - t0
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary_path, metrics = _extract_summary_metrics(out_dir) if status == "success" else (None, {})

    _append_history_record(run_dir, {
        "epoch": int(epoch),
        "status": status,
        "dataset": plan.dataset,
        "checkpoint_prefer": plan.prefer_checkpoint,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_sec": round(elapsed, 3),
        "command": cmd,
        "output_dir": str(out_dir),
        "metrics_summary_path": summary_path,
        "metrics": metrics or None,
        "error": error_msg,
    })

    if status == "success":
        lg.info(f"[periodic-eval] epoch={epoch} done in {elapsed:.1f}s summary={summary_path}")
        lg.info(
            f"[periodic-eval] epoch={epoch} rmse_u={_fmt(metrics.get('rmse_u'))} "
            f"rmse_a={_fmt(metrics.get('rmse_a'))} "
            f"mean_ang_deg={_fmt(metrics.get('mean_ang_deg'))}"
        )
        return True

    lg.warning(f"[periodic-eval] epoch={epoch} failed: {error_msg}")
    if plan.continue_on_fail:
        lg.warning("[periodic-eval] continuing training because continue_on_fail=True")
    return False


__all__ = [
    "PeriodicEvalPlan",
    "compute_periodic_eval_epochs",
    "resolve_periodic_eval_plan",
    "resolve_eval_dataset_path",
    "build_periodic_eval_command",
    "periodic_evals_dir",
    "history_path",
    "epoch_output_dir",
    "load_periodic_eval_history",
    "completed_periodic_eval_epochs",
    "run_periodic_eval",
]
