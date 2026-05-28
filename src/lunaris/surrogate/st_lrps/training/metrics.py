# -*- coding: utf-8 -*-
"""Shared metric naming and checkpoint-score helpers for ST-LRPS.

This module is deliberately small: it centralizes the scalar score used for
``ckpt_best.pt`` selection and the flattened history schema consumed by logs,
CSV/JSONL files, ablation aggregation, and the UI.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

SUPPORTED_BEST_METRICS = ("val_total_loss", "val_base_loss", "hybrid", "direction_loss")
LOWER_IS_BETTER = True

_WARNED_ALIASES: set[str] = set()
_WARNED_FALLBACKS: set[str] = set()


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _warn_once(kind: str, message: str) -> None:
    if kind in _WARNED_FALLBACKS:
        return
    _WARNED_FALLBACKS.add(kind)
    logger.warning(message)


def normalize_best_metric(best_metric: Any) -> str:
    """Return a canonical best-metric name, preserving the old alias."""
    metric = str(best_metric or "hybrid").strip().lower()
    if metric == "total_loss":
        if metric not in _WARNED_ALIASES:
            _WARNED_ALIASES.add(metric)
            logger.warning(
                "best_metric='total_loss' is deprecated; treating it as "
                "best_metric='val_total_loss'."
            )
        return "val_total_loss"
    if metric not in SUPPORTED_BEST_METRICS:
        raise ValueError(
            f"Unsupported best_metric={metric!r}. Supported values: "
            + ", ".join(SUPPORTED_BEST_METRICS)
            + " (plus deprecated alias 'total_loss')."
        )
    return metric


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise KeyError(f"Required metric {name!r} is missing or non-numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Required metric {name!r} is not finite: {result!r}")
    return result


def _metric(
    stats: Mapping[str, Any],
    key: str,
    *,
    required: bool = False,
    fallback_keys: Iterable[str] = (),
    fallback_value: Any = None,
) -> Optional[float]:
    if key in stats and stats.get(key) is not None:
        return _finite_float(stats.get(key), name=key)
    for fb in fallback_keys:
        if fb in stats and stats.get(fb) is not None:
            _warn_once(
                f"{key}->{fb}",
                f"Metric {key!r} missing; using documented fallback {fb!r}.",
            )
            return _finite_float(stats.get(fb), name=fb)
    if fallback_value is not None:
        return _finite_float(fallback_value, name=key)
    if required:
        raise KeyError(f"Required metric {key!r} is missing from validation stats.")
    return None


def _val_total_loss(stats: Mapping[str, Any]) -> float:
    return _metric(stats, "val_total_loss", required=True, fallback_keys=("loss", "loss_ref"))  # type: ignore[return-value]


def _val_loss_u(stats: Mapping[str, Any]) -> float:
    return _metric(stats, "val_loss_u", required=True, fallback_keys=("mse_u",))  # type: ignore[return-value]


def _val_loss_a(stats: Mapping[str, Any]) -> float:
    return _metric(stats, "val_loss_a", required=True, fallback_keys=("mse_a",))  # type: ignore[return-value]


def _val_base_loss(stats: Mapping[str, Any]) -> float:
    value = _metric(stats, "val_base_loss", fallback_keys=("loss_base",))
    if value is not None:
        return value
    _warn_once(
        "val_base_loss->mse_u+mse_a",
        "Metric 'val_base_loss' missing; using documented fallback mse_u + mse_a.",
    )
    return _val_loss_u(stats) + _val_loss_a(stats)


def _val_loss_dir(stats: Mapping[str, Any]) -> float:
    return _metric(stats, "val_loss_dir", fallback_keys=("loss_dir",), fallback_value=0.0) or 0.0


def _val_physics_loss(stats: Mapping[str, Any], *, base: float, total: float) -> float:
    value = _metric(stats, "val_physics_loss", fallback_keys=("loss_physics",))
    if value is not None:
        return value
    _warn_once(
        "val_physics_loss->total-base",
        "Metric 'val_physics_loss' missing; using documented fallback val_total_loss - val_base_loss.",
    )
    return float(total - base)


def checkpoint_formula(best_metric: Any, cfg: Any = None) -> str:
    """Human-readable formula for the canonical checkpoint score."""
    metric = normalize_best_metric(best_metric)
    alpha = float(_cfg_get(cfg, "hybrid_direction_alpha", 0.30))
    if metric == "val_total_loss":
        return "val_total_loss"
    if metric == "val_base_loss":
        return "val_base_loss"
    if metric == "direction_loss":
        return "val_loss_dir"
    return f"val_base_loss + {alpha:.2f} * val_loss_dir"


def checkpoint_selection_block(
    cfg: Any,
    *,
    start_epoch: Optional[int] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Serializable checkpoint-selection description for config/manifest/logs."""
    metric = normalize_best_metric(_cfg_get(cfg, "best_metric", "hybrid"))
    alpha = float(_cfg_get(cfg, "hybrid_direction_alpha", 0.30))
    block = {
        "best_metric": metric,
        "formula": checkpoint_formula(metric, cfg),
        "lower_is_better": LOWER_IS_BETTER,
        "start_epoch": (int(start_epoch) if start_epoch is not None else None),
        "start_epoch_display": (int(start_epoch) + 1 if start_epoch is not None else None),
        "hybrid_direction_alpha": alpha,
    }
    if reason:
        block["reason"] = str(reason)
    return block


def compute_checkpoint_score(val_stats: Mapping[str, Any], cfg: Any) -> Tuple[float, Dict[str, Any]]:
    """Compute the scalar score used for best-checkpoint selection.

    Supported canonical ``best_metric`` values are ``val_total_loss``,
    ``val_base_loss``, ``hybrid``, and ``direction_loss``. The historical
    ``total_loss`` value is accepted as an alias for ``val_total_loss`` with a
    warning.
    """
    metric = normalize_best_metric(_cfg_get(cfg, "best_metric", "hybrid"))
    alpha = float(_cfg_get(cfg, "hybrid_direction_alpha", 0.30))
    total = _val_total_loss(val_stats)
    loss_u = _val_loss_u(val_stats)
    loss_a = _val_loss_a(val_stats)
    base = _val_base_loss(val_stats)
    physics = _val_physics_loss(val_stats, base=base, total=total)
    loss_dir = _val_loss_dir(val_stats)

    if metric == "val_total_loss":
        score = total
    elif metric == "val_base_loss":
        score = base
    elif metric == "direction_loss":
        score = loss_dir
    elif metric == "hybrid":
        score = base + alpha * loss_dir
    else:  # pragma: no cover - normalize_best_metric guards this.
        raise ValueError(metric)

    report = {
        "best_metric": metric,
        "formula": checkpoint_formula(metric, cfg),
        "score": float(score),
        "val_total_loss": float(total),
        "val_base_loss": float(base),
        "val_physics_loss": float(physics),
        "val_loss_u": float(loss_u),
        "val_loss_a": float(loss_a),
        "val_loss_dir": float(loss_dir),
        "hybrid_direction_alpha": float(alpha),
        "eligible_for_best": bool(val_stats.get("eligible_for_best", True)),
        "lower_is_better": LOWER_IS_BETTER,
    }
    return float(score), report


def _optional_float(stats: Mapping[str, Any], *keys: str, default: Optional[float] = None) -> Optional[float]:
    for key in keys:
        value = stats.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
    return default


def flatten_epoch_metrics(
    epoch: int,
    train_stats: Mapping[str, Any],
    val_stats: Mapping[str, Any],
    checkpoint_report: Mapping[str, Any],
    cfg: Any,
) -> Dict[str, Any]:
    """Create a flat, stable row for history CSV/JSONL and UI consumers."""
    train_base = _optional_float(
        train_stats,
        "train_base_loss",
        "train_loss_base",
        default=(
            (_optional_float(train_stats, "mse_u", "train_loss_u", default=0.0) or 0.0)
            + (_optional_float(train_stats, "mse_a", "train_loss_a", default=0.0) or 0.0)
        ),
    )
    train_total = _optional_float(train_stats, "loss", "loss_ref", "train_loss_total", default=None)
    val_total = _optional_float(val_stats, "val_total_loss", "loss", "loss_ref", "val_loss_total", default=None)
    val_base = _optional_float(
        val_stats,
        "val_base_loss",
        "val_loss_base",
        default=(
            (_optional_float(val_stats, "mse_u", "val_loss_u", default=0.0) or 0.0)
            + (_optional_float(val_stats, "mse_a", "val_loss_a", default=0.0) or 0.0)
        ),
    )

    row: Dict[str, Any] = {
        "epoch": int(epoch),
        "epoch_display": int(epoch) + 1,
        "train_loss_total": train_total,
        "train_loss_objective": _optional_float(train_stats, "objective_loss", "loss_opt", default=train_total),
        "train_loss_base": train_base,
        "train_loss_physics": _optional_float(train_stats, "train_physics_loss", "train_loss_physics", default=0.0),
        "train_loss_u": _optional_float(train_stats, "mse_u", "train_loss_u", default=0.0),
        "train_loss_a": _optional_float(train_stats, "mse_a", "train_loss_a", default=0.0),
        "train_loss_dir": _optional_float(train_stats, "loss_dir", "train_loss_dir", default=0.0),
        "train_cos_sim": _optional_float(train_stats, "cossim_mean", "train_cos_sim", "train_mean_cossim", default=None),
        "train_angular_mean_deg": _optional_float(train_stats, "angular_mean_deg", "train_angular_mean_deg", default=None),
        "train_radial_loss": _optional_float(train_stats, "loss_radial", "train_radial_loss", "train_loss_radial", default=0.0),
        "train_cross_loss": _optional_float(train_stats, "loss_cross", "train_cross_loss", "train_loss_cross", default=0.0),
        "train_laplacian_diag": _optional_float(train_stats, "loss_laplacian_diag", "train_laplacian_diag", default=0.0),
        "train_laplacian_train": _optional_float(train_stats, "loss_laplacian_train", "train_laplacian_train", default=0.0),
        "val_loss_total": val_total,
        "val_loss_base": val_base,
        "val_loss_physics": _optional_float(val_stats, "val_physics_loss", "val_loss_physics", default=0.0),
        "val_loss_u": _optional_float(val_stats, "mse_u", "val_loss_u", default=0.0),
        "val_loss_a": _optional_float(val_stats, "mse_a", "val_loss_a", default=0.0),
        "val_loss_dir": _optional_float(val_stats, "loss_dir", "val_loss_dir", default=0.0),
        "val_cos_sim": _optional_float(val_stats, "cossim_mean", "val_cos_sim", "val_mean_cossim", default=None),
        "val_angular_mean_deg": _optional_float(val_stats, "angular_mean_deg", "val_angular_mean_deg", default=None),
        "val_radial_loss": _optional_float(val_stats, "loss_radial", "val_radial_loss", "val_loss_radial", default=0.0),
        "val_cross_loss": _optional_float(val_stats, "loss_cross", "val_cross_loss", "val_loss_cross", default=0.0),
        "val_laplacian_diag": _optional_float(val_stats, "loss_laplacian_diag", "val_laplacian_diag", default=0.0),
        "val_laplacian_train": _optional_float(val_stats, "loss_laplacian_train", "val_laplacian_train", default=0.0),
        "checkpoint_score": _optional_float(checkpoint_report, "score", default=None),
        "checkpoint_formula": checkpoint_report.get("formula"),
        "best_metric": checkpoint_report.get("best_metric") or normalize_best_metric(_cfg_get(cfg, "best_metric", "hybrid")),
        "is_best_eligible": bool(checkpoint_report.get("eligible_for_best", False)),
        "is_best_update": bool(checkpoint_report.get("is_best_update", False)),
        "best_epoch": checkpoint_report.get("best_epoch"),
        "best_score": checkpoint_report.get("best_score"),
        "lr": _optional_float(train_stats, "lr", default=None),
        "w_u": _optional_float(train_stats, "w_u", default=None),
        "w_a_raw": _optional_float(train_stats, "w_a_raw", default=None),
        "w_a_eff": _optional_float(train_stats, "w_a_eff", "w_a", default=None),
        "grad_norm": _optional_float(train_stats, "grad_norm", default=None),
        "samples_seen": int(train_stats.get("samples_seen", 0) or 0),
        "optimizer_steps": int(train_stats.get("optimizer_steps", 0) or 0),
        "epoch_time_s": _optional_float(train_stats, "epoch_time_s", default=None),
        "lower_is_better": bool(checkpoint_report.get("lower_is_better", LOWER_IS_BETTER)),
        "hybrid_direction_alpha": _optional_float(checkpoint_report, "hybrid_direction_alpha", default=float(_cfg_get(cfg, "hybrid_direction_alpha", 0.30))),
        "lambda_dir_eff": _optional_float(train_stats, "lambda_dir_eff", default=0.0),
        "col_lap_attempts": int(train_stats.get("collocation_laplacian_attempt_count", train_stats.get("col_lap_attempts", 0)) or 0),
        "col_lap_success": int(train_stats.get("collocation_laplacian_success_count", train_stats.get("col_lap_success", 0)) or 0),
        "col_lap_fail": int(train_stats.get("collocation_laplacian_fail_count", train_stats.get("col_lap_fail", 0)) or 0),
    }

    # Backward-compatible aliases used by older plots/UI.
    row["train_mean_cossim"] = row["train_cos_sim"]
    row["val_mean_cossim"] = row["val_cos_sim"]
    row["val_ang_deg"] = row["val_angular_mean_deg"]
    row["train_loss_radial"] = row["train_radial_loss"]
    row["train_loss_cross"] = row["train_cross_loss"]
    row["train_loss_laplacian"] = row["train_laplacian_train"]
    row["val_loss_radial"] = row["val_radial_loss"]
    row["val_loss_cross"] = row["val_cross_loss"]
    row["val_loss_laplacian"] = row["val_laplacian_train"]
    row["val_checkpoint_score"] = row["checkpoint_score"]
    return row


def _fmt(value: Any, default: str = "n/a") -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f"{f:.3e}"


def format_epoch_summary(row: Mapping[str, Any], *, total_epochs: Optional[int] = None) -> str:
    """Return one compact human-readable epoch summary line."""
    epoch_display = int(row.get("epoch_display", int(row.get("epoch", 0)) + 1))
    total = int(total_epochs) if total_epochs is not None else None
    epoch_text = f"{epoch_display:03d}/{total:03d}" if total else f"{epoch_display:03d}"
    best_text = "YES" if row.get("is_best_update") else "no "
    eligible = "eligible" if row.get("is_best_eligible") else "wait    "
    best_epoch = row.get("best_epoch")
    best_epoch_text = f"{str(best_epoch):>3}" if best_epoch not in (None, "") else "  -"
    formula = str(row.get("checkpoint_formula") or "")
    metric = str(row.get("best_metric") or "")
    score = _fmt(row.get("checkpoint_score"))
    return (
        f"Epoch {epoch_text:^7} | "
        f"train opt: {_fmt(row.get('train_loss_objective')):>9}  ref: {_fmt(row.get('train_loss_total')):>9} | "
        f"val tot: {_fmt(row.get('val_loss_total')):>9}  base: {_fmt(row.get('val_loss_base')):>9}  "
        f"dir: {_fmt(row.get('val_loss_dir')):>9} | "
        f"score: {score:>9} [{metric}: {formula}] {eligible} | "
        f"best: {best_text}  ep: {best_epoch_text}  score: {_fmt(row.get('best_score')):>9} | "
        f"lr: {_fmt(row.get('lr')):>9} | {float(row.get('epoch_time_s') or 0.0):6.1f}s"
    )


def format_batch_summary(
    *,
    phase: str,
    epoch: int,
    batch: int,
    total_batches: int,
    loss_opt: float,
    loss_ref: float,
    loss_u: float,
    loss_a: float,
    lr: float,
    eta_s: Optional[float] = None,
    samples_per_s: Optional[float] = None,
    loss_dir: Optional[float] = None,
    memory: str = "",
) -> str:
    """Return a compact one-line batch progress summary."""
    extras = []
    if loss_dir is not None:
        extras.append(f"dir: {loss_dir:9.2e}")
    if samples_per_s is not None:
        extras.append(f"{samples_per_s:7,.0f} spl/s")
    if eta_s is not None:
        extras.append(f"eta: {eta_s:5.0f}s")
    if memory:
        extras.append(memory.strip())
    suffix = " | " + " | ".join(extras) if extras else ""
    return (
        f"[{phase:^5}] ep: {epoch:3d}  b: {batch:4d}/{total_batches:<4d} | "
        f"opt: {loss_opt:9.3e} | ref: {loss_ref:9.3e} | U: {loss_u:9.2e} | a: {loss_a:9.2e} | "
        f"lr: {lr:9.2e}{suffix}"
    )


HISTORY_FIELDNAMES = [
    "epoch",
    "epoch_display",
    "train_loss_total",
    "train_loss_objective",
    "train_loss_base",
    "train_loss_physics",
    "train_loss_u",
    "train_loss_a",
    "train_loss_dir",
    "train_cos_sim",
    "train_angular_mean_deg",
    "train_radial_loss",
    "train_cross_loss",
    "train_laplacian_diag",
    "train_laplacian_train",
    "val_loss_total",
    "val_loss_base",
    "val_loss_physics",
    "val_loss_u",
    "val_loss_a",
    "val_loss_dir",
    "val_cos_sim",
    "val_angular_mean_deg",
    "val_radial_loss",
    "val_cross_loss",
    "val_laplacian_diag",
    "val_laplacian_train",
    "checkpoint_score",
    "checkpoint_formula",
    "best_metric",
    "is_best_eligible",
    "is_best_update",
    "best_epoch",
    "best_score",
    "lr",
    "w_u",
    "w_a_raw",
    "w_a_eff",
    "grad_norm",
    "samples_seen",
    "optimizer_steps",
    "epoch_time_s",
    "lower_is_better",
    "hybrid_direction_alpha",
    "lambda_dir_eff",
    "col_lap_attempts",
    "col_lap_success",
    "col_lap_fail",
    # Legacy/compatibility aliases.
    "train_mean_cossim",
    "val_mean_cossim",
    "val_ang_deg",
    "train_loss_radial",
    "train_loss_cross",
    "train_loss_laplacian",
    "val_loss_radial",
    "val_loss_cross",
    "val_loss_laplacian",
    "val_checkpoint_score",
]


__all__ = [
    "SUPPORTED_BEST_METRICS",
    "LOWER_IS_BETTER",
    "normalize_best_metric",
    "checkpoint_formula",
    "checkpoint_selection_block",
    "compute_checkpoint_score",
    "flatten_epoch_metrics",
    "format_epoch_summary",
    "format_batch_summary",
    "HISTORY_FIELDNAMES",
]
