"""Frozen-orbit search report figures (roadmap R31).

Every figure declares its analytical question (title), source data (footer),
units (axis labels), frame, and scale — the scientific-figures discipline. All
functions are Agg-safe (no display) and return the written file path, or
``None`` when the input data cannot support the figure (never a misleading
placeholder plot).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_FOOTER_KW = {"fontsize": 7, "color": "0.45", "ha": "left", "va": "bottom"}


def _mpl():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _footer(fig: Any, source: str, frame: str = "inertial (identity frame)") -> None:
    fig.text(0.01, 0.005, f"data: {source} | frame: {frame} | linear scale", **_FOOTER_KW)


def _save(fig: Any, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _finite(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def plot_element_histories(
    histories: list[dict[str, Any]],
    out_dir: Path,
    *,
    source: str,
) -> list[Path]:
    """e(t), h_peri(t), omega(t) for the top validated candidates.

    ``histories``: one dict per candidate with ``t_s``, ``e``, ``h_peri_m``,
    ``argp_rad``, and ``label``.
    """
    if not histories:
        return []
    plt = _mpl()
    written: list[Path] = []
    panels = (
        ("e", 1.0, "eccentricity [-]", "Is the eccentricity envelope bounded?", "e_vs_t.png"),
        (
            "h_peri_m",
            1e-3,
            "perilune altitude [km]",
            "Is the perilune altitude bounded above the safety floor?",
            "h_peri_vs_t.png",
        ),
        (
            "argp_rad",
            180.0 / np.pi,
            "argument of periapsis [deg]",
            "Does omega librate (bounded) or circulate?",
            "omega_vs_t.png",
        ),
    )
    for key, scale, ylabel, question, filename in panels:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for hist in histories:
            t_day = np.asarray(hist["t_s"], dtype=np.float64) / 86_400.0
            series = np.asarray(hist[key], dtype=np.float64) * scale
            if key == "argp_rad":
                series = np.degrees(np.unwrap(np.radians(series)))
            ax.plot(t_day, series, lw=1.0, label=str(hist.get("label", "")))
        ax.set_xlabel("time [days]")
        ax.set_ylabel(ylabel)
        ax.set_title(question)
        if len(histories) <= 10:
            ax.legend(fontsize=7, ncol=2)
        _footer(fig, source)
        written.append(_save(fig, out_dir, filename))
    return written


def plot_hk_portrait(
    histories: list[dict[str, Any]],
    out_dir: Path,
    *,
    source: str,
) -> Path | None:
    """h-k phase portrait (h = e sin(omega), k = e cos(omega)) per candidate.

    A bounded loop indicates a frozen-like eccentricity vector.
    """
    if not histories:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for hist in histories:
        e = np.asarray(hist["e"], dtype=np.float64)
        omega = np.asarray(hist["argp_rad"], dtype=np.float64)
        ax.plot(e * np.cos(omega), e * np.sin(omega), lw=0.8, label=str(hist.get("label", "")))
    ax.set_xlabel("k = e cos(omega) [-]")
    ax.set_ylabel("h = e sin(omega) [-]")
    ax.set_title("Is the eccentricity vector bounded (closed h-k loop)?")
    ax.set_aspect("equal", adjustable="datalim")
    if len(histories) <= 10:
        ax.legend(fontsize=7)
    _footer(fig, source)
    return _save(fig, out_dir, "hk_phase_portrait.png")


def plot_score_histogram(
    scores: Any,
    out_dir: Path,
    *,
    source: str,
    score_definition: str,
) -> Path | None:
    """Distribution of finite screening scores (lower = more frozen)."""
    finite = _finite(scores)
    if finite.size == 0:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(finite, bins=min(60, max(10, finite.size // 20)), color="#4477AA")
    ax.set_xlabel("screening score [-] (lower = more frozen)")
    ax.set_ylabel("sample count")
    ax.set_title("How selective is the screening score across the Sobol set?")
    fig.text(0.01, 0.03, score_definition, fontsize=6, color="0.4", wrap=True)
    _footer(fig, source)
    return _save(fig, out_dir, "score_histogram.png")


def plot_ae_score_map(
    elements: np.ndarray,
    scores: Any,
    out_dir: Path,
    *,
    source: str,
) -> Path | None:
    """a-e map colored by screening score (finite scores only)."""
    el = np.asarray(elements, dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(sc)
    if not mask.any():
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.scatter(el[mask, 0], el[mask, 1], c=sc[mask], s=6, cmap="viridis")
    ax.set_xlabel("semi-major axis a [km]")
    ax.set_ylabel("eccentricity e [-]")
    ax.set_title("Where in (a, e) space do low frozen scores concentrate?")
    fig.colorbar(im, ax=ax, label="screening score [-]")
    _footer(fig, source)
    return _save(fig, out_dir, "ae_score_map.png")


def plot_i_omega_map(
    elements: np.ndarray,
    scores: Any,
    out_dir: Path,
    *,
    source: str,
) -> Path | None:
    """i-omega stability map colored by screening score."""
    el = np.asarray(elements, dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(sc)
    if not mask.any():
        return None
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.scatter(el[mask, 2], el[mask, 4], c=sc[mask], s=6, cmap="viridis")
    ax.set_xlabel("inclination i [deg]")
    ax.set_ylabel("argument of periapsis omega [deg]")
    ax.set_title("Which (i, omega) combinations screen as stable?")
    fig.colorbar(im, ax=ax, label="screening score [-]")
    _footer(fig, source)
    return _save(fig, out_dir, "i_omega_stability_map.png")


def plot_perilune_safety(
    candidates: list[dict[str, Any]],
    out_dir: Path,
    *,
    source: str,
    safety_floor_km: float,
) -> Path | None:
    """Minimum perilune altitude vs score for the candidate set."""
    if not candidates:
        return None
    h_min = np.array(
        [float(c["summary"]["h_peri_min_km"]) for c in candidates], dtype=np.float64
    )
    sc = np.array([float(c["screening_score"]) for c in candidates], dtype=np.float64)
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.scatter(sc, h_min, s=25, color="#EE6677")
    ax.axhline(safety_floor_km, color="0.3", lw=1.0, ls="--", label=f"safety floor {safety_floor_km:g} km")
    ax.set_xlabel("screening score [-]")
    ax.set_ylabel("minimum perilune altitude [km]")
    ax.set_title("Do the top candidates keep a safe perilune margin?")
    ax.legend(fontsize=8)
    _footer(fig, source)
    return _save(fig, out_dir, "perilune_safety_map.png")


def plot_screening_vs_validation(
    validated: list[dict[str, Any]],
    out_dir: Path,
    *,
    source: str,
) -> Path | None:
    """Screening score vs classical-SH validation score per candidate.

    Answers: does the (surrogate/GPU) screening ranking survive classical SH
    validation, or does it reorder?
    """
    pairs = [
        (float(c["screening_score"]), float(c["classification"]["score"]))
        for c in validated
        if np.isfinite(float(c["screening_score"]))
        and isinstance(c.get("classification"), dict)
        and _is_finite_score(c["classification"].get("score"))
    ]
    if not pairs:
        return None
    plt = _mpl()
    xs, ys = zip(*pairs, strict=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(xs, ys, s=30, color="#228833")
    ax.set_xlabel("screening score [-] (screening backend)")
    ax.set_ylabel("validation score [-] (classical SH)")
    ax.set_title("Does the screening ranking survive classical SH validation?")
    _footer(fig, source)
    return _save(fig, out_dir, "screening_vs_validation_score.png")


def _is_finite_score(value: Any) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def generate_frozen_report_figures(
    run_dir: str | Path,
    *,
    out_subdir: str = "figures",
) -> list[Path]:
    """Generate the full R31 figure set from a completed pipeline run directory.

    Reads the stage file contracts written by
    :class:`~lunaris.analysis.frozen.search.FrozenSearchPipeline`; skips any
    figure whose stage output is missing.
    """
    from lunaris.batch.summary import SCORE_DEFINITION, summarize_ensemble

    from .search import (
        STAGE0_SAMPLES,
        STAGE1_SCREENING,
        STAGE2_CANDIDATES,
        STAGE3_VALIDATION,
    )

    run_path = Path(run_dir)
    out_dir = run_path / out_subdir
    written: list[Path] = []
    source = f"frozen-search run {run_path.name}"

    elements = scores = t_out = Y_out = None
    manifest_path = run_path / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    mu = float(manifest.get("config", {}).get("mu_m3s2", 4.9028001224453001e12))
    r_ref = float(manifest.get("config", {}).get("reference_radius_m", 1.7374e6))

    samples_path = run_path / STAGE0_SAMPLES
    screening_path = run_path / STAGE1_SCREENING
    if samples_path.exists() and screening_path.exists():
        with np.load(samples_path) as data:
            elements = np.asarray(data["elements"], dtype=np.float64)
        with np.load(screening_path) as data:
            summary = summarize_ensemble(
                data["t_out"],
                data["Y_out"],
                data["impact_flags"],
                data["t_impact"],
                mu_m3s2=mu,
                r_ref_m=r_ref,
            )
            scores = np.asarray(summary["fields"]["score"], dtype=np.float64)
            t_out = np.asarray(data["t_out"], dtype=np.float64)
            Y_out = np.asarray(data["Y_out"], dtype=np.float64)

    if elements is not None and scores is not None:
        for fn in (plot_score_histogram,):
            path = fn(scores, out_dir, source=source, score_definition=SCORE_DEFINITION)
            if path:
                written.append(path)
        for fn in (plot_ae_score_map, plot_i_omega_map):
            path = fn(elements, scores, out_dir, source=source)
            if path:
                written.append(path)

    candidates_path = run_path / STAGE2_CANDIDATES
    candidates: list[dict[str, Any]] = []
    if candidates_path.exists():
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))["candidates"]
        path = plot_perilune_safety(
            candidates,
            out_dir,
            source=source,
            safety_floor_km=float(manifest.get("config", {}).get("perilune_safety_min_m", 0.0))
            / 1_000.0,
        )
        if path:
            written.append(path)

    # Element histories for the top candidates, recomputed from the screening
    # block (histories are not persisted per-sample; the screening npz is).
    if candidates and elements is not None and Y_out is not None:
        from lunaris.batch.summary import _osculating_elements

        histories: list[dict[str, Any]] = []
        for record in candidates[:8]:
            j = int(record["sample_index"])
            y = Y_out[:, j, :]
            a_m, e, _inc, argp = _osculating_elements(y[:, :3], y[:, 3:], mu)
            histories.append(
                {
                    "t_s": t_out,
                    "e": e,
                    "h_peri_m": a_m * (1.0 - e) - r_ref,
                    "argp_rad": argp,
                    "label": f"#{j}",
                }
            )
        written.extend(plot_element_histories(histories, out_dir, source=source))
        path = plot_hk_portrait(histories, out_dir, source=source)
        if path:
            written.append(path)

    validation_path = run_path / STAGE3_VALIDATION
    if validation_path.exists():
        validated = json.loads(validation_path.read_text(encoding="utf-8"))["candidates"]
        path = plot_screening_vs_validation(validated, out_dir, source=source)
        if path:
            written.append(path)

    logger.info("frozen report figures written: %d -> %s", len(written), out_dir)
    return written


__all__ = [
    "generate_frozen_report_figures",
    "plot_ae_score_map",
    "plot_element_histories",
    "plot_hk_portrait",
    "plot_i_omega_map",
    "plot_perilune_safety",
    "plot_score_histogram",
    "plot_screening_vs_validation",
]
