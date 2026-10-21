"""Fractional-factorial screening campaign for ST-LRPS training options.

Six two-level factors are screened in a 2^(6-3) resolution-III design
(8 runs, generators D=AB, E=AC, F=BC), plus two anchor runs at the shipped
settings. An optional fold-over (8 runs, every sign flipped) turns the 16
runs into a resolution-IV design, which separates main effects from
two-factor interactions. All runs use one reduced-fidelity budget with a
warmup-stable-decay schedule, so each run ends annealed at the same point
of its schedule.

Why a screening design and not Bayesian optimisation: the factors are
mostly binary. A Gaussian-process optimiser needs about 2-10 x d initial
points in d = 6 dimensions before its model is informative, so with ~10
runs it never leaves the initial design. A 2-level fraction uses every run
for every main-effect estimate (each effect compares 4 runs against 4, or
8 against 8 after the fold-over). Continuous tuning of the one or two
factors that matter can follow with a small response-surface or Bayesian
step. See PLANS notes and the docstrings of ``FACTORS`` for each factor's
rationale.

Sub-commands::

    python tools/st_lrps_doe.py design   [--phase phase1|foldover|all]
    python tools/st_lrps_doe.py scenarios [--check]
    python tools/st_lrps_doe.py commands --data-dir DIR --out-root DIR [--phase ...]
    python tools/st_lrps_doe.py run      --data-dir DIR --out-root DIR [--phase ...] [--only NAME ...]
    python tools/st_lrps_doe.py analyze  --out-root DIR [--phase ...] [--at final|best]
    python tools/st_lrps_doe.py promote  --levels ABcDef --stage confirm|final --data-dir DIR --out-root DIR

``run`` trains sequentially and skips runs whose ``history.jsonl`` already
reaches the final epoch; it puts this checkout's ``src`` first on
``PYTHONPATH`` so a worktree trains its own code, not an editable install
of another checkout. ``analyze`` reads only ``history.jsonl``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_MODULE = "lunaris.surrogate.st_lrps.training.cli"
SCENARIO_DIR = REPO_ROOT / "hpc" / "scenarios"
SCENARIO_FILES = {
    "phase1": "st_lrps_trajectory_doe_phase1.jsonl",
    "foldover": "st_lrps_trajectory_doe_foldover.jsonl",
}
SEED = 42

# Dataset file names inside --data-dir (the 2026-07-04 cloud suite layout).
DATA_FILES = {
    "--train-data": "train_hybrid_10M.h5",
    "--val-data": "val_uniform_2M.h5",
    "--suite-manifest": "manifest.json",
}


@dataclass(frozen=True)
class Factor:
    key: str
    name: str
    minus: tuple[str, ...]
    plus: tuple[str, ...]
    why: str


FACTORS: tuple[Factor, ...] = (
    Factor(
        "A", "potential_weight",
        ("--w-u", "1.0"), ("--w-u", "10.0"),
        "Along-track drift follows the potential error almost independently of degree "
        "(band-transfer data); the potential term held ~4% of the shipped loss.",
    ),
    Factor(
        "B", "scale_free_terms",
        ("--direction-loss-weight", "0.1", "--relative-loss-weight", "0.05"),
        ("--direction-loss-weight", "0.02", "--relative-loss-weight", "0.01"),
        "Direction + relative terms held ~83% of the shipped loss and are dominated by "
        "mid/high altitudes where the residual barely moves an orbit.",
    ),
    Factor(
        "C", "low_degree_leakage",
        ("--low-degree-loss-weight", "0.0"), ("--low-degree-loss-weight", "100.0"),
        "Cross-fitted on Plan D, 20% of the 100-150 km potential-error power sits in "
        "degrees <= 25 (5.6% in <= 12), where the residual is zero; low degrees transfer "
        "most strongly into along-track drift. The trained term uses L = 12 (K = 169) "
        "so ~2.3k samples below 300 km per batch give a well-conditioned fit; weight 100 "
        "is ~10% of the loss at screening fidelity (2.5e-9 per batch on the 40-epoch H1 "
        "checkpoint against a 3e-6 total).",
    ),
    Factor(
        "D", "band_init",
        ("--band-init", "legacy"), ("--band-init", "siren_first"),
        "The legacy init starts the high band below the low band's frequency; the "
        "high band was found dead (230/256 merge columns ~0) in the 2026-09-17 run.",
    ),
    Factor(
        "E", "harmonic_penalty",
        ("--laplacian-mode", "diagnostic"),
        ("--laplacian-mode", "train", "--laplacian-weight", "1e16",
         "--laplacian-every-n-batches", "4"),
        "The residual potential is harmonic; the shipped model violates Laplace by "
        "~0.6% at 100-150 km. Mean lap^2 is 8.8e-23 s^-4 on the 40-epoch H1 checkpoint "
        "(total loss 3e-6) and 6.2e-24 on Plan D (7e-8). At 1e16 the penalty is 29% "
        "(screening) and 89% (converged) of the loss on the batches where it is applied, "
        "i.e. ~7% and ~22% averaged over every 4th batch.",
    ),
    Factor(
        "F", "altitude_balance",
        ("--use-altitude-balanced-loss",), ("--no-altitude-balanced-loss",),
        "Balanced 50-km bins give the 1000-km shell the weight of the 100-km shell; "
        "orbit energy error is set by the absolute potential error.",
    ),
)
FACTOR_KEYS = tuple(f.key for f in FACTORS)

# Shipped Plan D architecture and loss; trajectory-neutral corrections that
# every DoE run shares (uncentred potential, fixed loss balance at the value
# NTK-init clamped to in Plan D) are listed separately so the anchors can undo them.
BASE_FLAGS: tuple[str, ...] = (
    "--dataset-name", "data", "--split-seed", str(SEED), "--seed", str(SEED),
    "--hidden", "512", "--depth", "5", "--activation", "sine",
    "--model-preset", "recommended_physical_radial_decay", "--no-fourier",
    "--use-residual-blocks", "--n-bands", "2", "--output-envelope-auto",
    "--batch-size", "8192", "--lr", "7e-05", "--weight-decay", "1e-06", "--max-grad-norm", "0.5",
    "--a-sign", "auto", "--use-si",
    "--altitude-bin-width-km", "50.0", "--altitude-min-km", "100.0", "--altitude-max-km", "1000.0",
    "--use-radial-cross-loss", "--radial-loss-weight", "0.05", "--cross-loss-weight", "0.05",
    "--use-laplacian-regularization", "--laplacian-subset-size", "512",
    "--direction-loss-floor-abs", "1e-7", "--relative-loss-floor-abs", "1e-8",
    "--best-metric", "hybrid_rel", "--hybrid-relative-beta", "0.3",
    "--fit-rows", "2000000", "--num-workers", "4", "--pin-memory",
    # Every run logs the cross-fitted leakage diagnostic at the same L.
    "--low-degree-lmax", "12",
)
CORRECTION_FLAGS: tuple[str, ...] = (
    "--u-center", "zero", "--gradnorm-mode", "fixed", "--w-a", "2.0",
)
SHIPPED_CORRECTION_FLAGS: tuple[str, ...] = (
    "--u-center", "mean", "--gradnorm-mode", "ntk_init", "--w-a", "1.0",
    "--gradnorm-w-a-min", "0.05", "--gradnorm-w-a-max", "2.0",
)

# Reduced fidelity "M": 48 epochs x 488 batches x 8192 = 192M samples (~2.5% of
# the 800-epoch shipped run), 31 validation batches (254k uniform samples).
# Ramps are compressed in proportion so the run sees every training phase.
FIDELITY_FLAGS: tuple[str, ...] = (
    "--epochs", "48", "--max-train-batches", "488", "--max-val-batches", "31",
    "--warmup-epochs", "2", "--lr-schedule", "wsd", "--lr-decay-fraction", "0.25",
    "--min-lr-ratio", "0.01",
    "--potential-only-epochs", "2", "--accel-ramp-epochs", "6", "--accel-min-factor", "0.05",
    "--direction-loss-start-epoch", "4", "--direction-loss-ramp-epochs", "6",
    "--checkpoint-settle-epochs", "2", "--best-ckpt-start-epoch", "-1", "--patience", "1000",
    "--periodic-eval-count", "0", "--log-every-mode", "fixed", "--log-every", "200",
)
FIDELITY_EPOCHS = 48

# Promotion stages for the winning levels. "confirm" (stage B) trains on the
# full data for 200 epochs; "final" (stage C) repeats the 800-epoch shipped
# budget. Both end in a 20% (1-sqrt) cooldown, average weights, and select the
# checkpoint on the low-altitude potential RMS.
STAGE_FLAGS: dict[str, tuple[str, ...]] = {
    "confirm": (
        "--epochs", "200", "--warmup-epochs", "5", "--lr-schedule", "wsd", "--lr-decay-fraction", "0.2",
        "--min-lr-ratio", "0.01", "--potential-only-epochs", "5", "--accel-ramp-epochs", "20",
        "--accel-min-factor", "0.05", "--direction-loss-start-epoch", "10", "--direction-loss-ramp-epochs", "20",
        "--checkpoint-settle-epochs", "5", "--best-ckpt-start-epoch", "-1", "--patience", "1000",
        "--ema-decay", "0.999", "--best-metric", "potential_low", "--periodic-eval-count", "5",
    ),
    "final": (
        "--epochs", "800", "--warmup-epochs", "5", "--lr-schedule", "wsd", "--lr-decay-fraction", "0.2",
        "--min-lr-ratio", "0.01", "--potential-only-epochs", "10", "--accel-ramp-epochs", "80",
        "--accel-min-factor", "0.05", "--direction-loss-start-epoch", "30", "--direction-loss-ramp-epochs", "70",
        "--checkpoint-settle-epochs", "5", "--best-ckpt-start-epoch", "-1", "--patience", "1000",
        "--ema-decay", "0.999", "--best-metric", "potential_low", "--periodic-eval-count", "10",
    ),
}

# Responses read from history.jsonl. "log" responses are analysed as log10.
RESPONSES: tuple[tuple[str, bool], ...] = (
    ("val_u_low_rmse_m2_s2", True),     # primary: orbit-energy (phase) proxy
    ("val_accel_rmse_m_s2", True),      # field accuracy, unweighted
    ("val_loss_low_degree", False),     # leakage diagnostic (cross-fitted, can be <= 0)
    ("val_angular_mean_deg", True),
    ("val_wtz_frac", False),            # constraint: must stay ~0
)


@dataclass(frozen=True)
class Run:
    name: str
    phase: str
    levels: dict[str, int]       # factor key -> -1/+1 (empty for anchors)
    flags: tuple[str, ...]
    description: str


def _resolution3_rows() -> list[dict[str, int]]:
    return [
        {"A": a, "B": b, "C": c, "D": a * b, "E": a * c, "F": b * c}
        for c in (-1, 1)
        for b in (-1, 1)
        for a in (-1, 1)
    ]


def _factor_flags(levels: dict[str, int]) -> tuple[str, ...]:
    out: list[str] = []
    for f in FACTORS:
        out += list(f.plus if levels[f.key] > 0 else f.minus)
    return tuple(out)


def _all_minus() -> dict[str, int]:
    return dict.fromkeys(FACTOR_KEYS, -1)


def _signs(levels: dict[str, int]) -> str:
    return "".join(k.lower() if levels[k] < 0 else k for k in FACTOR_KEYS)


ANCHOR_CORRECTED = "DoE_P1_anchor_corrected"


def build_runs(phase: str) -> list[Run]:
    """Runs of ``phase`` ("phase1", "foldover" or "all"), in launch order.

    The fold-over of design point 8 (all factors at -) is the corrected
    anchor of phase 1, so the fold-over launches only 7 new runs and its
    analysis reuses the anchor.
    """
    if phase not in ("phase1", "foldover", "all"):
        raise ValueError(f"unknown phase {phase!r}")
    runs: list[Run] = []
    if phase in ("phase1", "all"):
        for i, lv in enumerate(_resolution3_rows(), start=1):
            runs.append(Run(
                f"DoE_P1_R{i:02d}_{_signs(lv)}", "phase1", lv,
                BASE_FLAGS + CORRECTION_FLAGS + _factor_flags(lv) + FIDELITY_FLAGS,
                f"2^(6-3) design point {i}: {_signs(lv)} (upper case = + level).",
            ))
        runs.append(Run(
            "DoE_P1_anchor_shipped", "anchor", {},
            BASE_FLAGS + SHIPPED_CORRECTION_FLAGS + _factor_flags(_all_minus()) + FIDELITY_FLAGS,
            "Shipped Plan D recipe (mean-centred potential, NTK-init w_a) at DoE fidelity.",
        ))
        runs.append(Run(
            ANCHOR_CORRECTED, "anchor", _all_minus(),
            BASE_FLAGS + CORRECTION_FLAGS + _factor_flags(_all_minus()) + FIDELITY_FLAGS,
            "Shipped recipe plus the shared corrections; all six factors at the - level.",
        ))
    if phase in ("foldover", "all"):
        for i, lv in enumerate(_resolution3_rows(), start=1):
            fl = {k: -v for k, v in lv.items()}
            if fl == _all_minus():
                continue
            runs.append(Run(
                f"DoE_FO_R{i:02d}_{_signs(fl)}", "foldover", fl,
                BASE_FLAGS + CORRECTION_FLAGS + _factor_flags(fl) + FIDELITY_FLAGS,
                f"Fold-over of design point {i}: {_signs(fl)}.",
            ))
    return runs


def design_runs(phase: str) -> list[Run]:
    """The runs whose levels form the orthogonal fraction analysed for ``phase``."""
    runs = build_runs("all")
    p1 = [r for r in runs if r.phase == "phase1"]
    fo = [r for r in runs if r.phase == "foldover"] + [r for r in runs if r.name == ANCHOR_CORRECTED]
    return {"phase1": p1, "foldover": fo, "all": p1 + fo}[phase]


def parse_levels(text: str) -> dict[str, int]:
    """``"ABcDef"`` (upper case = + level) -> ``{"A": 1, "B": 1, "C": -1, ...}``."""
    if len(text) != len(FACTOR_KEYS) or text.upper() != "".join(FACTOR_KEYS):
        raise ValueError(f"levels must spell {''.join(FACTOR_KEYS)} with case marking +/-, got {text!r}")
    return {ch.upper(): (1 if ch.isupper() else -1) for ch in text}


def promoted_run(levels: dict[str, int], stage: str, seed: int = SEED) -> Run:
    """Full-data run of one level combination at a promotion stage."""
    if stage not in STAGE_FLAGS:
        raise ValueError(f"stage must be one of {sorted(STAGE_FLAGS)}, got {stage!r}")
    i = BASE_FLAGS.index("--seed")
    base = BASE_FLAGS[: i + 1] + (str(seed),) + BASE_FLAGS[i + 2 :]
    return Run(
        f"DoE_{stage}_{_signs(levels)}_Seed{seed}", stage, levels,
        base + CORRECTION_FLAGS + _factor_flags(levels) + STAGE_FLAGS[stage],
        f"Promoted {_signs(levels)} at stage {stage}.",
    )


def render_scenarios(phase: str) -> str:
    """HPC scenario JSONL for one phase (data flags come from the sbatch common flags)."""
    runs = build_runs(phase)
    header = [
        f"# ST-LRPS trajectory-aware screening campaign, {phase} ({len(runs)} runs).",
        "# GENERATED by tools/st_lrps_doe.py; do not edit by hand.",
        "# Regenerate: python tools/st_lrps_doe.py scenarios",
        f"# Submit: sbatch --array=0-{len(runs) - 1} hpc/slurm_train_scenario_array.sbatch <this file> "
        "--train-data ... --val-data ... --suite-manifest ...",
    ]
    lines = [
        json.dumps({
            "name": r.name,
            "entrypoint": "lunaris-train",
            "description": r.description,
            "runtime_model_kind": "potential_autograd",
            "tags": ["potential_autograd", "doe", r.phase, _signs(r.levels) if r.levels else "shipped"],
            "flags": list(r.flags),
        })
        for r in runs
    ]
    return "\n".join(header + lines) + "\n"


def _command(run: Run, data_dir: Path, out_root: Path) -> list[str]:
    data = [tok for flag, fname in DATA_FILES.items() for tok in (flag, str(data_dir / fname))]
    return [sys.executable, "-m", TRAIN_MODULE, *data, *run.flags, "--out", str(out_root / run.name)]


def _train_env() -> dict[str, str]:
    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _history(run_dir: Path) -> list[dict]:
    path = run_dir / "history.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _is_complete(run_dir: Path, epochs: int = FIDELITY_EPOCHS) -> bool:
    rows = _history(run_dir)
    return bool(rows) and int(rows[-1].get("epoch", -1)) + 1 >= epochs


FLAGS_FILE = "doe_flags.json"


def _recorded_flags(run_dir: Path) -> list[str] | None:
    path = run_dir / FLAGS_FILE
    if not path.is_file():
        return None
    return list(json.loads(path.read_text(encoding="utf-8")))


def plan_launch(run: Run, run_dir: Path) -> str:
    """``"skip"``, ``"resume"``, ``"start"`` or ``"stale"`` for one screening run.

    A run directory is reused only if it was launched with exactly this run's
    flags (recorded in ``doe_flags.json`` at launch); anything else is stale.
    """
    if not run_dir.exists():
        return "start"
    if _recorded_flags(run_dir) != list(run.flags):
        return "stale"
    if _is_complete(run_dir):
        return "skip"
    return "resume" if (run_dir / "checkpoints" / "ckpt_last.pt").is_file() else "stale"


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


def lenth_pse(effects: Sequence[float]) -> float:
    """Lenth's pseudo standard error of unreplicated factorial effects."""
    abs_e = [abs(e) for e in effects if math.isfinite(e)]
    if not abs_e:
        return float("nan")
    s0 = 1.5 * statistics.median(abs_e)
    trimmed = [e for e in abs_e if e < 2.5 * s0] or abs_e
    return 1.5 * statistics.median(trimmed)


def _response_value(row: dict, key: str, log: bool) -> float:
    v = row.get(key)
    if v is None:
        return float("nan")
    v = float(v)
    if log:
        return math.log10(v) if v > 0 else float("nan")
    return v


def estimate_effects(levels: list[dict[str, int]], y: list[float]) -> dict[str, float]:
    """Main effects (mean at + minus mean at -) for each factor present in ``levels``."""
    keys = [k for k in FACTOR_KEYS if all(k in lv for lv in levels)]
    effects = {}
    for k in keys:
        plus = [yi for lv, yi in zip(levels, y, strict=True) if lv[k] > 0 and math.isfinite(yi)]
        minus = [yi for lv, yi in zip(levels, y, strict=True) if lv[k] < 0 and math.isfinite(yi)]
        effects[k] = (statistics.fmean(plus) - statistics.fmean(minus)) if plus and minus else float("nan")
    return effects


def contrast_effects(levels: list[dict[str, int]], y: list[float]) -> dict[str, float]:
    """Effects of every distinct product contrast of the design columns.

    For a regular two-level fraction of n runs these are n - 1 mutually
    orthogonal contrasts: the six main effects plus their alias chains (one
    for the 8-run fraction, nine for the 16-run fold-over combination).
    Lenth's PSE is computed over all of them, as the method intends.
    """
    n = len(levels)
    seen: dict[tuple[int, ...], str] = {}
    for size in range(1, len(FACTOR_KEYS) + 1):
        for combo in itertools.combinations(FACTOR_KEYS, size):
            col = tuple(math.prod(lv[k] for k in combo) for lv in levels)
            if all(v == 1 for v in col) or all(v == -1 for v in col):
                continue
            key = col if col[0] > 0 else tuple(-v for v in col)
            if key not in seen:
                seen[key] = "".join(combo) if col[0] > 0 else "-" + "".join(combo)
    effects = {}
    for col, label in seen.items():
        sign = -1.0 if label.startswith("-") else 1.0
        name = label.lstrip("-")
        effects[name] = sign * 2.0 * sum(c * yi for c, yi in zip(col, y, strict=True)) / n
    return effects


def is_orthogonal(levels: list[dict[str, int]]) -> bool:
    """True when every pair of factor columns is balanced and orthogonal."""
    cols = [[lv[k] for lv in levels] for k in FACTOR_KEYS]
    if any(sum(c) != 0 for c in cols):
        return False
    return all(sum(a * b for a, b in zip(c1, c2, strict=True)) == 0
               for i, c1 in enumerate(cols) for c2 in cols[i + 1:])


DECISION_T = 2.0


def _arc_proxy_rms(run_dir: Path, proxy: Path, device: str) -> float:
    """RMS predicted along-track error [m] from tools/st_lrps_arc_proxy.py, cached per run."""
    cache = run_dir / "evals" / "arc_proxy.json"
    if cache.is_file():
        rep = json.loads(cache.read_text(encoding="utf-8"))
        if Path(rep["summary"].get("proxy_dir", "")) == Path(proxy):
            return float(rep["summary"]["rms_along_track_m"])
    spec = importlib.util.spec_from_file_location("st_lrps_arc_proxy", REPO_ROOT / "tools" / "st_lrps_arc_proxy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rep = module.score_run(Path(proxy), run_dir, device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return float(rep["summary"]["rms_along_track_m"])


def analyze(
    out_root: Path, phase: str, at: str, arc_proxy: Path | None = None, device: str = "cuda"
) -> dict:
    """Main effects of the screening responses.

    With ``arc_proxy`` (a ``tools/st_lrps_arc_proxy.py build`` directory) every
    complete run is also scored at trajectory level: the RMS over the proxy's
    reference orbits of the predicted along-track error at the final time,
    for the checkpoint the runtime loads. That becomes the primary response.
    """
    responses = ((("arc_proxy_rms_m", True),) if arc_proxy else ()) + RESPONSES
    design = {r.name for r in design_runs(phase)}
    runs = build_runs(phase) if phase != "foldover" else build_runs("all")
    rows: list[dict] = []
    for r in runs:
        hist = _history(out_root / r.name)
        if not hist:
            rows.append({"name": r.name, "status": "missing"})
            continue
        if at == "best":
            best = int(hist[-1].get("best_epoch", 0) or 0)
            pick = next((h for h in hist if int(h.get("epoch", -1)) + 1 == best), hist[-1])
        else:
            pick = hist[-1]
        entry = {"name": r.name, "status": "complete" if _is_complete(out_root / r.name) else "partial",
                 "epoch": int(pick.get("epoch", -1)) + 1, **{k: r.levels.get(k, 0) for k in FACTOR_KEYS}}
        for key, _ in RESPONSES:
            entry[key] = pick.get(key)
        if arc_proxy and entry["status"] == "complete":
            entry["arc_proxy_rms_m"] = _arc_proxy_rms(out_root / r.name, Path(arc_proxy), device)
        rows.append(entry)

    design_rows = [
        (r, e) for r, e in zip(runs, rows, strict=True) if r.name in design and e.get("status") == "complete"
    ]
    report: dict = {"phase": phase, "at": at, "runs": rows, "effects": {}}
    missing = sorted(design - {r.name for r, _ in design_rows})
    if missing:
        # An incomplete fraction is no longer orthogonal: main effects would
        # silently absorb aliased effects. Report nothing rather than that.
        report["incomplete_design"] = missing
        return report
    levels = [r.levels for r, _ in design_rows]
    report["orthogonal"] = is_orthogonal(levels)
    for key, log in responses:
        y = [_response_value(e, key, log) for _, e in design_rows]
        if not all(math.isfinite(v) for v in y):
            report["effects"][key] = {"log10": log, "error": "non-finite response in a design run"}
            continue
        contrasts = contrast_effects(levels, y)
        pse = lenth_pse(list(contrasts.values()))
        main = {k: contrasts[k] for k in FACTOR_KEYS}
        report["effects"][key] = {
            "log10": log,
            "effects": main,
            "alias_contrasts": {k: v for k, v in contrasts.items() if k not in FACTOR_KEYS},
            "lenth_pse": pse,
            "t_ratio": {k: (v / pse if pse > 0 else float("nan")) for k, v in main.items()},
            "multiplier_plus_vs_minus": ({k: 10.0 ** v for k, v in main.items()} if log else None),
        }
    primary = report["effects"].get(responses[0][0], {})
    report["primary_response"] = responses[0][0]
    if "effects" in primary:
        # +1 / -1 only where |t| >= DECISION_T; 0 = no evidence either way.
        report["recommended_levels"] = {
            k: ((+1 if v < 0 else -1) if abs(primary["t_ratio"][k]) >= DECISION_T else 0)
            for k, v in primary["effects"].items()
        }
    return report


def _write_report(report: dict, out_root: Path) -> Path:
    tag = f"doe_{report['phase']}_{report['at']}"
    (out_root / f"{tag}_effects.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(out_root / f"{tag}_runs.csv", "w", newline="", encoding="utf-8") as fh:
        keys = ["name", "status", "epoch", *FACTOR_KEYS, "arc_proxy_rms_m", *(k for k, _ in RESPONSES)]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in report["runs"]:
            w.writerow(row)
    lines = [f"# DoE {report['phase']} at {report['at']} epoch", ""]
    if report.get("incomplete_design"):
        lines.append("Design incomplete; effects not estimated. Missing or unfinished runs:")
        lines += [f"- {name}" for name in report["incomplete_design"]]
    for key, block in report["effects"].items():
        if "error" in block:
            lines += [f"## {key}", "", f"- {block['error']}", ""]
            continue
        scale = "log10" if block["log10"] else "linear"
        lines += [f"## {key} ({scale}); Lenth PSE {block['lenth_pse']:.3g} over "
                  f"{len(block['effects']) + len(block['alias_contrasts'])} contrasts", ""]
        for f in FACTORS:
            e = block["effects"][f.key]
            t = block["t_ratio"][f.key]
            mult = block["multiplier_plus_vs_minus"]
            tail = f", x{mult[f.key]:.3g} going - to +" if mult else ""
            flag = "  <- |t| >= 2" if abs(t) >= DECISION_T else ""
            lines.append(f"- {f.key} {f.name}: effect {e:+.3g}, t {t:+.2f}{tail}{flag}")
        lines.append("")
    if report.get("recommended_levels"):
        rec = report["recommended_levels"]
        lines.append(f"Levels supported by the primary response {report.get('primary_response')} "
                     "(|t| >= 2; '?' = no evidence): "
                     + ", ".join(f"{k}{'+' if v > 0 else '-' if v < 0 else '?'}" for k, v in rec.items()))
    path = out_root / f"{tag}_effects.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_design = sub.add_parser("design", help="print the design matrix")
    p_design.add_argument("--phase", default="all", choices=["phase1", "foldover", "all"])
    p_scen = sub.add_parser("scenarios", help="write (or --check) the HPC scenario files")
    p_scen.add_argument("--check", action="store_true")
    for name in ("commands", "run"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", type=Path, required=True)
        p.add_argument("--out-root", type=Path, required=True)
        p.add_argument("--phase", default="phase1", choices=["phase1", "foldover", "all"])
        p.add_argument("--only", nargs="*", default=None, help="run names to include")
    p_pr = sub.add_parser("promote", help="print the full-data command for chosen levels")
    p_pr.add_argument("--levels", required=True, help="e.g. ABcDef (upper case = + level)")
    p_pr.add_argument("--stage", required=True, choices=sorted(STAGE_FLAGS))
    p_pr.add_argument("--seed", type=int, default=SEED)
    p_pr.add_argument("--data-dir", type=Path, required=True)
    p_pr.add_argument("--out-root", type=Path, required=True)
    p_an = sub.add_parser("analyze")
    p_an.add_argument("--out-root", type=Path, required=True)
    p_an.add_argument("--phase", default="phase1", choices=["phase1", "foldover", "all"])
    p_an.add_argument("--at", default="final", choices=["final", "best"])
    p_an.add_argument("--arc-proxy", type=Path, default=None,
                      help="tools/st_lrps_arc_proxy.py build directory; adds the trajectory-level primary response")
    p_an.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    if a.cmd == "design":
        print("run".ljust(36) + " ".join(FACTOR_KEYS))
        for r in build_runs(a.phase):
            print(r.name.ljust(36) + " ".join(("+" if r.levels[k] > 0 else "-") if r.levels else "."
                                              for k in FACTOR_KEYS))
        for f in FACTORS:
            print(f"{f.key} {f.name}: - {' '.join(f.minus)} | + {' '.join(f.plus)}")
        return 0
    if a.cmd == "scenarios":
        bad = 0
        for phase, fname in SCENARIO_FILES.items():
            path = SCENARIO_DIR / fname
            text = render_scenarios(phase)
            if a.check:
                if not path.is_file() or path.read_text(encoding="utf-8") != text:
                    print(f"stale: {path.relative_to(REPO_ROOT)}")
                    bad += 1
            else:
                path.write_text(text, encoding="utf-8")
                print(f"wrote {path.relative_to(REPO_ROOT)}")
        return 1 if bad else 0
    if a.cmd == "promote":
        run = promoted_run(parse_levels(a.levels), a.stage, a.seed)
        print(subprocess.list2cmdline(_command(run, a.data_dir, a.out_root)))
        return 0
    if a.cmd in ("commands", "run"):
        runs = [r for r in build_runs(a.phase) if not a.only or r.name in a.only]
        for r in runs:
            cmd = _command(r, a.data_dir, a.out_root)
            if a.cmd == "commands":
                print(subprocess.list2cmdline(cmd))
                continue
            run_dir = a.out_root / r.name
            action = plan_launch(r, run_dir)
            if action == "skip":
                print(f"[skip] {r.name} (complete, same flags)")
                continue
            if action == "stale":
                print(f"[stale] {run_dir} exists but was not launched with this run's flags "
                      f"(or has no checkpoint to resume); move it away and rerun.")
                return 2
            if action == "resume":
                cmd = [sys.executable, "-m", TRAIN_MODULE, "--resume-from", str(run_dir),
                       "--epochs", str(FIDELITY_EPOCHS)]
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / FLAGS_FILE).write_text(json.dumps(list(r.flags)), encoding="utf-8")
            print(f"[{action}] {r.name}", flush=True)
            a.out_root.mkdir(parents=True, exist_ok=True)
            with open(a.out_root / f"{r.name}.log", "a", encoding="utf-8") as log:
                rc = subprocess.run(
                    cmd, cwd=REPO_ROOT, env=_train_env(), stdout=log, stderr=subprocess.STDOUT, check=False
                ).returncode
            if rc != 0:
                print(f"[fail] {r.name} exit {rc}; see {r.name}.log")
                return rc
        return 0
    report = analyze(a.out_root, a.phase, a.at, a.arc_proxy, a.device)
    path = _write_report(report, a.out_root)
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
