"""Single pre-training go/no-go gate for ST-LRPS.

Consolidates the checks that must hold before GPU hours are spent into ONE
report with a single ``go`` decision, written to ``preflight_report.json``:

* dataset validity (folds the existing ``validate_dataset_file`` report);
* split integrity — train/val/test/ood index sets are disjoint, train non-empty;
* OOD band ordering — for ``ood_*_altitude`` policies the held-out band sits
  strictly beyond the train band;
* train-only scaler — the scaler was fit on train rows only (no leakage);
* determinism consistency — the requested reproducibility flags are coherent;
* resource headroom — enough free disk (and, on CUDA, some free VRAM).

The checks are pure and unit-testable. ``run_preflight`` is tolerant of missing
inputs (each check ``skip``s when its input is absent, e.g. independent
train/val files have no in-file split to verify). Only ``fail`` statuses block
the run; ``warn`` is recorded but does not stop training.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PASS = "pass"
FAIL = "fail"
WARN = "warn"
SKIP = "skip"

# Minimum free disk on the run directory before training (checkpoints + logs).
_MIN_FREE_DISK_BYTES = 256 * 1024 * 1024
# Free VRAM below this only warns (estimates are model-dependent and rough).
_MIN_FREE_VRAM_WARN_BYTES = 512 * 1024 * 1024
# Altitude ordering tolerance [km] so a shared boundary point is not a violation.
_ALT_ORDER_TOL_KM = 1e-6


@dataclass
class PreflightCheck:
    name: str
    status: str
    detail: str = ""


@dataclass
class PreflightReport:
    go: bool
    checks: list[PreflightCheck] = field(default_factory=list)

    def summary(self) -> str:
        bad = [f"{c.name}: {c.detail}" for c in self.checks if c.status == FAIL]
        return "; ".join(bad) if bad else "all checks passed"

    def to_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0}
        for c in self.checks:
            counts[c.status] = counts.get(c.status, 0) + 1
        return {
            "schema_version": 1,
            "go": bool(self.go),
            "summary_counts": counts,
            "checks": [asdict(c) for c in self.checks],
        }


class PreflightError(RuntimeError):
    """Raised when the preflight gate fails and the caller did not allow bypass."""


def _index_set(indices: Any) -> set[int]:
    arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    return set(int(v) for v in arr.tolist())


def check_dataset_validation(report: Mapping[str, Any] | None) -> PreflightCheck:
    if report is None:
        return PreflightCheck("dataset_validation", SKIP, "no dataset validation report provided")
    if report.get("passed"):
        warns = report.get("warnings") or []
        detail = "dataset valid" + (f" ({len(warns)} warning(s))" if warns else "")
        return PreflightCheck("dataset_validation", PASS, detail)
    errors = report.get("errors") or []
    return PreflightCheck("dataset_validation", FAIL, "; ".join(str(e) for e in errors) or "dataset invalid")


def check_split_integrity(splits: Mapping[str, Any] | None) -> PreflightCheck:
    if splits is None:
        return PreflightCheck("split_integrity", SKIP, "no in-file split (independent train/val files)")
    names = [n for n in ("train", "val", "test", "ood") if n in splits]
    sets = {n: _index_set(splits[n]) for n in names}
    overlaps = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = sets[a] & sets[b]
            if inter:
                overlaps.append(f"{a}∩{b}={len(inter)}")
    if overlaps:
        return PreflightCheck("split_integrity", FAIL, "split index leakage: " + ", ".join(overlaps))
    if not sets.get("train"):
        return PreflightCheck("split_integrity", FAIL, "empty train split")
    sizes = ", ".join(f"{n}={len(sets[n])}" for n in names)
    return PreflightCheck("split_integrity", PASS, f"disjoint splits ({sizes})")


def check_ood_band_ordering(
    split_policy: str | None,
    split_manifest: Mapping[str, Any] | None,
) -> PreflightCheck:
    policy = str(split_policy or "").strip().lower()
    if policy not in {"ood_low_altitude", "ood_high_altitude"}:
        return PreflightCheck("ood_band_ordering", SKIP, f"policy {policy or 'unset'!r} is not an OOD altitude split")
    ranges = (split_manifest or {}).get("altitude_range_per_split")
    if not isinstance(ranges, Mapping):
        return PreflightCheck("ood_band_ordering", SKIP, "no altitude_range_per_split in manifest")
    train = ranges.get("train") or {}
    train_lo, train_hi = train.get("min"), train.get("max")
    if train_lo is None or train_hi is None:
        return PreflightCheck("ood_band_ordering", SKIP, "train altitude range unavailable")
    violations = []
    for held in ("val", "test", "ood"):
        band = ranges.get(held) or {}
        b_lo, b_hi = band.get("min"), band.get("max")
        if b_lo is None or b_hi is None:
            continue
        if policy == "ood_low_altitude":
            # Held-out band must sit strictly BELOW the train band.
            if b_hi > float(train_lo) + _ALT_ORDER_TOL_KM:
                violations.append(f"{held}.max={b_hi:.3f} not below train.min={float(train_lo):.3f}")
        else:  # ood_high_altitude: held-out band strictly ABOVE train band.
            if b_lo < float(train_hi) - _ALT_ORDER_TOL_KM:
                violations.append(f"{held}.min={b_lo:.3f} not above train.max={float(train_hi):.3f}")
    if violations:
        return PreflightCheck("ood_band_ordering", FAIL, "OOD band overlaps train: " + "; ".join(violations))
    return PreflightCheck("ood_band_ordering", PASS, f"{policy} held-out band is beyond the train band")


def check_scaler_train_only(scaler_provenance: Mapping[str, Any] | None) -> PreflightCheck:
    if scaler_provenance is None:
        return PreflightCheck("scaler_train_only", SKIP, "no scaler provenance provided")
    fit_scope = scaler_provenance.get("fit_scope")
    if fit_scope is None:
        return PreflightCheck("scaler_train_only", SKIP, "scaler provenance has no fit_scope")
    if str(fit_scope) == "train_only":
        return PreflightCheck("scaler_train_only", PASS, "scaler fit on train rows only")
    return PreflightCheck(
        "scaler_train_only", FAIL,
        f"scaler fit_scope={fit_scope!r}: validation/test/OOD rows may leak into the target scale",
    )


def check_determinism_consistency(
    *,
    deterministic: bool,
    benchmark_cudnn: bool,
    determinism_flags: Mapping[str, Any] | None = None,
) -> PreflightCheck:
    if deterministic and benchmark_cudnn:
        return PreflightCheck(
            "determinism_consistency", WARN,
            "deterministic=True with benchmark_cudnn=True: the autotuner makes runs non-reproducible",
        )
    if deterministic and determinism_flags is not None:
        if determinism_flags.get("use_deterministic_algorithms") is False:
            return PreflightCheck(
                "determinism_consistency", WARN,
                "deterministic requested but use_deterministic_algorithms could not be enabled",
            )
    mode = "deterministic" if deterministic else "throughput (non-deterministic)"
    return PreflightCheck("determinism_consistency", PASS, f"reproducibility mode: {mode}")


# Environment variables that indicate the scheduler allocated a GPU to this job.
_GPU_ALLOCATION_ENV_VARS = ("SLURM_JOB_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE", "CUDA_VISIBLE_DEVICES")


def check_cuda_visibility(
    *,
    device_type: str,
    environ: Mapping[str, str] | None = None,
) -> PreflightCheck:
    """FAIL when the job context demands a GPU but training resolved to CPU.

    The silent CPU-fallback is the classic HPC footgun: a wrong CUDA module or
    a CPU-only torch wheel makes ``get_device()`` quietly return ``cpu`` and a
    400-epoch job crawls through its GPU-queue walltime. Two demand signals:

    * ``LUNARIS_REQUIRE_CUDA=1`` — explicit; exported by the ``hpc/*.sbatch``
      training templates (and settable anywhere a CPU run would be a mistake);
    * a scheduler GPU allocation — any of ``SLURM_JOB_GPUS`` / ``SLURM_GPUS`` /
      ``SLURM_GPUS_ON_NODE`` / ``CUDA_VISIBLE_DEVICES`` set non-empty
      (an empty string or ``-1`` explicitly disables GPUs and does not count).

    With no demand signal a CPU device is a normal local run → ``skip``.
    ``--allow-preflight-fail`` / ``--skip-preflight`` remain the escape hatches.
    """
    env = os.environ if environ is None else environ
    require_raw = str(env.get("LUNARIS_REQUIRE_CUDA", "")).strip().lower()
    require = require_raw in ("1", "true", "yes", "on")

    def _demands_gpu(value: str | None) -> bool:
        text = str(value or "").strip()
        return text not in ("", "-1")

    allocated = [k for k in _GPU_ALLOCATION_ENV_VARS if k in env and _demands_gpu(env.get(k))]

    if str(device_type) == "cuda":
        detail = "CUDA device resolved"
        if allocated:
            detail += f" (GPU allocation visible via {', '.join(allocated)})"
        return PreflightCheck("cuda_visibility", PASS, detail)
    if require:
        return PreflightCheck(
            "cuda_visibility", FAIL,
            f"LUNARIS_REQUIRE_CUDA={require_raw!r} but training resolved device_type="
            f"{device_type!r}. torch cannot see a CUDA device — check the CUDA module "
            "and that the installed torch wheel is a CUDA build.",
        )
    if allocated:
        return PreflightCheck(
            "cuda_visibility", FAIL,
            f"the scheduler allocated a GPU ({', '.join(allocated)}) but training resolved "
            f"device_type={device_type!r}; a silent CPU run would burn the GPU walltime. "
            "Check the CUDA module / torch build, or unset the GPU request for a CPU run.",
        )
    return PreflightCheck(
        "cuda_visibility", SKIP,
        f"no GPU demanded (device_type={device_type!r}; local/CPU run)",
    )


def check_resource_headroom(
    *,
    out_dir: str | Path,
    device_type: str = "cpu",
    cuda_free_bytes: int | None = None,
) -> PreflightCheck:
    out = Path(out_dir).expanduser()
    probe = out if out.exists() else out.parent
    try:
        free = int(shutil.disk_usage(probe).free)
    except OSError as exc:
        return PreflightCheck("resource_headroom", WARN, f"could not stat free disk on {probe}: {exc}")
    if free < _MIN_FREE_DISK_BYTES:
        return PreflightCheck(
            "resource_headroom", FAIL,
            f"insufficient free disk on {probe}: {free / 1e6:.0f} MB < {_MIN_FREE_DISK_BYTES / 1e6:.0f} MB",
        )
    detail = f"free disk {free / 1e9:.1f} GB"
    if str(device_type) == "cuda" and cuda_free_bytes is not None:
        if int(cuda_free_bytes) < _MIN_FREE_VRAM_WARN_BYTES:
            return PreflightCheck(
                "resource_headroom", WARN,
                f"{detail}; low free VRAM {int(cuda_free_bytes) / 1e6:.0f} MB",
            )
        detail += f"; free VRAM {int(cuda_free_bytes) / 1e9:.1f} GB"
    return PreflightCheck("resource_headroom", PASS, detail)


def run_preflight(
    *,
    out_dir: str | Path,
    deterministic: bool = True,
    benchmark_cudnn: bool = False,
    device_type: str = "cpu",
    cuda_free_bytes: int | None = None,
    validation_report: Mapping[str, Any] | None = None,
    splits: Mapping[str, Any] | None = None,
    split_policy: str | None = None,
    split_manifest: Mapping[str, Any] | None = None,
    scaler_provenance: Mapping[str, Any] | None = None,
    determinism_flags: Mapping[str, Any] | None = None,
) -> PreflightReport:
    """Run every pre-training check and return one go/no-go report.

    ``go`` is ``True`` iff no check has status ``fail``. Inputs that are ``None``
    cause their check to ``skip`` rather than fail, so the gate works for both
    in-file splits and independent train/val files.
    """
    checks: list[PreflightCheck] = [
        check_dataset_validation(validation_report),
        check_split_integrity(splits),
        check_ood_band_ordering(split_policy, split_manifest),
        check_scaler_train_only(scaler_provenance),
        check_determinism_consistency(
            deterministic=deterministic,
            benchmark_cudnn=benchmark_cudnn,
            determinism_flags=determinism_flags,
        ),
        check_cuda_visibility(device_type=device_type),
        check_resource_headroom(
            out_dir=out_dir, device_type=device_type, cuda_free_bytes=cuda_free_bytes
        ),
    ]
    go = not any(c.status == FAIL for c in checks)
    return PreflightReport(go=go, checks=checks)


def write_preflight_report(out_dir: str | Path, report: PreflightReport) -> Path:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "preflight_report.json"
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "PASS",
    "FAIL",
    "WARN",
    "SKIP",
    "PreflightCheck",
    "PreflightReport",
    "PreflightError",
    "check_dataset_validation",
    "check_split_integrity",
    "check_ood_band_ordering",
    "check_scaler_train_only",
    "check_determinism_consistency",
    "check_cuda_visibility",
    "check_resource_headroom",
    "run_preflight",
    "write_preflight_report",
]
