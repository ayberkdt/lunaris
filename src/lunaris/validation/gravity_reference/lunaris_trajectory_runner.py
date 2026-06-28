"""Trajectory-validation runner.

Two regimes:

* **Fail-closed** for incomplete reference classes (mission ephemerides, or any
  manifest without a committed gravity-only arc): the runner reports
  ``REFERENCE_GENERATION_REQUIRED`` and never fabricates a comparison.
* **Real comparison** when a complete, checksummed gravity-only reference arc is
  present. Lunaris propagates the manifest initial state with its production
  propagator (``core.propagation.propagator.propagate`` over ``DynamicsEngine``) under a
  gravity-only force model, and the result is compared epoch-by-epoch against the
  immutable reference.

Frame scope
-----------
This runner validates the well-posed *non-rotating* gravity-only system: the
spherical-harmonic field is held fixed in the integration frame
(``state_frame == gravity_fixed_frame``), which Lunaris realises through
``allow_identity_rotation=True``. A manifest that asks for a rotating body-fixed
field returns ``INCOMPLETE_CONTRACT`` rather than silently propagating the wrong
dynamics, because SPICE-based lunar orientation is not part of this comparison.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.validation.gravity_reference.independent_field_oracle import (
    coefficients_from_json,
    geopotential,
)
from lunaris.validation.gravity_reference.manifest import (
    STATUS_INCOMPLETE_CLASSES,
    ManifestError,
    ResolvedTrajectoryManifest,
    load_trajectory_manifest,
)
from lunaris.validation.gravity_reference.reporting import trajectory_report_markdown
from lunaris.validation.gravity_reference.source_hashes import (
    atomic_write_json,
    atomic_write_text,
    load_json,
    sha256_file,
    utc_now_iso,
)
from lunaris.validation.gravity_reference.thresholds import (
    INCOMPLETE_CONTRACT,
    REFERENCE_GENERATION_REQUIRED,
    classify_trajectory_metrics,
)
from lunaris.validation.gravity_reference.trajectory_metrics import (
    compute_trajectory_metrics,
    specific_energy_drift,
)
from lunaris.validation.gravity_reference.trajectory_reference_io import load_trajectory_csv


class _ContractViolation(ValueError):
    """A manifest/reference contract the runner refuses to compare across."""


def _enforce_trajectory_contract(payload: dict[str, Any], epochs_s: np.ndarray) -> None:
    """Fail closed unless the reference grid and frames honour the manifest.

    The runner never silently papers over a mismatch (no interpolation, no frame
    coercion); a violation here surfaces as ``INCOMPLETE_CONTRACT`` instead of a
    misleading PASS/FAIL.
    """
    frames = payload["frames"]
    if str(frames["comparison_frame"]) != str(frames["state_frame"]):
        raise _ContractViolation(
            f"comparison_frame={frames['comparison_frame']!r} must equal "
            f"state_frame={frames['state_frame']!r}; RIC/Cartesian errors are computed "
            "in the state frame."
        )

    epochs = np.asarray(epochs_s, dtype=np.float64)
    if epochs.shape[0] < 2:
        raise _ContractViolation("reference trajectory needs at least two epochs.")
    steps = np.diff(epochs)
    declared_step = float(payload["time"]["output_step_s"])
    declared_duration = float(payload["time"]["duration_s"])
    tol = max(declared_step * 1e-6, 1e-6)
    expected_intervals = int(round(declared_duration / declared_step))
    expected_duration = float(expected_intervals) * declared_step
    if abs(expected_duration - declared_duration) > tol:
        raise _ContractViolation(
            "manifest time.duration_s must be an integer multiple of "
            f"output_step_s={declared_step}; got duration_s={declared_duration}."
        )
    expected_count = expected_intervals + 1
    if epochs.shape[0] != expected_count:
        raise _ContractViolation(
            f"reference sample count {epochs.shape[0]} does not match manifest "
            f"duration_s/output_step_s grid: expected {expected_count} samples."
        )
    if not np.allclose(steps, declared_step, atol=tol, rtol=0.0):
        raise _ContractViolation(
            "reference epochs are not uniformly spaced at the manifest "
            f"output_step_s={declared_step}: observed step range "
            f"[{float(steps.min())}, {float(steps.max())}]."
        )
    observed_duration = float(epochs[-1] - epochs[0])
    if abs(observed_duration - declared_duration) > tol:
        raise _ContractViolation(
            f"reference duration {observed_duration} s does not match manifest "
            f"duration_s={declared_duration}."
        )


def _git_value(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(cwd), check=False, capture_output=True, text=True, timeout=5
        )
    except Exception:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _run_provenance(manifest: ResolvedTrajectoryManifest, out_dir: Path) -> dict[str, Any]:
    repo = manifest.path
    for parent in (manifest.path.parent, *manifest.path.parents):
        if (parent / "pyproject.toml").exists():
            repo = parent
            break
    status_short = _git_value(["status", "--short"], repo)
    return {
        "created_at_utc": utc_now_iso(),
        "manifest_path": str(manifest.path),
        "manifest_sha256": sha256_file(manifest.path),
        "reference_file_sha256": (
            sha256_file(manifest.reference_path) if manifest.reference_path else None
        ),
        "output_dir": str(out_dir),
        "git_commit": _git_value(["rev-parse", "HEAD"], repo),
        "git_dirty": bool(status_short),
        "git_status_short": status_short,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "implementation_path": "lunaris.core.propagation.propagator.propagate",
        "dtype": "float64",
        "backend": "cpu",
    }


def _manifest_contract_failure(
    manifest_path: Path,
    out_dir: Path,
    reason: str,
) -> dict[str, Any]:
    try:
        payload = load_json(manifest_path)
    except Exception:
        payload = {}
    status = {"status": INCOMPLETE_CONTRACT, "reason": reason}
    manifest_sha256 = sha256_file(manifest_path) if manifest_path.exists() else None
    atomic_write_json(out_dir / "resolved_manifest.json", payload)
    atomic_write_json(out_dir / "validation_status.json", status)
    atomic_write_json(
        out_dir / "run_provenance.json",
        {
            "created_at_utc": utc_now_iso(),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "output_dir": str(out_dir),
        },
    )
    return {"status": status["status"], "out_dir": str(out_dir), "reason": reason}


def _load_gravity(manifest: ResolvedTrajectoryManifest) -> tuple[Any, Callable[[np.ndarray], float]]:
    """Return (Lunaris GravityModel, independent potential fn) for the manifest.

    The potential function is the *independent* numpy oracle (no Condon-Shortley
    phase, matching the geodesy/GRAIL convention), so the energy invariant is
    evaluated against a reference that does not share the Lunaris kernel and does
    not depend on any (optional) engine potential API.
    """
    from lunaris.physics.spherical_harmonics import GravityModel

    gravity = manifest.payload["gravity"]
    degree = int(gravity["degree"])
    coeff_path = Path(gravity["coefficient_file"])
    if not coeff_path.is_absolute():
        from lunaris.validation.gravity_reference.source_hashes import find_repo_root

        coeff_path = find_repo_root(manifest.path) / coeff_path
    if coeff_path.suffix.lower() != ".json":
        raise ValueError("Trajectory gravity coefficient_file must be a JSON fixture.")
    _deg, r_ref, mu, c, s = coefficients_from_json(load_json(coeff_path))
    model = GravityModel.from_arrays(degree, r_ref, mu, c, s)
    if int(model.degree_max) != degree:
        raise ValueError(f"Loaded degree {model.degree_max} != manifest degree {degree}.")

    def potential_fn(r: np.ndarray) -> float:
        return float(
            geopotential(
                np.asarray(r, dtype=np.float64),
                mu_m3_s2=mu,
                reference_radius_m=r_ref,
                c_coeffs=c,
                s_coeffs=s,
                degree=degree,
            )
        )

    return model, potential_fn


def _propagate_lunaris(
    model: Any,
    y0: np.ndarray,
    epochs_s: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """Propagate a gravity-only orbit and sample it on the reference epoch grid."""
    from lunaris.common.type_defs import (
        PerturbationFlags,
        PropagatorConfig,
        SpacecraftProps,
        TimeConfig,
    )
    from lunaris.core.dynamics import DynamicsEngine
    from lunaris.core.propagation.propagator import propagate

    epochs_s = np.asarray(epochs_s, dtype=np.float64)
    step = float(epochs_s[1] - epochs_s[0])
    duration = float(epochs_s[-1] - epochs_s[0])

    engine = DynamicsEngine(
        sc_props=SpacecraftProps(mass_kg=100.0, area_m2=1.0, cr=1.3),
        flags=PerturbationFlags(enable_sh=True),
        gravity_model=model,
        ephem_manager=None,
        surface_provider=None,
        earth_j2=None,
        allow_identity_rotation=True,  # field fixed in the integration frame
    )
    cfg = PropagatorConfig(
        method="DOP853", rtol=float(rtol), atol=float(atol), user_max_step_s=step
    )
    time_cfg = TimeConfig(duration_s=duration, output_dt_s=step, t0_s=float(epochs_s[0]))
    result = propagate(engine, np.asarray(y0, dtype=np.float64), cfg, time_cfg=time_cfg)

    t_out = np.asarray(result.t, dtype=np.float64)
    states = np.asarray(result.y, dtype=np.float64)
    # Fail closed: the comparison is only meaningful when Lunaris is sampled on
    # the exact reference epoch grid. We do NOT silently interpolate (that would
    # inject an alignment error and mask a real disagreement); instead we require
    # the propagator's output grid to match the reference epochs.
    atol = max(step * 1e-6, 1e-6)
    if t_out.shape[0] != epochs_s.shape[0] or not np.allclose(
        t_out - t_out[0], epochs_s - epochs_s[0], atol=atol, rtol=0.0
    ):
        raise _ContractViolation(
            "Lunaris output grid does not align with the reference epochs "
            f"(got {t_out.shape[0]} samples, expected {epochs_s.shape[0]}); the "
            "reference must be on a uniform grid whose step and duration match the "
            "manifest so no interpolation is needed."
        )
    return states


def run_trajectory_validation(manifest_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Validate a gravity-only trajectory contract against an external reference."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest = load_trajectory_manifest(manifest_path)
    except ManifestError as exc:
        return _manifest_contract_failure(manifest_path, out, str(exc))

    payload = manifest.payload
    reference_class = payload["reference_class"]

    # --- Fail-closed paths ---------------------------------------------------
    if reference_class in STATUS_INCOMPLETE_CLASSES or manifest.reference_path is None:
        status = {
            "status": REFERENCE_GENERATION_REQUIRED,
            "reason": (
                "No complete immutable gravity-only trajectory reference is present. "
                "Generate one with a pinned independent tool and update the manifest."
            ),
        }
        atomic_write_json(out / "resolved_manifest.json", payload)
        atomic_write_json(out / "validation_status.json", status)
        atomic_write_json(
            out / "run_provenance.json", _run_provenance(manifest, out)
        )
        return {"status": status["status"], "out_dir": str(out), "reason": status["reason"]}

    frames = payload["frames"]
    if str(frames["state_frame"]) != str(frames["gravity_fixed_frame"]):
        status = {
            "status": INCOMPLETE_CONTRACT,
            "reason": (
                "state_frame and gravity_fixed_frame differ: this runner validates the "
                "non-rotating gravity-only system (field fixed in the integration frame). "
                "Rotating body-fixed propagation requires SPICE lunar orientation and is "
                "out of scope for this comparison."
            ),
        }
        atomic_write_json(out / "resolved_manifest.json", payload)
        atomic_write_json(out / "validation_status.json", status)
        atomic_write_json(out / "run_provenance.json", _run_provenance(manifest, out))
        return {"status": status["status"], "out_dir": str(out), "reason": status["reason"]}

    # --- Real comparison -----------------------------------------------------
    reference = load_trajectory_csv(manifest.reference_path)
    epochs = reference["epoch_s"]
    ref_states = reference["state"]

    try:
        _enforce_trajectory_contract(payload, epochs)
        model, potential_fn = _load_gravity(manifest)
        integration = payload.get("integration", {}) if isinstance(payload.get("integration"), dict) else {}
        rtol = float(integration.get("rtol", 1e-12))
        atol = float(integration.get("atol", 1e-15))
        y0 = np.asarray(payload["initial_state"]["state"], dtype=np.float64)
        lunaris_states = _propagate_lunaris(model, y0, epochs, rtol=rtol, atol=atol)
    except _ContractViolation as exc:
        status = {"status": INCOMPLETE_CONTRACT, "reason": str(exc)}
        atomic_write_json(out / "resolved_manifest.json", payload)
        atomic_write_json(out / "validation_status.json", status)
        atomic_write_json(out / "run_provenance.json", _run_provenance(manifest, out))
        return {"status": status["status"], "out_dir": str(out), "reason": status["reason"]}

    metrics = compute_trajectory_metrics(ref_states, lunaris_states)
    metrics["lunaris_energy_drift"] = specific_energy_drift(lunaris_states, potential_fn)
    metrics["reference_energy_drift"] = specific_energy_drift(ref_states, potential_fn)

    status = classify_trajectory_metrics(metrics, payload["comparison"])
    provenance = _run_provenance(manifest, out)

    atomic_write_json(out / "resolved_manifest.json", payload)
    atomic_write_json(out / "run_provenance.json", provenance)
    atomic_write_json(out / "validation_status.json", status)
    atomic_write_json(out / "trajectory_metrics_summary.json", metrics)
    _write_samples_csv(out / "trajectory_samples.csv", epochs, ref_states, lunaris_states)
    atomic_write_text(
        out / "comparison_report.md",
        trajectory_report_markdown(
            manifest=payload, status=status, metrics=metrics, provenance=provenance
        ),
    )
    return {
        "status": status["status"],
        "out_dir": str(out),
        "metrics": metrics,
        "provenance": provenance,
    }


def _write_samples_csv(
    path: Path, epochs: np.ndarray, ref: np.ndarray, got: np.ndarray
) -> None:
    header = (
        "epoch_s,ref_x_m,ref_y_m,ref_z_m,lun_x_m,lun_y_m,lun_z_m,"
        "pos_err_m,vel_err_m_s\n"
    )
    lines = [header]
    for i in range(epochs.shape[0]):
        pos_err = float(np.linalg.norm(got[i, :3] - ref[i, :3]))
        vel_err = float(np.linalg.norm(got[i, 3:] - ref[i, 3:]))
        lines.append(
            f"{epochs[i]:.6f},{ref[i,0]:.9e},{ref[i,1]:.9e},{ref[i,2]:.9e},"
            f"{got[i,0]:.9e},{got[i,1]:.9e},{got[i,2]:.9e},{pos_err:.9e},{vel_err:.9e}\n"
        )
    atomic_write_text(path, "".join(lines))
