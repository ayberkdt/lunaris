"""Trajectory-validation runner.

Two regimes:

* **Fail-closed** for incomplete reference classes (mission ephemerides, or any
  manifest without a committed gravity-only arc): the runner reports
  ``REFERENCE_GENERATION_REQUIRED`` and never fabricates a comparison.
* **Real comparison** when a complete, checksummed gravity-only reference arc is
  present. Lunaris propagates the manifest initial state with its production
  propagator (``core.propagator.propagate`` over ``DynamicsEngine``) under a
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
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.validation.gravity_reference.independent_field_oracle import coefficients_from_json
from lunaris.validation.gravity_reference.manifest import (
    STATUS_INCOMPLETE_CLASSES,
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
        "implementation_path": "lunaris.core.propagator.propagate",
        "dtype": "float64",
        "backend": "cpu",
    }


def _load_gravity_model(manifest: ResolvedTrajectoryManifest) -> Any:
    from lunaris.physics.spherical_harmonics import GravityModel

    gravity = manifest.payload["gravity"]
    degree = int(gravity["degree"])
    coeff_path = Path(gravity["coefficient_file"])
    if not coeff_path.is_absolute():
        from lunaris.validation.gravity_reference.source_hashes import find_repo_root

        coeff_path = find_repo_root(manifest.path) / coeff_path
    if coeff_path.suffix.lower() == ".json":
        _deg, r_ref, mu, c, s = coefficients_from_json(load_json(coeff_path))
        model = GravityModel.from_arrays(degree, r_ref, mu, c, s)
    else:
        model = GravityModel.from_file(str(coeff_path), requested_degree=degree)
    if int(model.degree_max) != degree:
        raise ValueError(f"Loaded degree {model.degree_max} != manifest degree {degree}.")
    return model


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
    from lunaris.core.propagator import propagate

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
    rel = step * 1e-6
    if t_out.shape[0] != epochs_s.shape[0] or not np.allclose(
        t_out - t_out[0], epochs_s - epochs_s[0], atol=max(rel, 1e-6), rtol=0.0
    ):
        # Reference epochs are not on the propagator's native grid; interpolate
        # each component with a cubic so the comparison stays epoch-aligned.
        aligned = np.empty((epochs_s.shape[0], 6), dtype=np.float64)
        rel_ref = epochs_s - epochs_s[0]
        rel_out = t_out - t_out[0]
        for j in range(6):
            aligned[:, j] = np.interp(rel_ref, rel_out, states[:, j])
        return aligned
    return states


def run_trajectory_validation(manifest_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Validate a gravity-only trajectory contract against an external reference."""
    manifest = load_trajectory_manifest(manifest_path)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
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
    model = _load_gravity_model(manifest)

    integration = payload.get("integration", {}) if isinstance(payload.get("integration"), dict) else {}
    rtol = float(integration.get("rtol", 1e-12))
    atol = float(integration.get("atol", 1e-15))

    y0 = np.asarray(payload["initial_state"]["state"], dtype=np.float64)
    lunaris_states = _propagate_lunaris(model, y0, epochs, rtol=rtol, atol=atol)

    metrics = compute_trajectory_metrics(ref_states, lunaris_states)
    metrics["lunaris_energy_drift"] = specific_energy_drift(
        lunaris_states, lambda r: model.potential_fixed(r)
    )
    metrics["reference_energy_drift"] = specific_energy_drift(
        ref_states, lambda r: model.potential_fixed(r)
    )

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
