"""Shared helpers for the Lunaris examples.

Run examples from the repository root after installing the package:

    python -m pip install -e .
    python examples/01_basic_propagation.py
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lunaris.common.constants import R_MOON
from lunaris.core.config import SimConfig, load_default_config
from lunaris.core.dynamics import DynamicsEngine
from lunaris.core.propagation.propagator import propagate
from lunaris.physics.ephemeris import EphemerisManager
from lunaris.physics.spherical_harmonics import GravityModel


def example_output_dir(name: str) -> Path:
    out_dir = REPO_ROOT / "outputs" / "examples" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def short_config(
    name: str,
    *,
    hours: float = 2.0,
    output_dt_s: float = 120.0,
    degree: int = 20,
) -> SimConfig:
    cfg = load_default_config()
    cfg = replace(
        cfg,
        time=replace(
            cfg.time,
            duration_s=float(hours) * 3600.0,
            output_dt_s=float(output_dt_s),
        ),
        gravity=replace(cfg.gravity, degree=int(degree)),
        output=replace(
            cfg.output,
            out_dir=example_output_dir(name),
            make_3d_plots=False,
        ),
        propagator=replace(
            cfg.propagator,
            verbose=False,
            compute_2body_baseline=False,
        ),
    )
    cfg.validate()
    return cfg


def enable_demo_perturbations(cfg: SimConfig) -> SimConfig:
    """Enable perturbations that do not require surface rasters or ST-LRPS artifacts."""
    flags = replace(
        cfg.flags,
        enable_3rd_body_sun=True,
        enable_3rd_body_earth=True,
        enable_srp=True,
        enable_tides_k2=True,
        enable_relativity_1pn=True,
    )
    cfg = replace(cfg, flags=flags)
    cfg.validate()
    return cfg


def build_classic_engine(cfg: SimConfig) -> DynamicsEngine:
    gravity = GravityModel.from_file(
        path=str(cfg.gravity.file_path),
        requested_degree=cfg.gravity.degree,
    )
    ephem = EphemerisManager.from_time_and_spice(
        cfg.time,
        cfg.spice,
        auto_fix_kernel_paths=True,
        need_moon_fixed_rotation=True,
    )
    return DynamicsEngine(
        sc_props=cfg.spacecraft,
        flags=cfg.flags,
        gravity_model=gravity if cfg.flags.enable_sh else None,
        gravity_adaptive=cfg.gravity.adaptive,
        ephem_manager=ephem,
        earth_j2=cfg.earth_j2,
        srp=cfg.srp,
        thermal=cfg.thermal,
        albedo=cfg.albedo,
        solid_tides=cfg.solid_tides,
    )


def propagate_config(cfg: SimConfig, engine: DynamicsEngine, y0: np.ndarray | None = None):
    if y0 is None:
        y0 = cfg.initial_state.to_array()
    return propagate(
        dynamics=engine,
        y0=y0,
        cfg=cfg.propagator,
        time_cfg=cfg.time,
    )


def altitude_km(result) -> np.ndarray:
    r_norm = np.linalg.norm(result.y[:, 0:3], axis=1)
    return (r_norm - float(R_MOON)) / 1000.0


def save_altitude_plot(result, out_dir: Path, *, title: str = "Altitude") -> Path:
    alt = altitude_km(result)
    t_hr = np.asarray(result.t, dtype=float) / 3600.0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t_hr, alt, lw=1.8)
    ax.set_title(title)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Altitude [km]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "altitude.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_multi_altitude_plot(series: Iterable[tuple[str, object]], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, result in series:
        ax.plot(np.asarray(result.t) / 3600.0, altitude_km(result), lw=1.6, label=label)
    ax.set_title("Batch altitude histories")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Altitude [km]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "batch_altitudes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
