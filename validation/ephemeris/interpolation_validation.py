"""Validate schema-v2 Hermite ephemerides against direct SPICE states.

This is an evidence tool, not a unit test. It reports off-node position errors
for the canonical Hermite interpolant and the position-only Catmull-Rom
compatibility path, plus a matched DOP853 function-evaluation comparison.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import spiceypy as spice
from scipy.integrate import solve_ivp

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.physics.ephemeris import build_tables, interp_vec3_hermite, interp_vec3_safe

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _portable_path_hint(path: Path) -> str:
    """Describe an input without publishing workstation-specific absolute paths."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def _third_body(r_sc: np.ndarray, r_body: np.ndarray, mu: float) -> np.ndarray:
    delta = r_body - r_sc
    return mu * (delta / np.linalg.norm(delta) ** 3 - r_body / np.linalg.norm(r_body) ** 3)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-dir", type=Path, default=Path("data/ephemeris_models"))
    parser.add_argument("--start-utc", default="2026-01-01T00:00:00")
    parser.add_argument("--duration-s", type=float, default=86_400.0)
    parser.add_argument("--grid-step-s", type=float, default=3600.0)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output", type=Path)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    kernel_dir = args.kernel_dir.resolve()
    kernels = (
        kernel_dir / "naif0012.tls",
        kernel_dir / "de440.bsp",
        kernel_dir / "moon_pa_de440_200625.bpc",
        kernel_dir / "moon_de440_250416.tf",
    )
    missing = [str(path) for path in kernels if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing SPICE kernels: " + ", ".join(missing))
    tables = build_tables(
        start_utc=args.start_utc,
        duration_s=args.duration_s,
        output_dt_s=args.grid_step_s,
        kernels=[str(path) for path in kernels],
        clear_kernels_after=True,
    )
    rng = np.random.default_rng(args.seed)
    query_t = np.sort(rng.uniform(0.0, args.duration_s, size=args.queries))
    errors: dict[str, dict[str, list[float]]] = {
        "earth": {"catmull_rom_m": [], "hermite_m": []},
        "sun": {"catmull_rom_m": [], "hermite_m": []},
    }
    spice.kclear()
    try:
        for path in kernels:
            spice.furnsh(str(path))
        for t_s in query_t:
            for body, p_tab, v_tab in (
                ("earth", tables.r_earth_tab_m, tables.v_earth_tab_m_s),
                ("sun", tables.r_sun_tab_m, tables.v_sun_tab_m_s),
            ):
                state_km, _ = spice.spkezr(body.upper(), tables.et0 + float(t_s), "J2000", "NONE", "MOON")
                truth = np.asarray(state_km[:3], dtype=np.float64) * 1000.0
                cat = np.asarray(interp_vec3_safe(float(t_s), tables.dt_s, p_tab))
                herm = np.asarray(interp_vec3_hermite(float(t_s), tables.dt_s, p_tab, v_tab))
                errors[body]["catmull_rom_m"].append(float(np.linalg.norm(cat - truth)))
                errors[body]["hermite_m"].append(float(np.linalg.norm(herm - truth)))
    finally:
        spice.kclear()

    def rhs(kind: str):
        def evaluate(t_s: float, state: np.ndarray) -> np.ndarray:
            r = state[:3]
            if kind == "hermite":
                earth = np.asarray(interp_vec3_hermite(t_s, tables.dt_s, tables.r_earth_tab_m, tables.v_earth_tab_m_s))
                sun = np.asarray(interp_vec3_hermite(t_s, tables.dt_s, tables.r_sun_tab_m, tables.v_sun_tab_m_s))
            else:
                earth = np.asarray(interp_vec3_safe(t_s, tables.dt_s, tables.r_earth_tab_m))
                sun = np.asarray(interp_vec3_safe(t_s, tables.dt_s, tables.r_sun_tab_m))
            accel = -MU_MOON * r / np.linalg.norm(r) ** 3
            accel += _third_body(r, earth, tables.mu_earth_m3s2)
            accel += _third_body(r, sun, tables.mu_sun_m3s2)
            return np.concatenate((state[3:6], accel))
        return evaluate

    radius = R_MOON + 100_000.0
    y0 = np.asarray((radius, 0.0, 0.0, 0.0, np.sqrt(MU_MOON / radius), 0.0))
    solutions = {
        kind: solve_ivp(
            rhs(kind),
            (0.0, float(args.duration_s)),
            y0,
            method="DOP853",
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        for kind in ("catmull_rom", "hermite")
    }
    summary = {
        body: {
            kind: {
                "max": float(np.max(values)),
                "rms": float(np.sqrt(np.mean(np.square(values)))),
            }
            for kind, values in methods.items()
        }
        for body, methods in errors.items()
    }
    payload: dict[str, object] = {
        "schema_version": "lunaris_ephemeris_interpolation_validation_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "configuration": {
            "kernel_dir_hint": _portable_path_hint(kernel_dir),
            "start_utc": str(args.start_utc),
            "duration_s": float(args.duration_s),
            "grid_step_s": float(args.grid_step_s),
            "queries": int(args.queries),
            "seed": int(args.seed),
            "output_hint": (
                None if args.output is None else _portable_path_hint(args.output)
            ),
        },
        "ephemeris_contract": {
            "kernel_provenance": [
                {
                    "name": str(record.get("name", "")),
                    "kind": str(record.get("kind", "UNKNOWN")),
                    "sha256": record.get("sha256"),
                    "path_hint": (
                        Path(_portable_path_hint(kernel_dir))
                        / str(record.get("name", ""))
                    ).as_posix(),
                }
                for record in tables.kernel_provenance
            ],
            "schema_version": tables.schema_version,
            "interpolation_kind": tables.interpolation_kind,
            "aberration_correction": tables.aberration_correction,
        },
        "off_node_position_error_m": summary,
        "dop853": {
            kind: {"success": bool(solution.success), "nfev": int(solution.nfev)}
            for kind, solution in solutions.items()
        },
        "dop853_final_position_difference_m": float(
            np.linalg.norm(solutions["hermite"].y[:3, -1] - solutions["catmull_rom"].y[:3, -1])
        ),
        "claim_scope": "interpolation_and_step_response_for_this_configuration_only",
    }
    return payload


def main() -> int:
    args = _parser().parse_args()
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
