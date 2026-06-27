"""Small batch sweep over circular orbit altitudes."""

from __future__ import annotations

import csv
import math

import numpy as np
from _common import build_classic_engine, propagate_config, save_multi_altitude_plot, short_config

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.core.state import create_state_from_keplerian


def circular_state(alt_km: float) -> np.ndarray:
    state = create_state_from_keplerian(
        semi_major_axis=float(R_MOON) + float(alt_km) * 1000.0,
        eccentricity=0.0,
        inclination=math.radians(30.0),
        raan=0.0,
        argp=0.0,
        true_anomaly=0.0,
        mu=float(MU_MOON),
    )
    return np.asarray(state.y, dtype=float)


def main() -> int:
    cfg = short_config("batch_run", hours=1.0, output_dt_s=120.0, degree=16)
    engine = build_classic_engine(cfg)
    out_dir = cfg.output.ensure_out_dir()

    rows: list[dict[str, float]] = []
    series = []
    for alt_km in (50.0, 100.0, 300.0):
        result = propagate_config(cfg, engine, y0=circular_state(alt_km))
        alt_hist = np.linalg.norm(result.y[:, 0:3], axis=1) / 1000.0 - float(R_MOON) / 1000.0
        rows.append({
            "initial_alt_km": alt_km,
            "final_alt_km": float(alt_hist[-1]),
            "min_alt_km": float(np.min(alt_hist)),
            "max_alt_km": float(np.max(alt_hist)),
        })
        series.append((f"{alt_km:.0f} km", result))

    csv_path = out_dir / "batch_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = save_multi_altitude_plot(series, out_dir)
    print(f"wrote {csv_path}")
    print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
