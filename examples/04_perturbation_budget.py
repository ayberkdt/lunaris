"""Compute a compact acceleration breakdown at the initial state."""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import build_classic_engine, enable_demo_perturbations, short_config


def main() -> int:
    cfg = short_config("perturbation_budget", hours=1.0, output_dt_s=300.0, degree=20)
    cfg = enable_demo_perturbations(cfg)
    engine = build_classic_engine(cfg)
    out_dir = cfg.output.ensure_out_dir()

    breakdown = engine.get_acceleration_breakdown(0.0, cfg.initial_state.to_array())
    rows = sorted(breakdown.items(), key=lambda item: item[1], reverse=True)

    csv_path = out_dir / "acceleration_breakdown.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("component", "accel_norm_m_s2"))
        writer.writerows(rows)

    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(labels, values)
    ax.set_xscale("log")
    ax.set_xlabel("Acceleration norm [m/s^2]")
    ax.set_title("Perturbation budget at t=0")
    ax.invert_yaxis()
    fig.tight_layout()
    plot_path = out_dir / "acceleration_breakdown.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
