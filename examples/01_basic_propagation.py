"""Basic single-orbit propagation with classical spherical-harmonic gravity."""

from __future__ import annotations

from _common import build_classic_engine, propagate_config, save_altitude_plot, short_config


def main() -> int:
    cfg = short_config("basic_propagation", hours=2.0, output_dt_s=120.0, degree=20)
    engine = build_classic_engine(cfg)
    result = propagate_config(cfg, engine)
    plot_path = save_altitude_plot(result, cfg.output.ensure_out_dir(), title="Basic propagation")
    print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
