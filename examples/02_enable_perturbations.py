"""Enable common perturbations without requiring surface rasters."""

from __future__ import annotations

from _common import (
    build_classic_engine,
    enable_demo_perturbations,
    propagate_config,
    save_altitude_plot,
    short_config,
)


def main() -> int:
    cfg = short_config("enable_perturbations", hours=2.0, output_dt_s=120.0, degree=20)
    cfg = enable_demo_perturbations(cfg)
    engine = build_classic_engine(cfg)
    result = propagate_config(cfg, engine)
    plot_path = save_altitude_plot(result, cfg.output.ensure_out_dir(), title="Perturbed propagation")
    print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
