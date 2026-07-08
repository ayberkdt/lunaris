"""Human-runnable smoke check for the Lunaris default configuration.

Relocated out of ``lunaris.core.config`` so the core module stays a silent,
import-safe library. Run it directly to verify that the local environment can
assemble the default :class:`~lunaris.core.config.SimConfig` (requires the SPICE
kernels and gravity-model assets to be present):

    python tools/check_config.py
"""

from __future__ import annotations

from lunaris.common.constants import DAY_S
from lunaris.core.config import load_default_config


def main() -> int:
    print("\n" + "=" * 60)
    print("LUNARIS CONFIGURATION CHECK")
    print("=" * 60 + "\n")

    try:
        test_cfg = load_default_config()
        print("✅ [PASS] load_default_config() successful.")
    except Exception as e:
        print(f"❌ [FAIL] Could not load config: {e}")
        raise

    print("\n📂 [PATHS]")
    print(f"   Gravity Model : {test_cfg.gravity.file_path}")
    print(f"   SPICE Kernels : {len(test_cfg.spice.kernels)} selected.")

    print("\n⚙️  [PHYSICS FLAGS]")
    print(f"   Spherical Harmonics : {test_cfg.flags.enable_sh}")
    print(f"   3rd Body (Sun/Earth): {test_cfg.flags.enable_3rd_body_sun} / {test_cfg.flags.enable_3rd_body_earth}")
    print(f"   Earth J2            : {test_cfg.flags.enable_earth_j2}")
    print(f"   Tides (K2/K3)       : {test_cfg.flags.enable_tides_k2} / {test_cfg.flags.enable_tides_k3}")
    print(f"   Relativity (1PN)    : {test_cfg.flags.enable_relativity_1pn}")

    print("\n🚀 [MISSION PARAMETERS]")
    print(f"   Spacecraft Mass : {test_cfg.spacecraft.mass_kg} kg")
    print(f"   Total Duration  : {test_cfg.total_seconds / DAY_S:.2f} days")
    dt_display = test_cfg.time.output_dt_s
    dt_str = f"{dt_display}s" if dt_display is not None else "Auto (Variable)"
    print(f"   Time Step (Out) : {dt_str}")
    print(f"   Propagator      : {test_cfg.propagator.method} (Tol: {test_cfg.propagator.rtol})")

    print("\n" + "=" * 60)
    print("✅ CONFIGURATION INTEGRITY CHECK COMPLETE.")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
