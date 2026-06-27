"""Optional ST-LRPS runtime smoke example.

This example is intentionally optional. Classical Lunaris propagation does not
need an ST-LRPS artifact. Pass a trained model directory only when you have one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ST-LRPS runtime smoke example.")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Trained ST-LRPS run directory. If omitted, the example exits without error.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model_dir is None:
        print("No --model-dir supplied; ST-LRPS is optional, so nothing to run.")
        print(f"Example artifacts normally live outside source, e.g. {REPO_ROOT / 'outputs' / 'training'}")
        return 0

    import numpy as np

    from lunaris.common.constants import R_MOON
    from lunaris.surrogate.runtime_adapter import SurrogateGravityModel

    model = SurrogateGravityModel.from_model_dir(args.model_dir)
    r_fixed_m = np.asarray([float(R_MOON) + 100_000.0, 0.0, 0.0], dtype=float)
    accel = model.acceleration_fixed(r_fixed_m)
    print("fixed-frame acceleration [m/s^2]:", accel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
