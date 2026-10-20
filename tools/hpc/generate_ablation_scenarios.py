#!/usr/bin/env python3
"""Write the HPC ablation scenario file from the ablation registry.

The A0..A6 ladder has one source, ``ABLATION_REGISTRY`` in
``lunaris.surrogate.st_lrps.evaluation.ablation``. This tool renders it into
``hpc/scenarios/<HPC_SCENARIO_FILE>`` (one line per arm and seed);
``tests/test_hpc_scenarios.py`` fails when the committed file differs.

    python tools/hpc/generate_ablation_scenarios.py           # write
    python tools/hpc/generate_ablation_scenarios.py --check   # exit 1 if stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lunaris.surrogate.st_lrps.evaluation.ablation import (
    DEFAULT_SEEDS,
    HPC_SCENARIO_FILE,
    render_hpc_scenarios,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS), help="Training seeds per arm.")
    ap.add_argument("--split-seed", type=int, default=None, help="Shared split seed (default: first seed).")
    ap.add_argument("--check", action="store_true", help="Do not write; exit 1 when the file is stale.")
    args = ap.parse_args(argv)

    path = _REPO_ROOT / "hpc" / "scenarios" / HPC_SCENARIO_FILE
    text = render_hpc_scenarios(args.seeds, args.split_seed)
    current = path.read_text(encoding="utf-8") if path.is_file() else None
    if args.check:
        if current != text:
            print(f"{path} is stale; run tools/hpc/generate_ablation_scenarios.py", file=sys.stderr)
            return 1
        print(f"{path} is up to date.")
        return 0
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {path} ({text.count(chr(10)) - 5} scenario lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
