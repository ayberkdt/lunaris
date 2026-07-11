#!/usr/bin/env python3
"""
Scan the mission-studio UI for layout spacing that falls off the 4px scale.

`SpacingTokens` defines the rhythm {0, 4, 6, 8, 12, 16, 20, 24, 32} (plus 2 as a
half-step for tight label/value pairs). Any integer literal passed to
``setSpacing`` / ``setContentsMargins`` / ``set{Horizontal,Vertical}Spacing`` that
is not on that scale — and not in ``ALLOWLIST`` — is reported so spacing drift is
visible in review.

Usage:
    python tools/ui/spacing_scan.py           # human report, exit 1 on drift
    python tools/ui/spacing_scan.py --quiet    # exit code only
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = REPO_ROOT / "src" / "lunaris" / "ui"

# On-scale values. 0 is a legitimate reset; 2 is a deliberate half-step.
SCALE = {0, 2, 4, 6, 8, 12, 16, 20, 24, 32}

SPACING_CALLS = {
    "setSpacing",
    "setContentsMargins",
    "setHorizontalSpacing",
    "setVerticalSpacing",
}

# Deliberate off-scale values, keyed by file (relative to UI_DIR). Each entry is
# ``{value: reason}``; a value here is exempt no matter how many times it occurs.
ALLOWLIST: dict[str, dict[int, str]] = {
    "launcher.py": {
        40: "splash hero horizontal margin — generous by design, not a data grid",
    },
    "pages/orbit_config_page.py": {
        1: "1px inner gap keeps a metric value tight under its label",
    },
}


def scan() -> dict[str, Counter]:
    offenders: dict[str, Counter] = {}
    for path in sorted(UI_DIR.rglob("*.py")):
        if "web" in path.parts or "node_modules" in path.parts:
            continue
        rel = path.relative_to(UI_DIR).as_posix()
        allowed = ALLOWLIST.get(rel, {})
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr in SPACING_CALLS):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                    if arg.value in SCALE or arg.value in allowed:
                        continue
                    offenders.setdefault(rel, Counter())[arg.value] += 1
    return offenders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quiet", action="store_true", help="Suppress the report.")
    args = parser.parse_args(argv)

    offenders = scan()
    if not offenders:
        if not args.quiet:
            print("spacing scan clean: all literals on the 4px scale "
                  "(or explicitly allow-listed)")
        return 0
    if not args.quiet:
        print("off-scale spacing literals (value: count):")
        for rel, counts in sorted(offenders.items()):
            pretty = ", ".join(f"{v}px x{c}" for v, c in sorted(counts.items()))
            print(f"  {rel}: {pretty}")
        print("\nfix to the nearest SpacingTokens step, or add a justified "
              "entry to ALLOWLIST in this script.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
