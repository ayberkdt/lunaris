#!/usr/bin/env python3
"""WCAG 2.2 contrast-ratio checker for Lunaris UI color pairs.

Computes the WCAG relative-luminance contrast ratio between a foreground and a
background color and reports pass/fail against the requested conformance level.
Pure standard library.

Thresholds (WCAG 2.2, SC 1.4.3 / 1.4.6 / 1.4.11)
------------------------------------------------
* normal text  : AA 4.5:1, AAA 7:1
* large text   : AA 3:1,   AAA 4.5:1   (>= 18pt or 14pt bold)
* non-text UI  : 3:1       (component boundaries, focus rings, chart marks)

Usage
-----
    python contrast_check.py <fg_hex> <bg_hex> [--level aa|aaa]
                             [--kind normal|large|nontext]
Colors accept ``#6AA9FF`` or ``6AA9FF`` (3- or 6-digit hex).

Exit codes
----------
* ``0`` — pair meets the requested threshold;
* ``1`` — pair fails;
* ``2`` — bad input.
"""
from __future__ import annotations

import argparse
import sys

REQUIRED = {
    ("aa", "normal"): 4.5,
    ("aa", "large"): 3.0,
    ("aa", "nontext"): 3.0,
    ("aaa", "normal"): 7.0,
    ("aaa", "large"): 4.5,
    ("aaa", "nontext"): 3.0,  # AAA does not relax non-text contrast
}


def parse_hex(value: str) -> tuple[int, int, int]:
    s = value.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6 or any(c not in "0123456789abcdefABCDEF" for c in s):
        raise ValueError(f"invalid hex color: {value!r}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _linearize(channel: float) -> float:
    c = channel / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = (_linearize(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg: tuple[int, int, int], bg: tuple[int, int, int]) -> float:
    l1, l2 = relative_luminance(fg), relative_luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="WCAG 2.2 contrast-ratio checker", epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("fg", help="foreground color, e.g. 6AA9FF or #6AA9FF")
    parser.add_argument("bg", help="background color, e.g. 0E1116")
    parser.add_argument("--level", choices=["aa", "aaa"], default="aa")
    parser.add_argument("--kind", choices=["normal", "large", "nontext"], default="normal")
    args = parser.parse_args(argv)

    try:
        fg = parse_hex(args.fg)
        bg = parse_hex(args.bg)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    ratio = contrast_ratio(fg, bg)
    required = REQUIRED[(args.level, args.kind)]
    ok = ratio >= required
    verdict = "PASS" if ok else "FAIL"
    print(
        f"{verdict}  {args.fg} on {args.bg}: {ratio:.2f}:1 "
        f"(needs {required:.1f}:1 for {args.level.upper()} {args.kind})"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
