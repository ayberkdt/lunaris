#!/usr/bin/env python3
"""Scan Lunaris PySide6 UI code for hard-coded colors that bypass the theme.

The Lunar Graphite system (``docs/UI_THEME.md``) requires every color to route
through ``THEME`` / ``ORBIT_THEME`` / ``LOG_COLORS`` in
``lunaris.ui.core.ui_commons`` (translucency via ``with_alpha(...)``). Raw hex or
``rgba(...)`` literals in page/widget code are theme drift. This scanner reports
them. Pure standard library.

The token-definition modules themselves legitimately contain literals and are
excluded by default: ``ui_commons`` (any path) and ``ui/theme/`` (the stylesheet
generator). ``ORBIT_THEME``/3D color tuples live in those modules.

Usage
-----
    python scan_hardcoded_colors.py [PATH ...] [--exclude SUBSTR ...]
Default PATH is ``src/lunaris/ui``.

Exit codes
----------
* ``0`` — no hard-coded colors found;
* ``1`` — at least one found;
* ``2`` — bad input (no files).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEX_RE = re.compile(r"#[0-9A-Fa-f]{6}\b|#[0-9A-Fa-f]{3}\b")
RGBA_RE = re.compile(r"\brgba?\(")  # rgb( / rgba(
DEFAULT_EXCLUDES = ("ui_commons", "ui/theme/", "ui\\theme\\", "__pycache__")


def is_excluded(path: Path, excludes: tuple[str, ...]) -> bool:
    p = str(path).replace("\\", "/")
    return any(ex.replace("\\", "/") in p for ex in excludes)


def scan_file(path: Path) -> list[tuple[int, str]]:
    findings: list[tuple[int, str]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return findings
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # whole-line comment
        if HEX_RE.search(line) or RGBA_RE.search(line):
            findings.append((i, stripped[:120]))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan for hard-coded UI colors that bypass the Lunar Graphite theme.",
        epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="*", default=["src/lunaris/ui"])
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDES))
    args = parser.parse_args(argv)

    excludes = tuple(args.exclude)
    files: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
    files = [f for f in files if not is_excluded(f, excludes)]

    if not files:
        print("error: no Python files to scan", file=sys.stderr)
        return 2

    total = 0
    for f in files:
        findings = scan_file(f)
        for lineno, snippet in findings:
            total += 1
            print(f"{f}:{lineno}: hard-coded color -> {snippet}")

    print(f"\nScanned {len(files)} file(s); {total} hard-coded color literal(s) found.")
    if total:
        print("Route colors through THEME/ORBIT_THEME (with_alpha for translucency).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
