"""Forbid new cross-unit imports of single-underscore symbols in src/lunaris.

A single leading underscore means private to its defining module and its own
subsystem (docs/PUBLIC_API.md, "Naming And Boundary Policy"). Units are
top-level packages, with each ST-LRPS subpackage counted separately —
matching the import-linter contracts. The allowlist below is intentionally
empty: the 17 historical offenders were renamed, moved, or given facades in
the 2026-07 API-boundary cleanup. If this test fails, give the helper a
public name in its defining module (adding it to ``__all__``) or move it to
a shared home — do not extend the allowlist without a written reason.

White-box *tests* importing internals are out of scope here on purpose:
this gate protects production boundaries only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "api_inventory.py"

# "importer <- source :: _symbol" rows tolerated for a documented reason.
ALLOWED_CROSS_UNIT_UNDERSCORE_IMPORTS: frozenset[str] = frozenset()


def _load_tool():
    spec = importlib.util.spec_from_file_location("api_inventory", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_no_new_cross_unit_underscore_imports() -> None:
    tool = _load_tool()
    rows = set(tool.cross_unit_underscore_imports())
    unexpected = sorted(rows - ALLOWED_CROSS_UNIT_UNDERSCORE_IMPORTS)
    stale_allow = sorted(ALLOWED_CROSS_UNIT_UNDERSCORE_IMPORTS - rows)
    assert unexpected == [], (
        "New cross-unit imports of private (_-prefixed) symbols:\n  "
        + "\n  ".join(unexpected)
        + "\nGive the helper a public name in its defining module or move it "
        "to a shared home (docs/PUBLIC_API.md, Naming And Boundary Policy)."
    )
    assert stale_allow == [], f"Allowlist entries no longer needed: {stale_allow}"
