"""Keep docs/api_snapshot.json in sync with the actual facade surfaces.

The snapshot is the machine-readable public-API inventory (facade ``__all__``
lists plus every cross-unit single-underscore import inside ``src/lunaris``).
When this test fails, the public surface changed: regenerate the snapshot with
``python tools/api_inventory.py --write`` and review the diff in the same PR —
the diff *is* the API review artifact.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "api_inventory.py"
SNAPSHOT_PATH = REPO_ROOT / "docs" / "api_snapshot.json"


def _load_tool():
    spec = importlib.util.spec_from_file_location("api_inventory", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_api_snapshot_is_current() -> None:
    tool = _load_tool()
    live = tool.build_inventory()
    committed = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert committed == live, (
        "docs/api_snapshot.json is stale. Run `python tools/api_inventory.py "
        "--write` and review the resulting diff as part of the change."
    )


def test_snapshot_facades_have_static_all() -> None:
    """Every tracked facade must keep a literal, AST-readable ``__all__``."""
    tool = _load_tool()
    inventory = tool.build_inventory()
    missing = [
        module
        for module, exported in inventory["facade_all"].items()
        if exported is None
    ]
    assert missing == [], f"facades lost their literal __all__: {missing}"
