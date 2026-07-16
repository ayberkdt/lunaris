"""PUBLIC_API.md's module table must match the real facade ``__all__`` lists.

Only facades with an exact, enumerable public surface are checked (the same
set tracked by ``docs/api_snapshot.json``); prose rows such as
``lunaris.common`` ("flat re-exports ...") are out of scope here.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "api_inventory.py"
PUBLIC_API_MD = REPO_ROOT / "docs" / "PUBLIC_API.md"

CHECKED_FACADES = (
    "lunaris",
    "lunaris.api",
    "lunaris.batch",
    "lunaris.core.propagation",
    "lunaris.surrogate.runtime",
)


def _load_tool():
    spec = importlib.util.spec_from_file_location("api_inventory", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _python_surface_section(doc_text: str) -> str:
    section = doc_text.split("## Stable Python Surface", 1)
    assert len(section) == 2, "PUBLIC_API.md lost its 'Stable Python Surface' section"
    return section[1].split("\n## ", 1)[0]


def _documented_symbols(section: str, module: str) -> set[str]:
    pattern = re.compile(rf"^\|\s*`{re.escape(module)}`\s*\|(.+)\|\s*$", re.MULTILINE)
    match = pattern.search(section)
    assert match is not None, f"PUBLIC_API.md has no module-table row for {module}"
    return set(re.findall(r"`([^`]+)`", match.group(1)))


def test_documented_module_rows_match_facade_all() -> None:
    tool = _load_tool()
    facade_all = tool.build_inventory()["facade_all"]
    section = _python_surface_section(PUBLIC_API_MD.read_text(encoding="utf-8"))

    problems: list[str] = []
    for module in CHECKED_FACADES:
        exported = facade_all.get(module)
        assert exported is not None, f"{module} lost its literal __all__"
        documented = _documented_symbols(section, module)
        missing_in_doc = sorted(set(exported) - documented)
        stale_in_doc = sorted(documented - set(exported))
        if missing_in_doc:
            problems.append(f"{module}: exported but undocumented: {missing_in_doc}")
        if stale_in_doc:
            problems.append(f"{module}: documented but not exported: {stale_in_doc}")
    assert problems == [], "\n".join(problems)
