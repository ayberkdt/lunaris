"""PUBLIC_API.md and the machine-readable module manifest must stay aligned.

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
    section = doc_text.split("## Python Surface Tiers", 1)
    assert len(section) == 2, "PUBLIC_API.md lost its 'Python Surface Tiers' section"
    return section[1].split("\n## ", 1)[0]


def _documented_symbols(section: str, module: str) -> set[str]:
    pattern = re.compile(rf"^\|\s*`{re.escape(module)}`\s*\|(.+)\|\s*$", re.MULTILINE)
    match = pattern.search(section)
    assert match is not None, f"PUBLIC_API.md has no module-table row for {module}"
    return set(re.findall(r"`([^`]+)`", match.group(1)))


def test_documented_module_rows_match_facade_all() -> None:
    tool = _load_tool()
    module_all = tool.build_inventory()["module_all"]
    section = _python_surface_section(PUBLIC_API_MD.read_text(encoding="utf-8"))

    problems: list[str] = []
    for module in CHECKED_FACADES:
        exported = module_all.get(module)
        assert exported is not None, f"{module} lost its literal __all__"
        documented = _documented_symbols(section, module)
        missing_in_doc = sorted(set(exported) - documented)
        stale_in_doc = sorted(documented - set(exported))
        if missing_in_doc:
            problems.append(f"{module}: exported but undocumented: {missing_in_doc}")
        if stale_in_doc:
            problems.append(f"{module}: documented but not exported: {stale_in_doc}")
    assert problems == [], "\n".join(problems)


def test_documented_static_all_modules_are_in_manifest() -> None:
    """A documented direct-import module with ``__all__`` cannot evade snapshots."""
    tool = _load_tool()
    section = _python_surface_section(PUBLIC_API_MD.read_text(encoding="utf-8"))
    documented = set(
        re.findall(r"^\|\s*`(lunaris(?:\.[^`]+)*)`\s*\|", section, re.MULTILINE)
    )
    tracked = {entry["module"] for entry in tool.load_public_api_manifest()}
    missing: list[str] = []
    for module in sorted(documented):
        path = tool.module_path_for(module)
        if path.is_file() and tool.extract_all(path) is not None and module not in tracked:
            missing.append(module)
    assert missing == [], (
        "documented modules with literal __all__ missing from "
        f"docs/public_api_manifest.json: {missing}"
    )


def test_manifest_modules_are_documented_with_their_tier() -> None:
    doc = PUBLIC_API_MD.read_text(encoding="utf-8")
    for entry in _load_tool().load_public_api_manifest():
        assert f"`{entry['module']}`" in doc, f"manifest module is undocumented: {entry}"
        tier_anchor = entry["tier"].replace("_", "-")
        assert tier_anchor in doc, f"PUBLIC_API.md is missing tier anchor {tier_anchor!r}"
