"""UI copy style lint: no marketing tone in user-facing strings.

The design system (docs/UI_DESIGN_SYSTEM.md, redesign plan M7) requires UI
text to state a fact or an action — no value promises, no marketing adverbs.
This test walks every string literal in the Mission Studio UI sources
(docstrings and comments exempt) and fails on the banned vocabulary.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

UI_ROOT = Path(__file__).resolve().parents[1] / "src" / "lunaris" / "ui"

# Directories that are not Python UI sources (bundled web preview, caches).
EXCLUDED_PARTS = {"web", "__pycache__"}

BANNED = re.compile(
    r"\b(prefer|automatically|seamless(?:ly)?|effortless(?:ly)?|"
    r"at a glance|powerful|premium|beautiful)\b",
    re.IGNORECASE,
)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Return ids of Constant nodes that are docstrings."""
    doc_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    doc_ids.add(id(value))
    return doc_ids


def _violations_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    doc_ids = _docstring_nodes(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in doc_ids
        ):
            match = BANNED.search(node.value)
            if match:
                found.append(
                    f"{path.relative_to(UI_ROOT.parents[2])}:{node.lineno}: "
                    f"banned word {match.group(0)!r} in {node.value[:70]!r}"
                )
    return found


def test_ui_strings_have_no_marketing_tone():
    assert UI_ROOT.is_dir(), f"missing UI root: {UI_ROOT}"
    violations: list[str] = []
    for path in sorted(UI_ROOT.rglob("*.py")):
        if EXCLUDED_PARTS.intersection(path.parts):
            continue
        violations.extend(_violations_in_file(path))
    assert not violations, "Marketing-tone strings in UI copy:\n" + "\n".join(violations)
