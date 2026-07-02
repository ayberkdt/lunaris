from __future__ import annotations

import ast
import importlib
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "lunaris"


def _python_files(package: str) -> Iterable[Path]:
    root = SRC_ROOT / package
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts or "node_modules" in path.parts:
            continue
        yield path


def _module_parts_for_path(path: Path) -> tuple[str, ...]:
    rel = path.relative_to(SRC_ROOT).with_suffix("")
    parts = ("lunaris", *rel.parts)
    if parts[-1] == "__init__":
        return parts[:-1]
    return parts


def _resolve_import_from(path: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module

    module_parts = _module_parts_for_path(path)
    package_parts = module_parts if path.name == "__init__.py" else module_parts[:-1]
    keep = len(package_parts) - (node.level - 1)
    if keep < 1:
        return node.module

    base = package_parts[:keep]
    if node.module:
        base = (*base, *node.module.split("."))
    return ".".join(base)


def _is_type_checking_guard(expr: ast.AST) -> bool:
    if isinstance(expr, ast.Name):
        return expr.id == "TYPE_CHECKING"
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        return expr.value.id == "typing" and expr.attr == "TYPE_CHECKING"
    return False


def _iter_import_nodes(node: ast.AST) -> Iterable[ast.Import | ast.ImportFrom]:
    if isinstance(node, ast.If) and _is_type_checking_guard(node.test):
        return
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        yield node
        return
    for child in ast.iter_child_nodes(node):
        yield from _iter_import_nodes(child)


def _imports_for(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    imports: set[str] = set()
    for node in _iter_import_nodes(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        else:
            resolved = _resolve_import_from(path, node)
            if resolved:
                imports.add(resolved)
    return imports


def _matches_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _violations(package: str, forbidden_prefixes: Iterable[str]) -> list[tuple[str, str]]:
    blocked = tuple(forbidden_prefixes)
    found: list[tuple[str, str]] = []
    for path in _python_files(package):
        for imported in sorted(_imports_for(path)):
            if _matches_prefix(imported, blocked):
                found.append((path.relative_to(REPO_ROOT).as_posix(), imported))
    return found


def test_physics_has_no_upward_runtime_imports() -> None:
    forbidden = (
        "lunaris.core",
        "lunaris.ui",
        "lunaris.cli",
        "lunaris.analysis",
        "lunaris.surrogate",
    )

    assert _violations("physics", forbidden) == []


def test_common_stays_dependency_light() -> None:
    forbidden = ("torch", "PySide6", "PyQt6", "matplotlib", "h5py", "scipy")

    assert _violations("common", forbidden) == []


def test_core_avoids_presentation_and_ml_pipeline_imports() -> None:
    forbidden = (
        "lunaris.ui",
        "lunaris.visualization",
        "lunaris.analysis.reporting",
        "lunaris.analysis.ensemble.plotting",
        "lunaris.surrogate.st_lrps.training",
        "lunaris.surrogate.st_lrps.evaluation",
        "lunaris.surrogate.st_lrps.ui",
        "lunaris.surrogate.st_lrps.data",
    )

    assert _violations("core", forbidden) == []


def test_refactor_public_modules_import_smoke() -> None:
    modules = (
        "lunaris.batch",
        "lunaris.batch.backend_policy",
        "lunaris.batch.memory_policy",
        "lunaris.batch.provenance",
        "lunaris.batch.requirements",
        "lunaris.batch.sampling",
        "lunaris.batch.storage",
        "lunaris.cli.batch",
        "lunaris.cli.main",
        "lunaris.cli.options",
        "lunaris.cli.run",
        "lunaris.cli.summary",
        "lunaris.core.dynamics",
        "lunaris.core.propagation",
        "lunaris.core.propagation.propagator",
        "lunaris.surrogate.runtime",
    )

    for module_name in modules:
        importlib.import_module(module_name)
