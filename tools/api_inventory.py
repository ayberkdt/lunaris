"""Machine-readable public-API inventory for Lunaris.

Computes, without importing ``lunaris`` (pure AST, safe on machines missing
optional dependencies):

1. The ``__all__`` surface of every module named in the machine-readable
   ``docs/public_api_manifest.json``.
2. Every cross-unit access to a single-underscore module or symbol inside
   ``src/lunaris`` (``from X import _y``, ``import X._y``, or
   ``import X as alias; alias._y``), where a *unit* is a top-level subsystem
   package except that ST-LRPS subpackages each count as their own unit.

Usage:
    python tools/api_inventory.py            # print inventory JSON to stdout
    python tools/api_inventory.py --check    # exit 1 if the snapshot is stale
    python tools/api_inventory.py --write    # regenerate the snapshot
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SNAPSHOT_PATH = REPO_ROOT / "docs" / "api_snapshot.json"
MANIFEST_PATH = REPO_ROOT / "docs" / "public_api_manifest.json"
SCHEMA = "lunaris_api_snapshot_v2"
MANIFEST_SCHEMA = "lunaris_public_api_manifest_v1"
PUBLIC_API_TIERS = frozenset(
    {"user_stable", "documented_provisional", "cross_subsystem_internal"}
)

_ST_LRPS_PREFIX = "lunaris.surrogate.st_lrps"


def module_name_for(path: Path) -> str:
    """Map a file under ``src/`` to its dotted module name."""
    rel = path.relative_to(SRC_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def module_path_for(module: str) -> Path:
    """Map a dotted module name to its file (package ``__init__`` preferred)."""
    base = SRC_ROOT / Path(*module.split("."))
    init = base / "__init__.py"
    if init.is_file():
        return init
    return base.with_suffix(".py")


def unit_of(module: str) -> str:
    """Boundary unit for cross-unit accounting (see module docstring)."""
    if module == _ST_LRPS_PREFIX or module.startswith(_ST_LRPS_PREFIX + "."):
        parts = module.split(".")
        return ".".join(parts[:4])
    parts = module.split(".")
    return ".".join(parts[:2])


def extract_all(path: Path) -> list[str] | None:
    """Return the literal ``__all__`` of a module, or None if absent/dynamic."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    try:
                        value = ast.literal_eval(node.value)
                    except ValueError:
                        return None
                    return sorted(str(name) for name in value)
    return None


def load_public_api_manifest() -> tuple[dict[str, str], ...]:
    """Load and validate the module inventory that drives API snapshots."""
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if raw.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"{MANIFEST_PATH.relative_to(REPO_ROOT)} has unsupported schema "
            f"{raw.get('schema')!r}"
        )
    entries = raw.get("modules")
    if not isinstance(entries, list):
        raise ValueError("public API manifest 'modules' must be a list")

    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("public API manifest entries must be objects")
        module = entry.get("module")
        tier = entry.get("tier")
        if not isinstance(module, str) or not module.startswith("lunaris"):
            raise ValueError(f"invalid public API module: {module!r}")
        if tier not in PUBLIC_API_TIERS:
            raise ValueError(f"invalid public API tier for {module}: {tier!r}")
        if module in seen:
            raise ValueError(f"duplicate public API module: {module}")
        if not module_path_for(module).is_file():
            raise ValueError(f"public API module does not exist: {module}")
        seen.add(module)
        normalized.append({"module": module, "tier": tier})
    return tuple(normalized)


def resolve_relative(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    """Resolve a (possibly relative) ImportFrom to an absolute module name."""
    if node.level == 0:
        return node.module or ""
    parts = module.split(".")
    up = node.level - (1 if is_package else 0)
    base = parts[: len(parts) - up] if up > 0 else parts
    return ".".join(base + node.module.split(".")) if node.module else ".".join(base)


def iter_source_files() -> list[Path]:
    return sorted(
        p
        for p in (SRC_ROOT / "lunaris").rglob("*.py")
        if "__pycache__" not in p.parts
    )


def _is_private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def _private_path_rows(importer: str, imported: str) -> set[str]:
    """Describe private components in one absolute imported module path."""
    rows: set[str] = set()
    parts = imported.split(".")
    for index, name in enumerate(parts):
        if not _is_private(name):
            continue
        source = ".".join(parts[:index])
        if source.startswith("lunaris") and unit_of(source) != unit_of(importer):
            rows.add(f"{importer} <- {source} :: {name}")
    return rows


def _attribute_parts(node: ast.Attribute) -> list[str] | None:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    parts.reverse()
    return parts


def private_boundary_violations(
    module: str,
    source_text: str,
    *,
    is_package: bool = False,
) -> list[str]:
    """Return private cross-unit imports/accesses found in one source string."""
    tree = ast.parse(source_text)
    rows: set[str] = set()

    # local binding -> (absolute module base, attribute prefix required for an
    # unaliased dotted import). A binding can represent several imports, most
    # commonly the shared ``lunaris`` root.
    bindings: dict[str, list[tuple[str, tuple[str, ...]]]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported = alias.name
                if not imported.startswith("lunaris"):
                    continue
                rows.update(_private_path_rows(module, imported))
                parts = imported.split(".")
                if alias.asname:
                    binding = alias.asname
                    target = imported
                    required: tuple[str, ...] = ()
                else:
                    binding = parts[0]
                    target = parts[0]
                    required = tuple(parts[1:])
                bindings.setdefault(binding, []).append((target, required))
        elif isinstance(node, ast.ImportFrom):
            imported_from = resolve_relative(module, is_package, node)
            if not imported_from.startswith("lunaris"):
                continue
            rows.update(_private_path_rows(module, imported_from))
            for alias in node.names:
                name = alias.name
                if _is_private(name) and unit_of(imported_from) != unit_of(module):
                    rows.add(f"{module} <- {imported_from} :: {name}")
                    continue
                # Track ``from package import module as alias`` only when the
                # target is demonstrably a module. This avoids treating a
                # class/function import as a module alias.
                candidate = f"{imported_from}.{name}"
                if name != "*" and module_path_for(candidate).is_file():
                    bindings.setdefault(alias.asname or name, []).append((candidate, ()))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        parts = _attribute_parts(node)
        if not parts or parts[0] not in bindings:
            continue
        attrs = parts[1:]
        for target, required in bindings[parts[0]]:
            if tuple(attrs[: len(required)]) != required:
                continue
            effective_attrs = attrs[len(required):]
            base_parts = target.split(".") + list(required)
            for index, name in enumerate(effective_attrs):
                if not _is_private(name):
                    continue
                owner = ".".join(base_parts + effective_attrs[:index])
                if owner.startswith("lunaris") and unit_of(owner) != unit_of(module):
                    rows.add(f"{module} <- {owner} :: {name}")
    return sorted(rows)


def cross_unit_private_accesses() -> list[str]:
    """List ``importer <- source :: _name`` rows crossing a unit boundary."""
    rows: set[str] = set()
    for path in iter_source_files():
        module = module_name_for(path)
        try:
            rows.update(
                private_boundary_violations(
                    module,
                    path.read_text(encoding="utf-8-sig"),
                    is_package=path.name == "__init__.py",
                )
            )
        except SyntaxError as exc:
            # Fail closed: a file the scanner cannot parse is a file it cannot
            # vouch for. Silently skipping it would let boundary violations
            # hide behind a syntax error.
            raise SyntaxError(f"api_inventory could not parse {path}: {exc}") from exc
    return sorted(rows)


def cross_unit_underscore_imports() -> list[str]:
    """Backward-compatible name for :func:`cross_unit_private_accesses`."""
    return cross_unit_private_accesses()


def build_inventory() -> dict[str, object]:
    module_all: dict[str, list[str] | None] = {}
    module_tiers: dict[str, str] = {}
    for entry in load_public_api_manifest():
        module = entry["module"]
        module_all[module] = extract_all(module_path_for(module))
        module_tiers[module] = entry["tier"]
    return {
        "schema": SCHEMA,
        "module_all": module_all,
        "module_tiers": module_tiers,
        "cross_unit_private_accesses": cross_unit_private_accesses(),
    }


def render(inventory: dict[str, object]) -> str:
    return json.dumps(inventory, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write", action="store_true", help="regenerate docs/api_snapshot.json"
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed snapshot is stale",
    )
    args = parser.parse_args(argv)

    text = render(build_inventory())
    if args.write:
        SNAPSHOT_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {SNAPSHOT_PATH.relative_to(REPO_ROOT)}")
        return 0
    if args.check:
        committed = (
            SNAPSHOT_PATH.read_text(encoding="utf-8")
            if SNAPSHOT_PATH.is_file()
            else ""
        )
        if committed != text:
            print(
                "docs/api_snapshot.json is stale; run "
                "`python tools/api_inventory.py --write` and review the diff.",
                file=sys.stderr,
            )
            return 1
        print("api snapshot is current")
        return 0
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
