"""Traceability and consistency checks for the Lunaris algorithm registry.

These tests keep ``docs/algorithms/algorithm_registry.yaml``,
``references/references.bib`` and the generated ``docs/ALGORITHM_CATALOG.md`` in
sync with the source tree. They exercise ``tools/algorithm_registry.py`` (a
standalone tool that does not import :mod:`lunaris`), loaded here by file path so
the ``tools`` directory does not need to be a package.

``pyyaml`` and ``jsonschema`` ship in the ``dev`` extra; on a minimal
install these tests skip rather than fail (mirroring the pyshtools/tudatpy
optional-dependency pattern used elsewhere in the suite).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("yaml")
pytest.importorskip("jsonschema")

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "algorithm_registry.py"

# Append-only: ids that must never silently disappear. Extend as domains land;
# never remove an id from this list (retire the entry with status: retired
# instead, which keeps it in the registry).
CORE_ENTRY_IDS = (
    "LUNARIS-ALG-TB-001",
    "LUNARIS-ALG-SUM-001",
    "LUNARIS-ALG-INT-001",
    "LUNARIS-ALG-SH-001",
    "LUNARIS-ALG-SH-002",
    "LUNARIS-ALG-FRM-001",
    "LUNARIS-ALG-INTP-001",
    "LUNARIS-DATA-EPH-001",
    "LUNARIS-ALG-ML-001",
    "LUNARIS-ALG-ML-002",
)


def _load_tool():
    spec = importlib.util.spec_from_file_location("algorithm_registry", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


@pytest.fixture(scope="module")
def registry() -> dict:
    return TOOL.load_registry()


@pytest.fixture(scope="module")
def bib() -> dict:
    return TOOL.parse_bib()


def test_registry_validates_against_schema(registry) -> None:
    schema = TOOL.load_schema()
    errors = TOOL.validate_schema(registry, schema)
    assert errors == [], "schema violations:\n" + "\n".join(errors)


def test_referential_integrity(registry, bib) -> None:
    # Only meaningful once the structure is schema-valid.
    schema_errors = TOOL.validate_schema(registry, TOOL.load_schema())
    assert schema_errors == [], "fix schema errors first:\n" + "\n".join(schema_errors)
    errors = TOOL.check_referential_integrity(registry, bib, REPO_ROOT)
    assert errors == [], "referential integrity failures:\n" + "\n".join(errors)


def test_catalog_is_fresh() -> None:
    stale = TOOL.run_generate(check=True)
    assert stale == [], (
        "docs/ALGORITHM_CATALOG.md is stale; run "
        "`python tools/algorithm_registry.py generate`:\n" + "\n".join(stale)
    )


def test_id_format_and_uniqueness(registry) -> None:
    import re

    pattern = re.compile(r"^LUNARIS-(ALG|MODEL|HEUR|DATA|STD)-[A-Z0-9]{2,4}-[0-9]{3}$")
    ids = [entry["id"] for entry in registry["entries"]]
    for eid in ids:
        assert pattern.match(eid), f"malformed id: {eid}"
    assert len(ids) == len(set(ids)), "duplicate ids present"
    slugs = [entry["slug"] for entry in registry["entries"]]
    assert len(slugs) == len(set(slugs)), "duplicate slugs present"
    assert ids == sorted(ids), "entries are not sorted by id"


def test_core_entries_present(registry) -> None:
    present = {entry["id"] for entry in registry["entries"]}
    missing = [eid for eid in CORE_ENTRY_IDS if eid not in present]
    assert missing == [], f"core registry entries disappeared: {missing}"


@pytest.mark.parametrize("entry_id", CORE_ENTRY_IDS)
def test_core_entry_symbols_resolve(registry, entry_id) -> None:
    entry = next(e for e in registry["entries"] if e["id"] == entry_id)
    for sym in entry["symbols"]:
        assert TOOL.symbol_exists(REPO_ROOT, sym["path"], sym.get("symbol")), (
            f"{entry_id}: unresolved symbol {sym['path']}::{sym.get('symbol')}"
        )


def test_no_vague_canonical_names(registry) -> None:
    offenders = [
        entry["id"]
        for entry in registry["entries"]
        if TOOL._VAGUE_NAME_RE.search(entry["canonical_name"]) and not entry.get("notes")
    ]
    assert offenders == [], f"vague canonical_name without notes waiver: {offenders}"


def test_verified_primary_source_has_locator(registry) -> None:
    """verified_primary_source must cite a resolvable identifier and a pointer."""
    for entry in registry["entries"]:
        if entry["verification_status"] != "verified_primary_source":
            continue
        primary = entry.get("primary_reference")
        assert primary is not None, f"{entry['id']}: verified but no primary_reference"
        assert primary["identifier_kind"] != "none", (
            f"{entry['id']}: verified_primary_source with identifier_kind 'none'"
        )
        pointer = any(primary.get(f) for f in ("section", "chapter", "pages")) or bool(
            primary.get("equations")
        )
        assert pointer, f"{entry['id']}: verified_primary_source without a locator"


def test_bib_records_have_identifiers(bib) -> None:
    for key, record in bib.items():
        assert any(record.get(field) for field in TOOL._IDENTIFIER_FIELDS), (
            f"references.bib: {key} has no persistent identifier"
        )
