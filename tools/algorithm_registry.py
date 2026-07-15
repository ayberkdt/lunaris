#!/usr/bin/env python3
"""Parser, validator and catalogue generator for the Lunaris algorithm registry.

The registry (``docs/algorithms/algorithm_registry.yaml``) is the machine-readable
source of truth that maps stable Lunaris algorithm identifiers to canonical method
names, verified bibliographic citations, implementing source symbols, validating
tests, and known assumptions/limitations. This tool:

* ``validate`` -- load the YAML, check it against
  ``schemas/algorithm_registry.schema.json``, then run the referential and
  cross-field integrity checks that JSON Schema cannot express (bib coverage,
  symbol existence, verification-status consistency, ...).
* ``generate [--check]`` -- render the human-readable ``docs/ALGORITHM_CATALOG.md``
  deterministically from the registry. ``--check`` fails if the on-disk catalogue
  is stale instead of rewriting it (used by CI / the test suite).
* ``audit`` -- authoring aid (not run in CI): grep ``src/lunaris`` for
  algorithm-ish keywords and list hits that no registry entry covers.

Design constraint: this tool must NOT import :mod:`lunaris`. It depends only on
the standard library plus ``pyyaml`` and ``jsonschema`` (the ``dev`` extra), so it
can validate the registry without importing the numerical package.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "docs" / "algorithms" / "algorithm_registry.yaml"
SCHEMA_PATH = REPO_ROOT / "schemas" / "algorithm_registry.schema.json"
BIB_PATH = REPO_ROOT / "references" / "references.bib"
CATALOG_PATH = REPO_ROOT / "docs" / "ALGORITHM_CATALOG.md"

# Stable display order for domain groups in the generated catalogue. Any domain
# present in the schema enum but missing here is appended alphabetically so the
# tool never silently drops entries.
DOMAIN_ORDER: tuple[str, ...] = (
    "TB",
    "SH",
    "SUM",
    "J2E",
    "TID",
    "REL",
    "RAD",
    "EPH",
    "FRM",
    "TIME",
    "INT",
    "EVT",
    "INTP",
    "OE",
    "SAMP",
    "UQ",
    "PHZ",
    "FRZ",
    "IMP",
    "ML",
    "OPT",
    "NUM",
    "DATA",
    "CST",
    "STD",
)

DOMAIN_TITLES: dict[str, str] = {
    "TB": "Third-body gravity",
    "SH": "Spherical-harmonic gravity",
    "SUM": "Compensated summation",
    "J2E": "Earth oblateness (J2)",
    "TID": "Solid tides",
    "REL": "Relativistic corrections",
    "RAD": "Radiation pressure",
    "EPH": "Ephemeris and interpolation",
    "FRM": "Frame conventions",
    "TIME": "Time systems",
    "INT": "Integrators",
    "EVT": "Event handling",
    "INTP": "Interpolation",
    "OE": "Orbital elements",
    "SAMP": "Sampling / design of experiments",
    "UQ": "Uncertainty quantification",
    "PHZ": "Phase / perturbation diagnostics",
    "FRZ": "Frozen-orbit search",
    "IMP": "Impact / terrain",
    "ML": "Neural architectures",
    "OPT": "Optimization",
    "NUM": "Numerical utilities",
    "DATA": "Scientific data products",
    "CST": "Physical constants",
    "STD": "Standards and conventions",
}

# Required bib fields per entry type (D6 of the plan). ``author_or_institution``
# means at least one of ``author`` / ``institution`` must be present.
_BIB_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "book": ("author", "title", "publisher", "year"),
    "article": ("author", "title", "journal", "year"),
    "inproceedings": ("author", "title", "booktitle", "year"),
    "incollection": ("author", "title", "booktitle", "year"),
    "techreport": ("title", "year"),  # author/institution checked separately
    "manual": ("title", "year"),
    "misc": ("title",),
    "online": ("title",),
    "dataset": ("title",),
    "standard": ("title",),
}

# At least one persistent identifier must be present on every bib record
# (task requirement #10). ``note`` is accepted only for books that explicitly
# explain an ISBN gap in the note text.
_IDENTIFIER_FIELDS: tuple[str, ...] = ("doi", "isbn", "number", "url", "isrn")

_VAGUE_NAME_RE = re.compile(
    r"\b(advanced|optimized|optimised|professional|improved|enhanced)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _require_deps() -> tuple[Any, Any]:
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only w/o dev extra
        raise SystemExit(
            "pyyaml is required (install the dev extra: pip install -e .[dev])"
        ) from exc
    try:
        import jsonschema  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "jsonschema is required (install the dev extra: pip install -e .[dev])"
        ) from exc
    return yaml, jsonschema


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    yaml, _ = _require_deps()
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"{path} did not parse to a mapping")
    return data


def load_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    import json  # noqa: PLC0415

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Minimal BibTeX reader (avoids adding a bibtexparser dependency)
# ---------------------------------------------------------------------------
def parse_bib(path: Path = BIB_PATH) -> dict[str, dict[str, str]]:
    """Return ``{citation_key: {field: value}}`` for a BibTeX file.

    Handles brace- and quote-delimited values with nested braces. Values are
    returned verbatim (outer delimiters stripped, inner whitespace collapsed).
    Field names and the entry type are lower-cased; the citation key keeps case.
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    records: dict[str, dict[str, str]] = {}
    i = 0
    n = len(text)
    while i < n:
        at = text.find("@", i)
        if at == -1:
            break
        brace = text.find("{", at)
        if brace == -1:
            break
        entry_type = text[at + 1 : brace].strip().lower()
        if entry_type in {"comment", "preamble", "string"}:
            i = brace + 1
            continue
        # Find the matching close brace for the whole entry.
        depth = 0
        j = brace
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[brace + 1 : j]
        i = j + 1
        comma = body.find(",")
        if comma == -1:
            continue
        key = body[:comma].strip()
        fields = _parse_bib_fields(body[comma + 1 :])
        fields["__type__"] = entry_type
        records[key] = fields
    return records


def _parse_bib_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    i = 0
    n = len(body)
    while i < n:
        eq = body.find("=", i)
        if eq == -1:
            break
        name = body[i:eq].strip().strip(",").lower()
        j = eq + 1
        while j < n and body[j] in " \t\r\n":
            j += 1
        if j >= n:
            break
        if body[j] == "{":
            depth = 0
            start = j
            while j < n:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            value = body[start + 1 : j]
            i = j + 1
        elif body[j] == '"':
            start = j + 1
            j = start
            while j < n and body[j] != '"':
                j += 1
            value = body[start:j]
            i = j + 1
        else:
            start = j
            while j < n and body[j] != ",":
                j += 1
            value = body[start:j]
            i = j + 1
        # Skip a trailing comma.
        while i < n and body[i] in " \t\r\n,":
            i += 1
        if name:
            fields[name] = re.sub(r"\s+", " ", value).strip()
    return fields


# ---------------------------------------------------------------------------
# Symbol resolution (AST, no import of the target module)
# ---------------------------------------------------------------------------
def _module_defined_names(path: Path) -> set[str]:
    """Return every ``def``/``class`` name defined in the module.

    Names are collected anywhere in the tree (module level, class bodies, and
    functions nested inside other functions -- e.g. a CUDA kernel defined inside
    a dispatch function) so a registry symbol resolves regardless of nesting.
    Direct ``Class.method`` children are also exposed in dotted form so an entry
    can point at a specific method.
    """
    # Some source files carry a UTF-8 BOM; utf-8-sig strips it so ast.parse does
    # not choke on a leading U+FEFF.
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(f"{node.name}.{child.name}")
    return names


def symbol_exists(repo_root: Path, rel_path: str, symbol: str | None) -> bool:
    target = repo_root / rel_path
    if not target.exists():
        return False
    if symbol is None:
        return True
    try:
        return symbol in _module_defined_names(target)
    except (SyntaxError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_schema(registry: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    _, jsonschema = _require_deps()
    validator = jsonschema.Draft202012Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(registry), key=lambda e: list(e.path)):
        location = "/".join(str(p) for p in err.path) or "<root>"
        errors.append(f"schema[{location}]: {err.message}")
    return errors


def _id_prefix(entry_id: str) -> str:
    return entry_id.split("-")[1]


def check_referential_integrity(
    registry: dict[str, Any],
    bib: dict[str, dict[str, str]],
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    errors: list[str] = []
    entries = registry.get("entries", [])
    allowlist = set(registry.get("unreferenced_bib_allowlist", []))

    ids = [e["id"] for e in entries]
    slugs = [e["slug"] for e in entries]
    for label, values in (("id", ids), ("slug", slugs)):
        seen: set[str] = set()
        for value in values:
            if value in seen:
                errors.append(f"duplicate {label}: {value}")
            seen.add(value)

    if ids != sorted(ids):
        errors.append("entries are not sorted by id (run generate to see order)")

    all_ids = set(ids)
    cited_keys: set[str] = set()

    for entry in entries:
        eid = entry["id"]
        prefix = _id_prefix(eid)
        is_heur = prefix == "HEUR"
        cite_keys = entry.get("citation_keys", [])
        cited_keys.update(cite_keys)
        primary = entry.get("primary_reference")

        # Cross-rule: non-HEUR entries must be traceable to literature.
        if not is_heur:
            if not cite_keys:
                errors.append(f"{eid}: non-HEUR entry has no citation_keys")
            if primary is None:
                errors.append(f"{eid}: non-HEUR entry has null primary_reference")

        # Citation keys must resolve in the bibliography.
        for key in cite_keys:
            if key not in bib:
                errors.append(f"{eid}: citation key {key!r} not found in references.bib")

        if primary is not None:
            pkey = primary.get("citation_key")
            if pkey not in cite_keys:
                errors.append(
                    f"{eid}: primary_reference.citation_key {pkey!r} not in citation_keys"
                )
            # verified_primary_source demands a resolvable identifier AND a
            # concrete pointer into the source (section/chapter/equations/pages).
            if entry["verification_status"] == "verified_primary_source":
                if primary.get("identifier_kind") == "none":
                    errors.append(f"{eid}: verified_primary_source but identifier_kind is 'none'")
                pointer = any(
                    primary.get(field) for field in ("section", "chapter", "pages")
                ) or bool(primary.get("equations"))
                if not pointer:
                    errors.append(
                        f"{eid}: verified_primary_source but no section/chapter/"
                        "equations/pages pointer recorded"
                    )

        # see_also targets must exist.
        for ref in entry.get("see_also", []):
            if ref not in all_ids:
                errors.append(f"{eid}: see_also target {ref!r} is not a known id")

        # Symbols must resolve.
        for sym in entry.get("symbols", []):
            if not symbol_exists(repo_root, sym["path"], sym.get("symbol")):
                target = sym["path"]
                name = sym.get("symbol")
                detail = f"{target}::{name}" if name else target
                errors.append(f"{eid}: symbol not found: {detail}")

        # Validation test paths must exist; tested entries need at least one.
        validation = entry.get("validation", [])
        if entry["scientific_status"] == "implemented_and_tested" and not validation:
            errors.append(
                f"{eid}: scientific_status=implemented_and_tested but validation is empty"
            )
        for test_path in validation:
            if not (repo_root / test_path).exists():
                errors.append(f"{eid}: validation path does not exist: {test_path}")

        # Naming hygiene (Layer 3): reject vague adjectives unless a notes waiver
        # gives the precise technical definition.
        if _VAGUE_NAME_RE.search(entry["canonical_name"]) and not entry.get("notes"):
            errors.append(f"{eid}: canonical_name uses a vague adjective without a notes waiver")

    # Bib coverage: every record is cited (unless explicitly allowlisted), and
    # every cited record satisfies the field/identifier requirements.
    for key, record in bib.items():
        if key not in cited_keys and key not in allowlist:
            errors.append(f"references.bib: {key} is never cited by any entry")
        errors.extend(_check_bib_record(key, record))

    return errors


def _check_bib_record(key: str, record: dict[str, str]) -> list[str]:
    errors: list[str] = []
    entry_type = record.get("__type__", "misc")
    required = _BIB_REQUIRED_FIELDS.get(entry_type, ("title",))
    for field in required:
        if not record.get(field):
            errors.append(f"references.bib: {key} ({entry_type}) missing field '{field}'")
    if entry_type == "techreport" and not (record.get("author") or record.get("institution")):
        errors.append(f"references.bib: {key} (techreport) needs author or institution")
    if not any(record.get(field) for field in _IDENTIFIER_FIELDS):
        errors.append(
            f"references.bib: {key} has no persistent identifier (doi/isbn/number/url required)"
        )
    return errors


def run_validation(repo_root: Path = REPO_ROOT) -> list[str]:
    registry = load_registry()
    schema = load_schema()
    errors = validate_schema(registry, schema)
    # Referential checks assume the structure is sound; only run them if the
    # schema passed so error output stays readable.
    if not errors:
        bib = parse_bib()
        errors = check_referential_integrity(registry, bib, repo_root)
    return errors


# ---------------------------------------------------------------------------
# Catalogue generation
# ---------------------------------------------------------------------------
_GENERATED_HEADER = (
    "<!-- GENERATED FILE - do not edit by hand.\n"
    "     Edit docs/algorithms/algorithm_registry.yaml and run\n"
    "     `python tools/algorithm_registry.py generate`. -->\n"
)


def _anchor(entry_id: str) -> str:
    return entry_id.lower().replace("-", "")


def _ordered_domains(entries: list[dict[str, Any]]) -> list[str]:
    present = {e["domain"] for e in entries}
    ordered = [d for d in DOMAIN_ORDER if d in present]
    ordered += sorted(present - set(DOMAIN_ORDER))
    return ordered


def _reference_line(entry: dict[str, Any], bib: dict[str, dict[str, str]]) -> str:
    primary = entry.get("primary_reference")
    if primary is None:
        return "Lunaris-specific; no external primary reference."
    key = primary.get("citation_key", "")
    record = bib.get(key, {})
    author = record.get("author", record.get("institution", "")).split(" and ")[0]
    if "," in author:
        author = author.split(",")[0]
    year = record.get("year", "")
    title = record.get("title", "")
    bits = [b for b in (author.strip(), year.strip()) if b]
    citation = ", ".join(bits)
    ident_kind = primary.get("identifier_kind", "none")
    ident_value = ""
    for field in ("doi", "isbn", "report_number", "standard_number", "official_url"):
        if primary.get(field):
            ident_value = f"{field.replace('_', ' ').upper()}: {primary[field]}"
            break
    locus_parts = []
    for field in ("edition", "chapter", "section", "pages"):
        if primary.get(field):
            locus_parts.append(f"{field} {primary[field]}")
    if primary.get("equations"):
        locus_parts.append("eq. " + ", ".join(primary["equations"]))
    locus = "; ".join(locus_parts)
    line = f"`{key}`"
    if citation:
        line += f" -- {citation}"
    if title:
        line += f'. "{title}"'
    if locus:
        line += f" ({locus})"
    if ident_value:
        line += f" [{ident_value}]"
    elif ident_kind == "none":
        line += " [no persistent identifier]"
    return line


def generate_catalog(
    registry: dict[str, Any],
    bib: dict[str, dict[str, str]],
) -> str:
    entries = sorted(registry.get("entries", []), key=lambda e: e["id"])
    lines: list[str] = []
    lines.append(_GENERATED_HEADER.rstrip("\n"))
    lines.append("")
    lines.append("# Lunaris Algorithm Catalogue")
    lines.append("")
    lines.append(
        "Human-readable view of the algorithm-traceability registry. The source "
        "of truth is [`docs/algorithms/algorithm_registry.yaml`](algorithms/"
        "algorithm_registry.yaml); this file is generated. See "
        "[`docs/ALGORITHM_TRACEABILITY_POLICY.md`](ALGORITHM_TRACEABILITY_POLICY.md) "
        "for the naming, citation and classification policy."
    )
    lines.append("")

    # Summary counts.
    by_class: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for entry in entries:
        by_class[entry["implementation_class"]] = by_class.get(entry["implementation_class"], 0) + 1
        by_status[entry["verification_status"]] = by_status.get(entry["verification_status"], 0) + 1
    lines.append(f"**{len(entries)} entries.**")
    lines.append("")
    lines.append(
        "Implementation class: "
        + ", ".join(f"{name} ({count})" for name, count in sorted(by_class.items()))
    )
    lines.append("")
    lines.append(
        "Verification status: "
        + ", ".join(f"{name} ({count})" for name, count in sorted(by_status.items()))
    )
    lines.append("")

    # Index table.
    lines.append("## Index")
    lines.append("")
    lines.append("| ID | Method | Class | Verification |")
    lines.append("| --- | --- | --- | --- |")
    for entry in entries:
        lines.append(
            f"| [`{entry['id']}`](#{_anchor(entry['id'])}) "
            f"| {entry['canonical_name']} "
            f"| {entry['implementation_class']} "
            f"| {entry['verification_status']} |"
        )
    lines.append("")

    # Per-domain sections.
    for domain in _ordered_domains(entries):
        domain_entries = [e for e in entries if e["domain"] == domain]
        title = DOMAIN_TITLES.get(domain, domain)
        lines.append(f"## {title} ({domain})")
        lines.append("")
        for entry in domain_entries:
            lines.extend(_render_entry(entry, bib))
    return "\n".join(lines).rstrip("\n") + "\n"


def _render_entry(entry: dict[str, Any], bib: dict[str, dict[str, str]]) -> list[str]:
    out: list[str] = []
    out.append(f'<a id="{_anchor(entry["id"])}"></a>')
    out.append(f"### {entry['id']} -- {entry['canonical_name']}")
    out.append("")
    out.append(f"- **Slug**: `{entry['slug']}`")
    out.append(
        f"- **Category**: {entry['category']} | **Domain**: {entry['domain']} "
        f"| **Status**: {entry['status']}"
    )
    out.append(f"- **Classification**: {entry['implementation_class']}")
    out.append(
        f"- **Verification**: {entry['verification_status']} "
        f"| **Scientific status**: {entry['scientific_status']}"
    )
    out.append(f"- **Primary reference**: {_reference_line(entry, bib)}")
    primary = entry.get("primary_reference")
    if primary and primary.get("verification_notes"):
        out.append(f"- **Verification notes**: {primary['verification_notes']}")

    contract = entry.get("mathematical_contract") or {}
    if contract:
        out.append("- **Mathematical contract**:")
        if contract.get("inputs"):
            out.append(f"  - Inputs: {contract['inputs']}")
        if contract.get("outputs"):
            out.append(f"  - Outputs: {contract['outputs']}")
        if contract.get("exact_or_approximate"):
            out.append(f"  - Exactness: {contract['exact_or_approximate']}")
        for prop in contract.get("preserved_properties", []):
            out.append(f"  - Preserves: {prop}")

    symbols = entry.get("symbols", [])
    if symbols:
        out.append("- **Implementing symbols**:")
        for sym in symbols:
            name = sym.get("symbol") or "(module)"
            out.append(f"  - `{sym['path']}` -- `{name}` ({sym['role']})")

    for label, field in (
        ("Lunaris modifications", "lunaris_modifications"),
        ("Assumptions", "assumptions"),
        ("Limitations", "limitations"),
    ):
        values = entry.get(field, [])
        if values:
            out.append(f"- **{label}**:")
            for value in values:
                out.append(f"  - {value}")

    validation = entry.get("validation", [])
    if validation:
        out.append("- **Validated by**:")
        for test_path in validation:
            out.append(f"  - `{test_path}`")

    see_also = entry.get("see_also", [])
    if see_also:
        out.append(
            "- **See also**: " + ", ".join(f"[`{ref}`](#{_anchor(ref)})" for ref in see_also)
        )
    if entry.get("notes"):
        out.append(f"- **Notes**: {entry['notes']}")
    out.append("")
    return out


def run_generate(check: bool) -> list[str]:
    registry = load_registry()
    bib = parse_bib()
    rendered = generate_catalog(registry, bib)
    if check:
        if not CATALOG_PATH.exists():
            return [f"{CATALOG_PATH} does not exist; run generate"]
        current = CATALOG_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
        if current != rendered:
            return [f"{CATALOG_PATH} is stale; run `python tools/algorithm_registry.py generate`"]
        return []
    CATALOG_PATH.write_text(rendered, encoding="utf-8", newline="\n")
    return []


# ---------------------------------------------------------------------------
# Coverage audit (authoring aid; not run in CI)
# ---------------------------------------------------------------------------
_AUDIT_KEYWORDS: tuple[str, ...] = (
    "kahan",
    "slerp",
    "catmull",
    "sobol",
    "yoshida",
    "pefrl",
    "verlet",
    "siren",
    "fourier",
    "legendre",
    "battin",
    "radau",
    "dop853",
    "nystrom",
    "leapfrog",
    "encke",
    "schwarzschild",
    "sitter",
    "love number",
    "smoothstep",
)


_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _covered_line_span(tree: ast.AST, covered_names: set[str]) -> set[int]:
    """Return the 1-based line numbers spanned by registered symbols in a file.

    Symbol-level (not file-level) coverage: the audit masks out exactly the
    definitions the registry already accounts for and keeps scanning the rest of
    the file, instead of declaring the whole file "covered" the moment any single
    symbol in it is registered. ``ast.walk`` reaches nested defs, so a nested
    helper can carry its own registered symbol.
    """
    covered_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, _DEF_NODES) and node.name in covered_names:
            end = getattr(node, "end_lineno", node.lineno)
            covered_lines.update(range(node.lineno, int(end) + 1))
    return covered_lines


def _uncovered_identifiers(tree: ast.AST, covered_lines: set[int]) -> set[str]:
    """Lowercased identifier tokens that live outside any registered symbol.

    Only *identifiers* count — def/class/arg names, ``Name`` references, and
    attribute accesses. String constants, docstrings, and comments are ignored on
    purpose: an algorithm keyword inside a config enum value (``"dop853"``), a
    ``Literal[...]`` annotation, a help string, or a prose comment is a *mention*,
    not an unregistered implementation, and flagging those buries the real signal.
    """
    idents: set[str] = set()
    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", None)
        if lineno is None or lineno in covered_lines:
            continue
        if isinstance(node, _DEF_NODES):
            idents.add(node.name.lower())
        elif isinstance(node, ast.Name):
            idents.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            idents.add(node.attr.lower())
        elif isinstance(node, ast.arg):
            idents.add(node.arg.lower())
    return idents


def run_audit(repo_root: Path = REPO_ROOT) -> list[str]:
    """List source files containing algorithm-ish code not covered by symbols.

    Coverage is evaluated at the *symbol* level, not the file level: a file with
    one registered function no longer masks a second, unregistered algorithm in
    the same file. Registered symbols' line spans are removed before the scan, and
    matching is restricted to identifier tokens (see :func:`_uncovered_identifiers`)
    so keyword mentions in strings/comments do not produce false positives. An
    entry whose ``symbol`` is ``null`` covers the whole file (an intentional
    file-wide registration), preserving that acknowledgement.
    """
    registry = load_registry()
    covered_names: dict[str, set[str]] = {}
    whole_file_covered: set[str] = set()
    for entry in registry.get("entries", []):
        for sym in entry.get("symbols", []):
            resolved = (repo_root / sym["path"]).resolve().as_posix()
            name = sym.get("symbol")
            if name:
                covered_names.setdefault(resolved, set()).add(name)
            else:
                whole_file_covered.add(resolved)

    src = repo_root / "src" / "lunaris"
    hits: list[str] = []
    for py in sorted(src.rglob("*.py")):
        resolved = py.resolve().as_posix()
        if resolved in whole_file_covered:
            continue
        try:
            source = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Unparseable file: fall back to a whole-file text scan so a keyword
            # is never missed just because the AST could not be built.
            scan_text = source.lower()
            matched = sorted({kw for kw in _AUDIT_KEYWORDS if kw in scan_text})
            if matched:
                hits.append(f"{py.relative_to(repo_root).as_posix()}: {', '.join(matched)}")
            continue

        covered_lines = _covered_line_span(tree, covered_names.get(resolved, set()))
        identifiers = _uncovered_identifiers(tree, covered_lines)
        matched = sorted(
            {kw for kw in _AUDIT_KEYWORDS if any(kw in ident for ident in identifiers)}
        )
        if matched:
            rel = py.relative_to(repo_root).as_posix()
            hits.append(f"{rel}: {', '.join(matched)}")
    return hits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="schema + referential integrity checks")
    gen = sub.add_parser("generate", help="render docs/ALGORITHM_CATALOG.md")
    gen.add_argument("--check", action="store_true", help="fail if catalogue is stale")
    aud = sub.add_parser("audit", help="list uncovered algorithm-ish source hits")
    aud.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when any uncovered hit is found (for CI gating)",
    )
    args = parser.parse_args(argv)

    if args.command == "validate":
        errors = run_validation()
        if errors:
            print(f"FAIL: {len(errors)} registry problem(s):")
            for err in errors:
                print(f"  - {err}")
            return 1
        registry = load_registry()
        print(f"OK: {len(registry.get('entries', []))} entries valid.")
        return 0

    if args.command == "generate":
        errors = run_generate(check=args.check)
        if errors:
            for err in errors:
                print(f"  - {err}")
            return 1
        print("OK: catalogue up to date." if args.check else f"Wrote {CATALOG_PATH}.")
        return 0

    if args.command == "audit":
        hits = run_audit()
        if not hits:
            print("No uncovered algorithm-ish source hits.")
            return 0
        print(f"{len(hits)} file(s) with algorithm-ish keywords not covered by symbols:")
        for hit in hits:
            print(f"  - {hit}")
        # Advisory by default (authoring aid); --strict lets CI gate on it.
        return 1 if args.strict else 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
