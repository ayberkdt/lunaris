#!/usr/bin/env python3
"""Validate the Lunaris agent-skills trigger-case fixture.

This is a *static* checker for ``tests/agent_skills/trigger_cases.json`` — it
makes the trigger-quality review repeatable without invoking a hosted model
(task SS11.2). It verifies referential integrity and coverage:

* every ``expected_primary`` (when not null), ``allowed_secondary`` and
  ``prohibited`` entry names a real skill (cross-checked against the skills on
  disk and the fixture's ``skills`` list);
* ``prohibited`` does not also appear as the expected primary for the same case;
* generic non-project skills are never an ``expected_primary``;
* every project skill is exercised as ``expected_primary`` by at least
  ``--min-positive`` case(s);
* at least ``--min-negative`` "no project skill" / near-miss cases exist.

Exit codes: ``0`` ok, ``1`` problems found, ``2`` usage error.

Usage
-----
    python .claude/skills/_tooling/check_trigger_cases.py
        [--fixture tests/agent_skills/trigger_cases.json]
        [--skills-dir .claude/skills] [--min-positive 1] [--min-negative 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", type=Path, default=repo_root / "tests/agent_skills/trigger_cases.json")
    parser.add_argument("--skills-dir", type=Path, default=here.parent.parent)
    parser.add_argument("--min-positive", type=int, default=1)
    parser.add_argument("--min-negative", type=int, default=3)
    args = parser.parse_args(argv)

    if not args.fixture.is_file():
        print(f"error: fixture not found: {args.fixture}", file=sys.stderr)
        return 2
    data = json.loads(args.fixture.read_text(encoding="utf-8"))

    on_disk = {
        d.name for d in args.skills_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "SKILL.md").is_file()
    }
    declared = set(data.get("skills", []))
    generic = set(data.get("generic_skills_to_avoid", []))
    project_skills = declared & on_disk

    errors: list[str] = []
    warnings: list[str] = []

    # Declared skills should exist on disk (allow the generic ones to be absent).
    for name in declared:
        if name not in on_disk:
            errors.append(f"fixture lists skill '{name}' that has no SKILL.md on disk")

    primary_counts: Counter[str] = Counter()
    negatives = 0
    cases = data.get("cases", [])
    if not cases:
        errors.append("fixture has no cases")

    valid_refs = on_disk | generic
    for idx, case in enumerate(cases):
        prompt = case.get("prompt", "")
        if not prompt:
            errors.append(f"case {idx}: empty prompt")
        if "rationale" not in case or not case["rationale"]:
            warnings.append(f"case {idx}: missing rationale")
        primary = case.get("expected_primary")
        if primary is None:
            negatives += 1
        else:
            if primary not in on_disk:
                errors.append(f"case {idx}: expected_primary '{primary}' is not a real skill")
            if primary in generic:
                errors.append(f"case {idx}: expected_primary '{primary}' is a generic non-project skill")
            primary_counts[primary] += 1
        for field in ("allowed_secondary", "prohibited"):
            for ref in case.get(field, []):
                if ref not in valid_refs:
                    errors.append(f"case {idx}: {field} references unknown skill '{ref}'")
        if primary in case.get("prohibited", []):
            errors.append(f"case {idx}: '{primary}' is both expected_primary and prohibited")

    # Coverage: every project skill is a primary at least min-positive times.
    for name in sorted(project_skills):
        if primary_counts[name] < args.min_positive:
            warnings.append(
                f"coverage: skill '{name}' is expected_primary in {primary_counts[name]} case(s) "
                f"(< {args.min_positive})"
            )
    if negatives < args.min_negative:
        warnings.append(f"coverage: only {negatives} negative/near-miss case(s) (< {args.min_negative})")

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print(
        f"\n{len(cases)} case(s), {len(project_skills)} project skill(s) covered, "
        f"{negatives} negative case(s): {len(errors)} error(s), {len(warnings)} warning(s)."
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
