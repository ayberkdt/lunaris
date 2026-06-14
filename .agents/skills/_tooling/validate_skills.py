#!/usr/bin/env python3
"""Static validator for the Lunaris Claude Code agent-skills system.

Checks every skill under ``.claude/skills/<name>/SKILL.md`` for the structural
and hygiene properties described in ``docs/AGENT_SKILLS_ARCHITECTURE.md`` and
``docs/SKILLS_MAINTENANCE.md``. Pure standard library — no PyYAML needed.

Checks
------
* a ``SKILL.md`` exists in every skill directory;
* frontmatter parses and contains non-empty ``name`` and ``description``;
* ``name`` matches the directory and the kebab-case convention;
* names are unique across skills;
* description length stays within the budget (helps survive truncation);
* the main ``SKILL.md`` stays within the line budget;
* supporting-file references (``references/``, ``checklists/``, ``scripts/`` …)
  resolve relative to the skill directory;
* repository references (``docs/``, ``src/``, ``tests/``, ``configs/``,
  ``.claude/...``) point at paths that exist;
* duplicate / near-duplicate descriptions are flagged;
* generic non-project library skills are flagged for review (not deleted).

Exit codes
----------
* ``0`` — no errors (warnings may still be printed);
* ``1`` — one or more errors;
* ``2`` — usage / environment error (e.g. skills dir not found).

Usage
-----
    python .claude/skills/_tooling/validate_skills.py [--skills-dir DIR]
                                                      [--repo-root DIR]
                                                      [--max-desc-chars N]
                                                      [--max-lines N]
                                                      [--strict]
``--strict`` turns warnings into a non-zero exit as well.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

# Generic library skills that are NOT Lunaris-specific. They are flagged for the
# maintainer to remove/relocate (see docs/AGENT_SKILLS_ARCHITECTURE.md § exiting
# skill audit). The validator never deletes anything.
GENERIC_SKILLS = {
    "astropy",
    "matplotlib",
    "optimize-for-gpu",
    "pymc",
    "pytorch-lightning",
    "scientific-visualization",
    "statistical-analysis",
    "sympy",
}

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
TOPKEY_RE = re.compile(r"^[A-Za-z_][\w-]*:")
# Skill-local supporting files.
LOCAL_REF_RE = re.compile(
    r"(?<![\w/])((?:references|checklists|scripts|templates|examples|assets)/[\w./-]+\.\w+)"
)
# Repository-rooted references.
REPO_REF_RE = re.compile(r"(?<![\w./])((?:docs|src|tests|configs|\.claude)/[\w./-]+\.\w+)")


def parse_frontmatter(text: str) -> Optional[dict[str, str]]:
    """Extract a minimal ``name``/``description`` mapping from YAML frontmatter.

    Handles inline values and folded/literal block scalars (``>-``/``>``/``|``)
    for ``description`` without requiring a YAML library.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return None
    block = lines[1:end]
    data: dict[str, str] = {}
    i = 0
    while i < len(block):
        line = block[i]
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key, val = m.group(1), m.group(2).strip()
        if val in (">", ">-", "|", "|-", ""):
            # Gather the more-indented continuation lines.
            collected: list[str] = []
            j = i + 1
            while j < len(block):
                nxt = block[j]
                if nxt.strip() == "":
                    collected.append("")
                    j += 1
                    continue
                if TOPKEY_RE.match(nxt) and not nxt.startswith((" ", "\t")):
                    break
                collected.append(nxt.strip())
                j += 1
            data[key] = " ".join(s for s in collected if s).strip()
            i = j
        else:
            data[key] = val
            i += 1
    return data


def validate(  # noqa: PLR0912, PLR0915 - a linear sequence of checks reads clearly
    skills_dir: Path,
    repo_root: Path,
    *,
    max_desc_chars: int,
    max_lines: int,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    skill_dirs = sorted(
        d for d in skills_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
    )
    if not skill_dirs:
        errors.append(f"no skill directories found under {skills_dir}")
        return errors, warnings

    seen_names: dict[str, str] = {}
    descriptions: dict[str, str] = {}

    for d in skill_dirs:
        rel = d.name
        skill_md = d / "SKILL.md"
        if not skill_md.is_file():
            errors.append(f"{rel}: missing SKILL.md")
            continue

        # utf-8-sig so a stray UTF-8 BOM (seen in some third-party skills) does
        # not break frontmatter parsing.
        text = skill_md.read_text(encoding="utf-8-sig")
        n_lines = len(text.splitlines())
        if n_lines > max_lines:
            warnings.append(
                f"{rel}: SKILL.md is {n_lines} lines (> {max_lines}); move detail into references/"
            )

        fm = parse_frontmatter(text)
        if fm is None:
            errors.append(f"{rel}: frontmatter missing or unparseable")
            continue

        name = fm.get("name", "").strip()
        desc = fm.get("description", "").strip()

        if not name:
            errors.append(f"{rel}: frontmatter has no 'name'")
        else:
            if name != rel:
                errors.append(f"{rel}: name '{name}' does not match directory '{rel}'")
            if not NAME_RE.match(name):
                errors.append(f"{rel}: name '{name}' is not kebab-case")
            if name in seen_names:
                errors.append(f"{rel}: duplicate name '{name}' (also {seen_names[name]})")
            seen_names[name] = rel

        if not desc:
            errors.append(f"{rel}: frontmatter has no 'description'")
        else:
            if len(desc) > max_desc_chars:
                warnings.append(
                    f"{rel}: description is {len(desc)} chars (> {max_desc_chars} budget)"
                )
            descriptions[rel] = desc

        # Unsupported-field heads-up (Claude Code core skill format = name +
        # description; allowed-tools/model are optional and operator-added).
        known = {"name", "description", "license", "metadata", "compatibility",
                 "allowed-tools", "disallowed-tools", "model", "when_to_use"}
        for key in fm:
            if key not in known:
                warnings.append(f"{rel}: unrecognized frontmatter field '{key}'")

        # Link resolution.
        for match in LOCAL_REF_RE.findall(text):
            target = (d / match)
            if not target.exists():
                errors.append(f"{rel}: broken skill-local reference '{match}'")
        for match in REPO_REF_RE.findall(text):
            target = (repo_root / match)
            if not target.exists():
                errors.append(f"{rel}: reference to missing repo path '{match}'")

    # Duplicate description detection (exact or high token overlap).
    items = list(descriptions.items())
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            na, da = items[a]
            nb, db = items[b]
            if da == db:
                errors.append(f"{na} and {nb}: identical descriptions")
                continue
            ta, tb = set(da.lower().split()), set(db.lower().split())
            if ta and tb:
                jacc = len(ta & tb) / len(ta | tb)
                if jacc > 0.8:
                    warnings.append(
                        f"{na} and {nb}: descriptions {jacc:.0%} similar — check for trigger overlap"
                    )

    # Generic non-project skills.
    for d in skill_dirs:
        if d.name in GENERIC_SKILLS:
            warnings.append(
                f"{d.name}: generic library skill, not Lunaris-specific — recommend removal/relocation "
                "(see docs/AGENT_SKILLS_ARCHITECTURE.md existing-skill audit)"
            )

    return errors, warnings


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve()
    default_skills = here.parent.parent
    default_root = default_skills.parent.parent
    parser.add_argument("--skills-dir", type=Path, default=default_skills)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--max-desc-chars", type=int, default=1024)
    parser.add_argument("--max-lines", type=int, default=500)
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args(argv)

    skills_dir = args.skills_dir.resolve()
    if not skills_dir.is_dir():
        print(f"error: skills dir not found: {skills_dir}", file=sys.stderr)
        return 2

    errors, warnings = validate(
        skills_dir,
        args.repo_root.resolve(),
        max_desc_chars=args.max_desc_chars,
        max_lines=args.max_lines,
    )

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")

    n_skills = len([d for d in skills_dir.iterdir() if d.is_dir() and not d.name.startswith("_")])
    print(f"\n{n_skills} skill(s): {len(errors)} error(s), {len(warnings)} warning(s).")

    if errors:
        return 1
    if args.strict and warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
