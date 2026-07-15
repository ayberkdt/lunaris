"""Release metadata must agree with itself.

These files describe the project to the outside world (PyPI, Zenodo, citations),
and nothing was checking that they told the same story. They drifted: after
0.1.0rc1 the CHANGELOG dated the release 2026-07-11 while CITATION.cff said
2026-07-14, and the "Unreleased" section read "(nothing yet)" through five
merged PRs and ~12.6k inserted lines.

The point of these tests is not tidiness. A version or date that is wrong in
metadata is wrong in every citation that quotes it, and a changelog that claims
nothing has changed is worse than no changelog. Neither failure mode is visible
to the lint, type, or test gates.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CITATION = REPO_ROOT / "CITATION.cff"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

#: `## <version> — <ISO date>` (em dash, as used by the changelog headings).
_RELEASE_HEADING = re.compile(
    r"^##\s+(?P<version>\d+\.\d+\.\d+(?:[a-z]+\d+)?)\s+—\s+(?P<date>\d{4}-\d{2}-\d{2})\s*$",
    re.MULTILINE,
)


def _project_version() -> str:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def _citation_field(name: str) -> str:
    # Deliberately not a YAML parse: CITATION.cff is the only consumer of a YAML
    # dependency in the test suite otherwise, and these two fields are flat
    # top-level scalars. Keep the check dependency-free.
    pattern = re.compile(rf'^{name}:\s*"?([^"\n#]+)"?\s*(?:#.*)?$', re.MULTILINE)
    match = pattern.search(CITATION.read_text(encoding="utf-8"))
    assert match is not None, f"{name} not found in CITATION.cff"
    return match.group(1).strip()


def _released_versions() -> list[tuple[str, str]]:
    text = CHANGELOG.read_text(encoding="utf-8")
    return [(m.group("version"), m.group("date")) for m in _RELEASE_HEADING.finditer(text)]


def test_citation_version_matches_pyproject() -> None:
    assert _citation_field("version") == _project_version(), (
        "CITATION.cff version and pyproject.toml version disagree; a citation "
        "generated from this repo would name the wrong release"
    )


def test_changelog_documents_the_current_version() -> None:
    version = _project_version()
    versions = [v for v, _ in _released_versions()]
    assert version in versions, (
        f"pyproject version {version} has no '## {version} — <date>' heading in "
        f"CHANGELOG.md (found: {versions or 'none'}). Either the release is "
        f"undocumented or the version was bumped without a changelog section."
    )


def test_citation_release_date_matches_the_changelog() -> None:
    version = _project_version()
    dates = dict(_released_versions())
    assert _citation_field("date-released") == dates[version], (
        f"CITATION.cff date-released does not match the CHANGELOG date for "
        f"{version}. date-released is the *release* date, not the date the file "
        f"was last edited — that is exactly how these two drifted apart before."
    )


def test_unreleased_section_is_not_stale() -> None:
    """An 'Unreleased' section must not claim emptiness while holding content.

    This does not (and cannot) assert that every merged PR is logged — only a
    human knows that. It catches the specific regression we hit: the placeholder
    surviving after real work landed on top of it.
    """
    text = CHANGELOG.read_text(encoding="utf-8")
    match = re.search(r"^##\s+Unreleased\s*$(.*?)(?=^##\s)", text, re.MULTILINE | re.DOTALL)
    if match is None:
        pytest.skip("no Unreleased section to check")
    body = match.group(1).strip()
    placeholder = body.lower() in {"", "(nothing yet)", "nothing yet", "- nothing yet", "tbd", "n/a"}
    has_entries = bool(re.search(r"^\s*[-*]\s+\S", body, re.MULTILINE))
    assert placeholder or has_entries, (
        "the Unreleased section has prose but no entries; if work has landed, "
        "list it — if it genuinely has not, use the '(nothing yet)' placeholder"
    )
