"""Every benchmark manifest's ``generator.script_sha256`` names the script that
produced its data.

The generators hash their own bytes. ``.gitattributes`` pins the generator
scripts to LF so a Windows checkout hashes the same bytes git stores; the check
below still normalises CRLF so a checkout made before that rule cannot fail it.
If this test fails, rerun the generator and commit its output (the data files
must come out byte-identical, or the change is a new reference) - never edit the
pin by hand.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "validation" / "gravity_reference" / "benchmarks"


def _manifests() -> list[Path]:
    return sorted(BENCHMARKS.glob("*/*.json"))


def _lf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_benchmark_manifests_exist() -> None:
    assert len(_manifests()) >= 4


@pytest.mark.parametrize(
    "manifest", _manifests(), ids=lambda p: p.relative_to(BENCHMARKS).as_posix()
)
def test_generator_script_hash_matches_the_committed_script(manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    generator = payload["generator"]
    script = ROOT / generator["script_path"]
    assert script.is_file(), f"generator script missing: {generator['script_path']}"
    recorded = generator["script_sha256"]
    actual = _lf_sha256(script)
    assert actual == recorded, (
        f"{manifest.name}: generator.script_sha256 {recorded[:12]} does not match "
        f"{script.name} (LF bytes {actual[:12]}); rerun the generator, do not edit the pin"
    )
