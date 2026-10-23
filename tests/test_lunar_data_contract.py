"""Discovery-vs-strict lunar numeric contract separation."""

from __future__ import annotations

from pathlib import Path

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.lunar_data import (
    _default_lunar_gravity_path,
    is_lunar_body_signature,
    looks_lunar_like,
    validate_lunar_contract,
)


def test_discovery_heuristic_remains_loose_for_legacy_search() -> None:
    legacy_rounded = float(MU_MOON) * 1.10

    assert looks_lunar_like(mu_si=legacy_rounded)
    assert is_lunar_body_signature(mu_si=legacy_rounded)


def test_strict_lunar_contract_rejects_discovery_scale_mismatch() -> None:
    assert not validate_lunar_contract(mu_si=float(MU_MOON) * 1.001, r_ref_m=float(R_MOON))
    assert not validate_lunar_contract(mu_si=float(MU_MOON), r_ref_m=float(R_MOON) * 1.001)


def test_strict_lunar_contract_accepts_source_specific_gravity_gm() -> None:
    assert validate_lunar_contract(mu_si=4.9028003063302e12, r_ref_m=1_738_000.0)


def test_strict_lunar_contract_accepts_established_mean_radius_convention() -> None:
    assert validate_lunar_contract(mu_si=float(MU_MOON), r_ref_m=1_737_400.0)


def test_strict_lunar_contract_accepts_common_rounded_lunar_constants() -> None:
    assert validate_lunar_contract(mu_si=4.902e12, r_ref_m=1_737_000.0)


def test_default_gravity_file_is_found_under_its_downloaded_name(tmp_path: Path) -> None:
    """`lunaris-data download` installs jggrx_1800f_sha.tab; a checkout may hold the
    .txt copy. Before 2026-10-23 only the .txt name was looked up, so a fresh HPC
    data directory could not supply the ST-LRPS SH25 baseline."""
    gravity = tmp_path / "gravity_models"
    gravity.mkdir()
    assert _default_lunar_gravity_path(tmp_path) == gravity / "jggrx_1800f_sha.tab.txt"  # nothing there
    (gravity / "jggrx_1800f_sha.tab.txt").write_text("x", encoding="utf-8")
    assert _default_lunar_gravity_path(tmp_path) == gravity / "jggrx_1800f_sha.tab.txt"
    (gravity / "jggrx_1800f_sha.tab").write_text("x", encoding="utf-8")
    assert _default_lunar_gravity_path(tmp_path) == gravity / "jggrx_1800f_sha.tab"  # core's priority
    (gravity / "jggrx_1800f_sha.tab.txt").unlink()
    assert _default_lunar_gravity_path(tmp_path) == gravity / "jggrx_1800f_sha.tab"
