"""Discovery-vs-strict lunar numeric contract separation."""

from __future__ import annotations

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.lunar_data import (
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
