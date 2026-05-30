# -*- coding: utf-8 -*-
"""
Regression tests for the main-entry ephemeris bootstrap policy.

The user-facing launcher should only request Sun/Earth vector tables when the
active force model set truly needs them. This keeps SH/topography runs quiet and
lightweight while preserving full ephemeris tables for third-body / SRP cases.
"""

from __future__ import annotations

from types import SimpleNamespace

from lunaris.common.type_defs import PerturbationFlags, TimeConfig
from lunaris.physics.ephemeris import SpiceBuildConfig

from lunaris.core.config import load_default_config
import lunaris.cli.main as main


def _make_cfg(*, flags: PerturbationFlags) -> SimpleNamespace:
    """Create the minimal config shape consumed by ``main.init_ephemeris``."""

    return SimpleNamespace(
        time=TimeConfig(
            start_date="2027-03-02T23:32:37",
            duration_s=86400.0,
            output_dt_s=60.0,
        ),
        spice=SpiceBuildConfig(
            kernels=("naif0012.tls", "de440.bsp", "moon_pa_de440_200625.bpc"),
            include_third_body=True,
        ),
        flags=flags,
    )


def test_init_ephemeris_disables_third_body_sampling_for_q_only_runs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_from_time_and_spice(time_cfg, spice_cfg, **kwargs):
        captured["time_cfg"] = time_cfg
        captured["spice_cfg"] = spice_cfg
        captured["kwargs"] = kwargs
        return "mock-ephem"

    monkeypatch.setattr(
        "lunaris.physics.ephemeris.EphemerisManager.from_time_and_spice",
        _fake_from_time_and_spice,
    )

    cfg = _make_cfg(
        flags=PerturbationFlags(
            enable_sh=True,
            enable_3rd_body_sun=False,
            enable_3rd_body_earth=False,
            enable_earth_j2=False,
            enable_srp=False,
            enable_albedo=False,
            enable_thermal=False,
            enable_tides_k2=False,
            enable_tides_k3=False,
            enable_relativity_1pn=False,
        )
    )

    result = main.init_ephemeris(cfg, tf_s=3600.0)

    assert result == "mock-ephem"
    assert captured["spice_cfg"].include_third_body is False
    assert captured["kwargs"]["need_moon_fixed_rotation"] is True
    assert captured["time_cfg"].duration_s > 3600.0


def test_init_ephemeris_keeps_third_body_sampling_when_sun_vector_is_needed(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_from_time_and_spice(time_cfg, spice_cfg, **kwargs):
        captured["spice_cfg"] = spice_cfg
        return "mock-ephem"

    monkeypatch.setattr(
        "lunaris.physics.ephemeris.EphemerisManager.from_time_and_spice",
        _fake_from_time_and_spice,
    )

    cfg = _make_cfg(
        flags=PerturbationFlags(
            enable_sh=True,
            enable_3rd_body_sun=False,
            enable_3rd_body_earth=False,
            enable_earth_j2=False,
            enable_srp=True,
            enable_albedo=False,
            enable_thermal=False,
            enable_tides_k2=False,
            enable_tides_k3=False,
            enable_relativity_1pn=False,
        )
    )

    result = main.init_ephemeris(cfg, tf_s=3600.0)

    assert result == "mock-ephem"
    assert captured["spice_cfg"].include_third_body is True


def test_apply_args_to_config_canonicalizes_offset_start_dates_to_utc() -> None:
    cfg = load_default_config()
    args = main.parse_args(["--start-date", "2026-05-10T19:19:47+03:00"])

    cfg2 = main.apply_args_to_config(cfg, args)

    assert cfg2.time.start_date == "2026-05-10T16:19:47Z"
