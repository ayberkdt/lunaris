"""Broad force-budget sanity checks for the single-run dynamics stack.

These tests are intentionally order-of-magnitude regressions, not precision
validation. They use deterministic ephemeris stubs so they can run on CPU-only
CI without SPICE kernels or gravity files.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.common.constants import AU, MU_MOON, R_MOON
from lunaris.common.type_defs import PerturbationFlags, SolidTideConfig, SpacecraftProps
from lunaris.core.dynamics import DynamicsEngine
from lunaris.physics.solar_effects import SRPConfig
from lunaris.physics.surface_effects import AlbedoConfig, ThermalConfig
from lunaris.physics.third_body_effects import EarthJ2Params

ALTITUDES_KM = (100.0, 300.0, 1000.0)


class _StaticEphem:
    def __init__(
        self,
        *,
        sun_m: tuple[float, float, float] = (AU, 0.0, 0.0),
        earth_m: tuple[float, float, float] = (384_400_000.0, 2.0e6, -1.0e6),
    ) -> None:
        self.sun_m = np.asarray(sun_m, dtype=np.float64)
        self.earth_m = np.asarray(earth_m, dtype=np.float64)

    def get_data_provider(self):
        return {
            "dt_s": 60.0,
            "r_sun_tab_m": np.vstack([self.sun_m, self.sun_m]).astype(np.float64),
            "r_earth_tab_m": np.vstack([self.earth_m, self.earth_m]).astype(np.float64),
            "q_i2f_tab": np.asarray(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
        }


def _spacecraft() -> SpacecraftProps:
    return SpacecraftProps(mass_kg=12.0, area_m2=2.0, cr=1.4, cd=2.2)


def _state(alt_km: float) -> np.ndarray:
    r = float(R_MOON) + float(alt_km) * 1000.0
    v = math.sqrt(float(MU_MOON) / r)
    return np.asarray([r, 0.0, 0.0, 0.0, v, 0.0], dtype=np.float64)


def _breakdown(engine: DynamicsEngine, alt_km: float) -> dict[str, float]:
    out = engine.get_acceleration_breakdown(0.0, _state(alt_km))
    for name, value in out.items():
        assert math.isfinite(float(value)), name
        assert float(value) >= 0.0, name
    return out


def _engine(*, flags: PerturbationFlags, **kwargs) -> DynamicsEngine:
    return DynamicsEngine(
        sc_props=_spacecraft(),
        flags=flags,
        gravity_model=kwargs.pop("gravity_model", None),
        ephem_manager=kwargs.pop("ephem_manager", _StaticEphem()),
        surface_provider=kwargs.pop("surface_provider", None),
        earth_j2=kwargs.pop("earth_j2", None),
        srp=kwargs.pop("srp", None),
        albedo=kwargs.pop("albedo", None),
        thermal=kwargs.pop("thermal", None),
        solid_tides=kwargs.pop("solid_tides", None),
        allow_identity_rotation=True,
        **kwargs,
    )


@pytest.mark.parametrize("alt_km", ALTITUDES_KM)
def test_central_lunar_gravity_budget_is_finite_and_plausible(alt_km: float) -> None:
    engine = _engine(flags=PerturbationFlags(enable_sh=False), ephem_manager=None)
    value = _breakdown(engine, alt_km)["Gravity (PM)"]
    expected = float(MU_MOON) / (float(R_MOON) + alt_km * 1000.0) ** 2

    assert value == pytest.approx(expected, rel=1.0e-12)
    assert 0.1 < value < 10.0


@pytest.mark.parametrize(
    ("flag_name", "key", "lo", "hi"),
    [
        ("enable_3rd_body_sun", "3rd Body (Sun)", 1.0e-8, 1.0e-5),
        ("enable_3rd_body_earth", "3rd Body (Earth)", 1.0e-7, 1.0e-3),
    ],
)
@pytest.mark.parametrize("alt_km", ALTITUDES_KM)
def test_third_body_budget_is_finite_nonzero_and_plausible(
    alt_km: float,
    flag_name: str,
    key: str,
    lo: float,
    hi: float,
) -> None:
    flags = PerturbationFlags(enable_sh=False, **{flag_name: True})
    comp = _breakdown(_engine(flags=flags), alt_km)

    assert key in comp
    assert lo < comp[key] < hi

    disabled = _breakdown(_engine(flags=PerturbationFlags(enable_sh=False)), alt_km)
    assert key not in disabled


@pytest.mark.parametrize("alt_km", ALTITUDES_KM)
def test_earth_j2_budget_is_finite_nonzero_and_plausible(alt_km: float) -> None:
    engine = _engine(
        flags=PerturbationFlags(enable_sh=False, enable_earth_j2=True),
        earth_j2=EarthJ2Params(),
    )
    value = _breakdown(engine, alt_km)["3rd Body (Earth J2)"]

    assert 1.0e-12 < value < 1.0e-6


@pytest.mark.parametrize("alt_km", ALTITUDES_KM)
def test_srp_budget_is_finite_nonzero_and_plausible(alt_km: float) -> None:
    engine = _engine(
        flags=PerturbationFlags(enable_sh=False, enable_srp=True),
        srp=SRPConfig(enable_moon_eclipse=False),
    )
    value = _breakdown(engine, alt_km)["SRP"]

    assert 1.0e-9 < value < 1.0e-5


@pytest.mark.parametrize(
    ("flags", "kwargs", "key", "lo", "hi"),
    [
        (
            PerturbationFlags(enable_sh=False, enable_albedo=True),
            {
                "albedo": AlbedoConfig(
                    albedo_mode="constant_albedo",
                    facet_lat_count=18,
                    facet_lon_count=36,
                    enable_eclipse=False,
                )
            },
            "Albedo",
            1.0e-12,
            1.0e-6,
        ),
        (
            PerturbationFlags(enable_sh=False, enable_thermal=True),
            {
                "thermal": ThermalConfig(
                    thermal_mode="constant_temperature",
                    facet_lat_count=18,
                    facet_lon_count=36,
                    enable_eclipse=False,
                )
            },
            "Thermal IR",
            1.0e-13,
            1.0e-6,
        ),
    ],
)
@pytest.mark.parametrize("alt_km", ALTITUDES_KM)
def test_surface_radiation_budgets_are_finite_nonzero_and_plausible(
    alt_km: float,
    flags: PerturbationFlags,
    kwargs: dict[str, object],
    key: str,
    lo: float,
    hi: float,
) -> None:
    value = _breakdown(_engine(flags=flags, **kwargs), alt_km)[key]

    assert lo < value < hi


@pytest.mark.parametrize("alt_km", ALTITUDES_KM)
def test_solid_tide_budget_is_finite_nonzero_and_plausible(alt_km: float) -> None:
    engine = _engine(
        flags=PerturbationFlags(enable_sh=False, enable_tides_k2=True),
        solid_tides=SolidTideConfig(tide_bodies=("earth",), k2=0.02416),
    )
    comp = _breakdown(engine, alt_km)

    assert 1.0e-12 < comp["Solid Tides (Earth)"] < 1.0e-6
    assert comp["Solid Tides"] == pytest.approx(comp["Solid Tides (Earth)"])


@pytest.mark.parametrize("alt_km", ALTITUDES_KM)
def test_relativity_budget_is_finite_nonzero_and_flag_controlled(alt_km: float) -> None:
    engine = _engine(
        flags=PerturbationFlags(enable_sh=False, enable_relativity_1pn=True),
        ephem_manager=None,
    )
    comp = _breakdown(engine, alt_km)

    assert 1.0e-11 < comp["Relativity (Moon Schwarzschild)"] < 1.0e-6
    assert comp["Relativity (1PN)"] == pytest.approx(comp["Relativity (Moon Schwarzschild)"])

    disabled = _breakdown(
        _engine(flags=PerturbationFlags(enable_sh=False), ephem_manager=None),
        alt_km,
    )
    assert "Relativity (1PN)" not in disabled


def test_force_budget_breakdown_does_not_need_ad_hoc_dict_state() -> None:
    engine = _engine(
        flags=PerturbationFlags(enable_sh=False, enable_srp=True),
        srp=SRPConfig(),
    )
    comp = _breakdown(engine, ALTITUDES_KM[0])

    assert isinstance(comp, dict)
    assert all(isinstance(name, str) for name in comp)
    assert all(isinstance(value, float) for value in comp.values())
