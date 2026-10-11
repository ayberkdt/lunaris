"""Cached lunar RHS kernels: compiled once per process, numerics unchanged.

The CPU RHS kernels used to be ``@njit(cache=False)`` closures over the run's
arrays and settings, which numba freezes as compile-time constants: every new
``DynamicsEngine`` (one per CPU batch sample) recompiled them (measured
1.6-2.5 s per engine for GL1800F degree 50/200, SH-only and SH + Sun/Earth).
They are now module-level ``cache=True`` kernels that read the run data from
one ``_RhsPack``.

``GOLDEN_ACCEL`` was recorded with the closure implementation (commit
5ac4c37f, before the change) on the scenarios below; the cached kernels must
reproduce it. Comparison policy: on the recording platform (OS, machine,
numpy and numba versions) the RHS must be bit-identical; elsewhere libm and
instruction selection can move single evaluations by a few ULP, so the check
is ``rtol=1e-12`` (four orders of magnitude above that). The full old-vs-new
matrix on real data (GL1800F, SPICE ephemeris; 1000 states and a 1-day DOP853
trajectory per configuration) was run when the change was made.

Regenerate ``GOLDEN_ACCEL`` only for a deliberate, reviewed numerical change:
``python tests/test_dynamics_rhs_kernel_cache.py``.

The visible-disk quadrature became the default albedo / thermal-IR surface
integration after the recording, so the scenarios above pin
``surface_integration="facets"`` (the recorded path, still bit-identical). The
``*_quadrature`` scenarios were recorded after that change (same platform) and
guard the quadrature branch of the cached kernel.
"""

from __future__ import annotations

import math
import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numba
import numpy as np
import pytest

from lunaris.common.type_defs import (
    AdaptiveDegreeConfig,
    PerturbationFlags,
    SolidTideConfig,
    SpacecraftProps,
)
from lunaris.core.dynamics import DynamicsEngine
from lunaris.core.dynamics import engine as engine_module
from lunaris.physics.solar_effects import SRPConfig
from lunaris.physics.spherical_harmonics import GravityModel
from lunaris.physics.surface_effects import AlbedoConfig, ThermalConfig

R_REF = 1_738_000.0
GM = 4.9028e12
AU_M = 1.495978707e11


# =============================================================================
# Deterministic, data-free scenarios
# =============================================================================
class _SyntheticEphem:
    """Schema-v2 (Hermite) tables: circular Sun/Earth motion, steady Moon spin."""

    def __init__(self, n_rows: int = 40, dt_s: float = 60.0) -> None:
        t = np.arange(n_rows, dtype=np.float64) * dt_s
        ws, we, wm = 2.0e-7, 2.66e-6, 2.66e-6
        rs, re = AU_M, 3.844e8
        self._sun = np.stack([rs * np.cos(ws * t), rs * np.sin(ws * t), 0.05 * rs * np.sin(ws * t)], axis=1)
        self._sun_v = np.stack([-rs * ws * np.sin(ws * t), rs * ws * np.cos(ws * t), 0.05 * rs * ws * np.cos(ws * t)], axis=1)
        self._earth = np.stack([re * np.cos(we * t + 1.0), re * np.sin(we * t + 1.0), 0.09 * re * np.ones_like(t)], axis=1)
        self._earth_v = np.stack([-re * we * np.sin(we * t + 1.0), re * we * np.cos(we * t + 1.0), np.zeros_like(t)], axis=1)
        axis = np.array([0.1, 0.2, 0.97])
        axis /= np.linalg.norm(axis)
        half = 0.5 * (0.3 + wm * t)
        self._q = np.column_stack([np.cos(half), np.outer(np.sin(half), axis)])
        self._dt = dt_s

    def get_data_provider(self) -> dict[str, Any]:
        return {
            "dt_s": self._dt,
            "r_sun_tab_m": self._sun,
            "r_earth_tab_m": self._earth,
            "v_sun_tab_m_s": self._sun_v,
            "v_earth_tab_m_s": self._earth_v,
            "q_i2f_tab": self._q,
            "ephemeris_schema_version": 2,
        }


def _gravity() -> GravityModel:
    degree = 8
    c = np.zeros((degree + 1, degree + 1))
    s = np.zeros((degree + 1, degree + 1))
    for n in range(2, degree + 1):
        for m in range(n + 1):
            c[n, m] = 1.0e-5 * math.cos(1.3 * n + 0.7 * m) / n**2
            if m > 0:
                s[n, m] = 1.0e-5 * math.sin(0.9 * n - 0.4 * m) / n**2
    c[2, 0] = -9.09e-5
    return GravityModel.from_arrays(degree, R_REF, GM, c, s)


def _scenarios() -> dict[str, tuple[PerturbationFlags, dict[str, Any], bool]]:
    """name -> (flags, DynamicsEngine kwargs, 7-element mass state)."""
    grav = _gravity()
    eph = _SyntheticEphem()
    sb = {"enable_3rd_body_sun": True, "enable_3rd_body_earth": True}
    ej2 = SimpleNamespace(j2_coeff=1.0826e-3, r_eq_m=6_378_137.0, spin_axis_i=(0.0, 0.3, 0.95))
    tides = SolidTideConfig(k3=0.0085)
    n_cells = 18 * 36
    lat = np.linspace(89.5, -89.5, 180)[:, None]
    lon = np.linspace(0.5, 359.5, 360)[None, :]
    grid = 0.12 + 0.05 * np.cos(np.radians(lat)) * np.sin(np.radians(2.0 * lon))
    surf_grid = {
        "albedo_grid": grid, "n_lines": 180, "n_samples": 360, "res_deg": 1.0, "lon0_deg": 0.0, "lat0_deg": 90.0,
    }
    surf_temp = {"facet_temperatures_K": 100.0 + 250.0 * np.abs(np.sin(np.arange(n_cells) * 0.37))}
    adaptive_power = AdaptiveDegreeConfig(enabled=True, power=2.5, min_degree=3, quantization_step=2)
    adaptive_table = AdaptiveDegreeConfig(enabled=True, altitude_table=((100.0, 8), (500.0, 5), (5000.0, 3)))
    F = PerturbationFlags
    facets = {"surface_integration": "facets"}
    return {
        "sh_identity": (F(enable_sh=True), {"gravity_model": grav, "allow_identity_rotation": True}, False),
        "sh_rotation": (F(enable_sh=True), {"gravity_model": grav, "ephem_manager": eph}, False),
        "sh_rotation_mass7": (F(enable_sh=True), {"gravity_model": grav, "ephem_manager": eph}, True),
        "sh_adaptive_power": (
            F(enable_sh=True), {"gravity_model": grav, "ephem_manager": eph, "gravity_adaptive": adaptive_power}, False,
        ),
        "sh_adaptive_table": (
            F(enable_sh=True), {"gravity_model": grav, "allow_identity_rotation": True, "gravity_adaptive": adaptive_table},
            False,
        ),
        "point_mass": (F(enable_sh=False), {"allow_identity_rotation": True}, False),
        "point_mass_third_body": (F(enable_sh=False, **sb), {"ephem_manager": eph}, False),
        "sh_third_body": (F(enable_sh=True, **sb), {"gravity_model": grav, "ephem_manager": eph}, False),
        "sh_third_body_adaptive_table": (
            F(enable_sh=True, **sb), {"gravity_model": grav, "ephem_manager": eph, "gravity_adaptive": adaptive_table},
            False,
        ),
        "sh_earth_j2": (
            F(enable_sh=True, enable_earth_j2=True), {"gravity_model": grav, "ephem_manager": eph, "earth_j2": ej2}, False,
        ),
        "sh_srp_mass7": (
            F(enable_sh=True, enable_srp=True),
            {"gravity_model": grav, "ephem_manager": eph, "srp": SRPConfig(enable_earth_eclipse=True)},
            True,
        ),
        "sh_albedo_facets": (
            F(enable_sh=True, enable_albedo=True),
            {"gravity_model": grav, "ephem_manager": eph, "albedo": AlbedoConfig(**facets)},
            False,
        ),
        "sh_albedo_quadrature": (
            F(enable_sh=True, enable_albedo=True), {"gravity_model": grav, "ephem_manager": eph, "albedo": AlbedoConfig()},
            False,
        ),
        "sh_albedo_grid_quadrature": (
            F(enable_sh=True, enable_albedo=True),
            {
                "gravity_model": grav, "ephem_manager": eph, "surface_provider": surf_grid,
                "albedo": AlbedoConfig(albedo_mode="albedo_grid"),
            },
            False,
        ),
        "sh_albedo_simple_grid": (
            F(enable_sh=True, enable_albedo=True),
            {
                "gravity_model": grav, "ephem_manager": eph, "surface_provider": surf_grid,
                "albedo": AlbedoConfig(albedo_model="simple", albedo_mode="albedo_grid"),
            },
            False,
        ),
        "sh_thermal_equilibrium": (
            F(enable_sh=True, enable_thermal=True),
            {"gravity_model": grav, "ephem_manager": eph, "thermal": ThermalConfig(thermal_mode="equilibrium", **facets)},
            False,
        ),
        "sh_thermal_temperature_grid": (
            F(enable_sh=True, enable_thermal=True),
            {
                "gravity_model": grav, "ephem_manager": eph, "surface_provider": surf_temp,
                "thermal": ThermalConfig(thermal_mode="temperature_grid", **facets),
            },
            False,
        ),
        "sh_thermal_equilibrium_quadrature": (
            F(enable_sh=True, enable_thermal=True),
            {"gravity_model": grav, "ephem_manager": eph, "thermal": ThermalConfig(thermal_mode="equilibrium")},
            False,
        ),
        "sh_thermal_temperature_grid_quadrature": (
            F(enable_sh=True, enable_thermal=True),
            {
                "gravity_model": grav, "ephem_manager": eph, "surface_provider": surf_temp,
                "thermal": ThermalConfig(thermal_mode="temperature_grid"),
            },
            False,
        ),
        "sh_tides_k2_k3": (
            F(enable_sh=True, enable_tides_k2=True, enable_tides_k3=True),
            {"gravity_model": grav, "ephem_manager": eph, "solid_tides": tides},
            False,
        ),
        "sh_relativity_1pn": (
            F(enable_sh=True, enable_relativity_1pn=True, **sb), {"gravity_model": grav, "ephem_manager": eph}, False,
        ),
        "all_forces_mass7": (
            F(
                enable_sh=True, enable_earth_j2=True, enable_srp=True, enable_albedo=True, enable_thermal=True,
                enable_tides_k2=True, enable_tides_k3=True, enable_relativity_1pn=True, **sb,
            ),
            {
                "gravity_model": grav, "ephem_manager": eph, "earth_j2": ej2, "solid_tides": tides,
                "gravity_adaptive": adaptive_power, "thermal": ThermalConfig(**facets),
                "albedo": AlbedoConfig(albedo_model="simple", albedo_mode="albedo_grid"), "surface_provider": surf_grid,
            },
            True,
        ),
    }


SC = SpacecraftProps(mass_kg=850.0, area_m2=4.2, cr=1.3)
# (t_s, y): low/medium/high orbits, one above the pole, one in eclipse.
STATES: tuple[tuple[float, tuple[float, ...]], ...] = (
    (0.0, (R_REF + 30e3, 1.0e4, -2.0e4, 15.0, 1650.0, 120.0)),
    (431.7, (-1.2e6, 1.1e6, 0.6e6, -900.0, -1000.0, 450.0)),
    (1234.5, (2.0e4, -3.0e4, R_REF + 80e3, 1600.0, 40.0, -5.0)),
    (1999.0, (-(R_REF + 400e3), 2.0e5, -1.0e5, 100.0, -1400.0, 300.0)),
)
STATE_MASS_KG = 640.0


def _state(y: tuple[float, ...], mass7: bool) -> np.ndarray:
    return np.array((*y, STATE_MASS_KG) if mass7 else y, dtype=np.float64)


def _evaluate(engine_cls: Any = DynamicsEngine) -> dict[str, list[list[float]]]:
    """Acceleration (dydt[3:6]) of every scenario at every state."""
    out: dict[str, list[list[float]]] = {}
    for name, (flags, kwargs, mass7) in _scenarios().items():
        rhs = engine_cls(SC, flags, **kwargs)._build_solver_rhs()
        out[name] = [[float(v) for v in rhs(t, _state(y, mass7))[3:6]] for t, y in STATES]
    return out


# Recorded with the closure kernels at commit 5ac4c37f (see module docstring).
GOLDEN_PLATFORM = ("Windows", "AMD64", "2.2.6", "0.64.0")
GOLDEN_ACCEL: dict[str, list[list[str]]] = {
    "all_forces_mass7": [
        ["-0x1.918ecd534cb4ep+0", "-0x1.21aaa8c361602p-7", "0x1.23df21aeaf466p-6"],
        ["0x1.2072329539b2bp+0", "-0x1.08669dac00125p+0", "-0x1.20a1377eec888p-1"],
        ["-0x1.0a3b0e88d0cc4p-6", "0x1.901af42f89481p-6", "-0x1.7b51e32cedd43p+0"],
        ["0x1.0e300c1ffe339p+0", "-0x1.94819ca582e23p-4", "0x1.946da6f120921p-5"],
    ],
    "point_mass": [
        ["-0x1.916f686946689p+0", "-0x1.22a1be4544599p-7", "0x1.22a1be4544599p-6"],
        ["0x1.2069de3964aabp+0", "-0x1.08610bb49c472p+0", "-0x1.2069de3964aabp-1"],
        ["-0x1.0b3653aca4b2bp-6", "0x1.90d17d82f70c0p-6", "-0x1.7b865c719b85cp+0"],
        ["0x1.0e22b1a7d0ae5p+0", "-0x1.94519f55b1fdfp-4", "0x1.94519f55b1fdfp-5"],
    ],
    "point_mass_third_body": [
        ["-0x1.916f80ff563dfp+0", "-0x1.2215f8c3cd765p-7", "0x1.22a9d6b44784ap-6"],
        ["0x1.206aa627a5d66p+0", "-0x1.08612c8e8aaafp+0", "-0x1.206a566862a04p-1"],
        ["-0x1.0b30296348b8ep-6", "0x1.90dcf16869c49p-6", "-0x1.7b872cd822576p+0"],
        ["0x1.0e22eed54ee48p+0", "-0x1.946513008dd22p-4", "0x1.944f3c73db451p-5"],
    ],
    "sh_adaptive_power": [
        ["-0x1.918eb4c3ca69fp+0", "-0x1.223993d453b78p-7", "0x1.23d6d92920b2ap-6"],
        ["0x1.20716d56acf55p+0", "-0x1.086682f9053cap+0", "-0x1.20a0c3f488809p-1"],
        ["-0x1.0a412f0d62fe7p-6", "0x1.900f5024fca3ap-6", "-0x1.7b5118dfd514ep+0"],
        ["0x1.0e2fd059fbb9ap+0", "-0x1.946e024774fc1p-4", "0x1.947011cced653p-5"],
    ],
    "sh_adaptive_table": [
        ["-0x1.918d51965df49p+0", "-0x1.21d87ddf3a615p-7", "0x1.22c2043c9ea98p-6"],
        ["0x1.20750784b2ee5p+0", "-0x1.086a2b4f8a050p+0", "-0x1.209f3256a8780p-1"],
        ["-0x1.0b092e4df1d05p-6", "0x1.90b2431795b6dp-6", "-0x1.7b4fd0a84bc7cp+0"],
        ["0x1.0e2f3008f3b03p+0", "-0x1.9466d1f36152ap-4", "0x1.94a41c911a287p-5"],
    ],
    "sh_albedo_facets": [
        ["-0x1.918ed3544d8b9p+0", "-0x1.221b3326fbe24p-7", "0x1.23dc5206caf17p-6"],
        ["0x1.20716d56acf55p+0", "-0x1.086682f9053cap+0", "-0x1.20a0c3f488809p-1"],
        ["-0x1.0a40a9e635d63p-6", "0x1.900778ca1cc50p-6", "-0x1.7b50656ef3ca9p+0"],
        ["0x1.0e2fad8306494p+0", "-0x1.946b9e8f182fdp-4", "0x1.947086c9009d9p-5"],
    ],
    "sh_albedo_grid_quadrature": [
        ["-0x1.918ed34dcb97fp+0", "-0x1.221b3a62c7b58p-7", "0x1.23dc502c3af3bp-6"],
        ["0x1.20716d56acf55p+0", "-0x1.086682f9053cap+0", "-0x1.20a0c3f488809p-1"],
        ["-0x1.0a40a9e2ae99bp-6", "0x1.900778c97662ap-6", "-0x1.7b50656f0a6cap+0"],
        ["0x1.0e2fad8306494p+0", "-0x1.946b9e8f182fdp-4", "0x1.947086c9009d9p-5"],
    ],
    "sh_albedo_quadrature": [
        ["-0x1.918ed350345c5p+0", "-0x1.221b3a58eefdfp-7", "0x1.23dc502da82bfp-6"],
        ["0x1.20716d56acf55p+0", "-0x1.086682f9053cap+0", "-0x1.20a0c3f488809p-1"],
        ["-0x1.0a40a9e2bf539p-6", "0x1.900778ca21924p-6", "-0x1.7b50656f0992ap+0"],
        ["0x1.0e2fad8306494p+0", "-0x1.946b9e8f182fdp-4", "0x1.947086c9009d9p-5"],
    ],
    "sh_albedo_simple_grid": [
        ["-0x1.918ed36e02624p+0", "-0x1.221b3a6136e23p-7", "0x1.23dc5035f0103p-6"],
        ["0x1.20716d56acf55p+0", "-0x1.086682f9053cap+0", "-0x1.20a0c3f488809p-1"],
        ["-0x1.0a40ad782a11cp-6", "0x1.900778ca2c6bep-6", "-0x1.7b50656f4acacp+0"],
        ["0x1.0e2fad8306494p+0", "-0x1.946b9e8f182fdp-4", "0x1.947086c9009d9p-5"],
    ],
    "sh_earth_j2": [
        ["-0x1.918ed35b5f262p+0", "-0x1.221b3a5e45af9p-7", "0x1.23dc5038a8efcp-6"],
        ["0x1.20716d56b4e35p+0", "-0x1.086682f8fdf81p+0", "-0x1.20a0c3f48cc13p-1"],
        ["-0x1.0a40a9ce6ea6bp-6", "0x1.900778cde34d9p-6", "-0x1.7b50656f55462p+0"],
        ["0x1.0e2fad8307612p+0", "-0x1.946b9e8f8344fp-4", "0x1.947086c7abb68p-5"],
    ],
    "sh_identity": [
        ["-0x1.918d57f03bd51p+0", "-0x1.21c240f7b33f3p-7", "0x1.22c94df654b60p-6"],
        ["0x1.20766314157e4p+0", "-0x1.086a9bd2f7090p+0", "-0x1.209f72e7dc1a5p-1"],
        ["-0x1.0b143dc9dc32bp-6", "0x1.9086a398161fep-6", "-0x1.7b4fdc14a2501p+0"],
        ["0x1.0e2f35d719566p+0", "-0x1.9466f0831c88fp-4", "0x1.94a19b691fec0p-5"],
    ],
    "sh_relativity_1pn": [
        ["-0x1.918eebf0dfd7fp+0", "-0x1.218f74ddd1d4ep-7", "0x1.23e468a48bb9ap-6"],
        ["0x1.207235446e607p+0", "-0x1.0866a3d297f6bp+0", "-0x1.20a13c22fd0a8p-1"],
        ["-0x1.0a3a7f874cf89p-6", "0x1.9012ecb569037p-6", "-0x1.7b5135d53b383p+0"],
        ["0x1.0e2feab037953p+0", "-0x1.947f123839d09p-4", "0x1.946e23e62c14bp-5"],
    ],
    "sh_rotation": [
        ["-0x1.918ed35b5e35fp+0", "-0x1.221b3a6136ecap-7", "0x1.23dc5035f01aap-6"],
        ["0x1.20716d56acf55p+0", "-0x1.086682f9053cap+0", "-0x1.20a0c3f488809p-1"],
        ["-0x1.0a40a9d11e953p-6", "0x1.900778ca67920p-6", "-0x1.7b50656f4aca9p+0"],
        ["0x1.0e2fad8306494p+0", "-0x1.946b9e8f182fdp-4", "0x1.947086c9009d9p-5"],
    ],
    "sh_rotation_mass7": [
        ["-0x1.918ed35b5e35fp+0", "-0x1.221b3a6136ecap-7", "0x1.23dc5035f01aap-6"],
        ["0x1.20716d56acf55p+0", "-0x1.086682f9053cap+0", "-0x1.20a0c3f488809p-1"],
        ["-0x1.0a40a9d11e953p-6", "0x1.900778ca67920p-6", "-0x1.7b50656f4aca9p+0"],
        ["0x1.0e2fad8306494p+0", "-0x1.946b9e8f182fdp-4", "0x1.947086c9009d9p-5"],
    ],
    "sh_srp_mass7": [
        ["-0x1.918ed401b78e9p+0", "-0x1.221b3a61368f6p-7", "0x1.23dc5035efbd6p-6"],
        ["0x1.20716d56acf55p+0", "-0x1.086682f9053cap+0", "-0x1.20a0c3f488809p-1"],
        ["-0x1.0a40d36734f7fp-6", "0x1.900778c7c61f7p-6", "-0x1.7b50656f4accbp+0"],
        ["0x1.0e2fad8306494p+0", "-0x1.946b9e8f182fdp-4", "0x1.947086c9009d9p-5"],
    ],
    "sh_thermal_equilibrium": [
        ["-0x1.918ed327533fep+0", "-0x1.221b05241868fp-7", "0x1.23dc5d9624022p-6"],
        ["0x1.20716d56acf55p+0", "-0x1.086682f9053cap+0", "-0x1.20a0c3f488809p-1"],
        ["-0x1.0a40aa7328588p-6", "0x1.900778ce40357p-6", "-0x1.7b50656c80992p+0"],
        ["0x1.0e2fad82c719ap+0", "-0x1.946b9e8eb692ep-4", "0x1.947086c89e37ep-5"],
    ],
    "sh_thermal_equilibrium_quadrature": [
        ["-0x1.918ed30922d0ap+0", "-0x1.221b3a2438d9ap-7", "0x1.23dc4ff8f207ap-6"],
        ["0x1.20716d56acf55p+0", "-0x1.086682f9053cap+0", "-0x1.20a0c3f488809p-1"],
        ["-0x1.0a40aa5229f33p-6", "0x1.900778c807ecdp-6", "-0x1.7b50656d1336ep+0"],
        ["0x1.0e2fad82c6d1fp+0", "-0x1.946b9e8eb9320p-4", "0x1.947086c8a19fcp-5"],
    ],
    "sh_thermal_temperature_grid": [
        ["-0x1.918ed35aa21e4p+0", "-0x1.221b39c98fae3p-7", "0x1.23dc50508cd9cp-6"],
        ["0x1.20716d56acf55p+0", "-0x1.086682f9053cap+0", "-0x1.20a0c3f488809p-1"],
        ["-0x1.0a40a9c1c6db8p-6", "0x1.900779fd8e28fp-6", "-0x1.7b5065642ce0ap+0"],
        ["0x1.0e2fad728071ep+0", "-0x1.946b9e56f1e0bp-4", "0x1.94708625a464cp-5"],
    ],
    "sh_thermal_temperature_grid_quadrature": [
        ["-0x1.918ed359f20ebp+0", "-0x1.221b3aac06c7ep-7", "0x1.23dc4ffe1926dp-6"],
        ["0x1.20716d56acf55p+0", "-0x1.086682f9053cap+0", "-0x1.20a0c3f488809p-1"],
        ["-0x1.0a40a8ab77f7bp-6", "0x1.900778fc9d62bp-6", "-0x1.7b5065614e87fp+0"],
        ["0x1.0e2fad72848edp+0", "-0x1.946b9e5c435f6p-4", "0x1.94708638db824p-5"],
    ],
    "sh_third_body": [
        ["-0x1.918eebf16e0b5p+0", "-0x1.218f74dfc0096p-7", "0x1.23e468a4f345bp-6"],
        ["0x1.20723544ee210p+0", "-0x1.0866a3d2f3a07p+0", "-0x1.20a13c2386762p-1"],
        ["-0x1.0a3a7f87c29b6p-6", "0x1.9012ecafda4a9p-6", "-0x1.7b5135d5d19c3p+0"],
        ["0x1.0e2feab0847f7p+0", "-0x1.947f1239f4040p-4", "0x1.946e23e729e4bp-5"],
    ],
    "sh_third_body_adaptive_table": [
        ["-0x1.918e862294909p+0", "-0x1.219a800480f54p-7", "0x1.23c99134e67aap-6"],
        ["0x1.207198afbcea1p+0", "-0x1.08662c0e3d907p+0", "-0x1.20a147ab360efp-1"],
        ["-0x1.0a2428a5d340dp-6", "0x1.9045e76b2c9cep-6", "-0x1.7b5128744c414p+0"],
        ["0x1.0e300d8779efdp+0", "-0x1.948175f250d04p-4", "0x1.946daeeb16ac5p-5"],
    ],
    "sh_tides_k2_k3": [
        ["-0x1.918ed2a25ce69p+0", "-0x1.221821373922ap-7", "0x1.23dc7d2e005d4p-6"],
        ["0x1.20716aa770599p+0", "-0x1.08667cd2749cdp+0", "-0x1.20a0bf5073bdfp-1"],
        ["-0x1.0a4083ac08800p-6", "0x1.9007a7b32e874p-6", "-0x1.7b505f65d0d43p+0"],
        ["0x1.0e2fac28a4766p+0", "-0x1.946bc55761883p-4", "0x1.94707ee54e0f6p-5"],
    ],
}


def _platform_fingerprint() -> tuple[str, str, str, str]:
    return (platform.system(), platform.machine(), np.__version__, numba.__version__)


# =============================================================================
# Tests
# =============================================================================
def test_cache_guard_tracks_every_callee_module() -> None:
    files, frozen = engine_module._rhs_kernel_dependencies()
    names = {Path(f).name for f in files}
    assert {
        "math_utils.py", "adaptive_degree.py", "ephemeris.py", "lunar_albedo.py", "relativity_effects.py",
        "solar_effects.py", "solid_tides.py", "spherical_harmonics.py", "surface_effects.py", "thermal_ir.py",
        "third_body_effects.py", "surface_radiation_quadrature.py",
    } <= names
    assert "engine.py" not in names  # numba itself validates against engine.py
    assert any(entry.startswith("lunaris.physics.relativity_effects.C_SQ=") for entry in frozen)


def _fake_cache(cache_dir: Path) -> tuple[list[Path], list[Path]]:
    stale = [
        cache_dir / "engine._rhs_kernel_numba-291.py310.nbi",
        cache_dir / "engine._rhs_kernel_numba-291.py310.1.nbc",
        cache_dir / "engine._rhs_sh_only_numba-217.py310.nbi",
    ]
    keep = [cache_dir / "engine.cpython-310.pyc", cache_dir / "math_utils.clamp-104.py310.nbi"]
    for path in stale + keep:
        path.write_bytes(b"x")
    return stale, keep


def test_cache_guard_removes_engine_cache_when_a_callee_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    callee = tmp_path / "callee_module.py"
    callee.write_text("A = 1" + chr(10), encoding="utf-8")
    real = engine_module._rhs_kernel_dependencies

    def with_fake_callee() -> tuple[list[str], list[str]]:
        files, frozen = real()
        return [*files, str(callee)], frozen

    monkeypatch.setattr(engine_module, "_rhs_kernel_dependencies", with_fake_callee)
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    stale, keep = _fake_cache(cache_dir)

    # No stamp yet: the cache's provenance is unknown, so it is dropped once.
    removed = engine_module._refresh_stale_rhs_kernel_cache([cache_dir])
    assert sorted(removed) == sorted(stale)
    stamp = cache_dir / engine_module._RHS_DEPS_STAMP_NAME
    first_digest = stamp.read_text(encoding="utf-8").strip()

    # Same callee sources: a freshly written cache is kept.
    stale, keep = _fake_cache(cache_dir)
    assert engine_module._refresh_stale_rhs_kernel_cache([cache_dir]) == []
    assert all(p.exists() for p in stale + keep)

    # A callee's source changes: engine.py's cache files go, nothing else does.
    callee.write_text("A = 2" + chr(10), encoding="utf-8")
    removed = engine_module._refresh_stale_rhs_kernel_cache([cache_dir])
    assert sorted(removed) == sorted(stale)
    assert not any(p.exists() for p in stale)
    assert all(p.exists() for p in keep)
    assert stamp.read_text(encoding="utf-8").strip() != first_digest


def test_cache_guard_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("", encoding="utf-8")
    assert engine_module._refresh_stale_rhs_kernel_cache([not_a_dir / "sub"], digest="abc") == []

    def broken() -> tuple[list[str], list[str]]:
        raise RuntimeError("unreadable source")

    monkeypatch.setattr(engine_module, "_rhs_kernel_dependencies", broken)
    assert engine_module._refresh_stale_rhs_kernel_cache([tmp_path]) == []


def test_kernels_are_module_level_and_disk_cached() -> None:
    for kernel in (engine_module._rhs_kernel_numba, engine_module._rhs_sh_only_numba, engine_module._new_rhs_pack):
        assert kernel.py_func.__qualname__ == kernel.py_func.__name__  # not a closure
        assert type(kernel._cache).__name__ != "NullCache"  # cache=True


@pytest.mark.parametrize(
    ("kernel_name", "path", "flags_a", "flags_b"),
    [
        ("_rhs_sh_only_numba", "sh_only_numba", PerturbationFlags(enable_sh=True), PerturbationFlags(enable_sh=True)),
        (
            "_rhs_kernel_numba",
            "general_numba",
            PerturbationFlags(enable_sh=True, enable_3rd_body_sun=True, enable_3rd_body_earth=True),
            PerturbationFlags(enable_sh=True, enable_3rd_body_earth=True, enable_srp=True, enable_tides_k2=True),
        ),
    ],
)
def test_second_engine_reuses_the_compiled_kernel(
    kernel_name: str, path: str, flags_a: PerturbationFlags, flags_b: PerturbationFlags
) -> None:
    """Different run values and force selections, same types: no recompilation."""
    kernel = getattr(engine_module, kernel_name)
    t, y = STATES[0]
    y6 = _state(y, False)
    first = DynamicsEngine(SC, flags_a, gravity_model=_gravity(), ephem_manager=_SyntheticEphem())
    a = first._build_solver_rhs()(t, y6)
    assert first._prep["rhs_path"] == path
    signatures = list(kernel.signatures)
    pack_signatures = list(engine_module._new_rhs_pack.signatures)

    other_sc = SpacecraftProps(mass_kg=120.0, area_m2=1.0, cr=1.1)
    second = DynamicsEngine(other_sc, flags_b, gravity_model=_gravity(), ephem_manager=_SyntheticEphem(n_rows=55))
    b = second._build_solver_rhs()(t, y6)
    assert second._prep["rhs_path"] == path
    assert list(kernel.signatures) == signatures
    assert list(engine_module._new_rhs_pack.signatures) == pack_signatures
    if flags_a != flags_b:
        assert not np.array_equal(a, b)  # each engine evaluates its own values


@pytest.mark.parametrize("name", sorted(_scenarios()))
def test_rhs_matches_golden_values(name: str) -> None:
    assert GOLDEN_ACCEL, "GOLDEN_ACCEL is empty; regenerate it from the reference implementation."
    flags, kwargs, mass7 = _scenarios()[name]
    rhs = DynamicsEngine(SC, flags, **kwargs)._build_solver_rhs()
    got = np.array([rhs(t, _state(y, mass7)) for t, y in STATES])
    expected = np.array([[float.fromhex(v) for v in row] for row in GOLDEN_ACCEL[name]])
    assert np.array_equal(got[:, :3], np.array([y[3:6] for _, y in STATES]))
    if mass7:
        assert np.all(got[:, 6] == 0.0)
    if _platform_fingerprint() == GOLDEN_PLATFORM:
        assert np.array_equal(got[:, 3:6], expected), f"{name}: RHS is no longer bit-identical"
    else:
        np.testing.assert_allclose(got[:, 3:6], expected, rtol=1e-12, atol=0.0)


if __name__ == "__main__":  # pragma: no cover - golden regeneration helper
    import pprint

    print(_platform_fingerprint())
    pprint.pprint(
        {k: [[v.hex() for v in row] for row in rows] for k, rows in _evaluate().items()}, width=110, sort_dicts=True
    )
