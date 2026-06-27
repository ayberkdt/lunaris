from __future__ import annotations

import importlib

import numpy as np

from lunaris.common.constants import MU_MOON
from lunaris.common.type_defs import PerturbationFlags, SpacecraftProps


def test_dynamics_package_exports_legacy_surface() -> None:
    dynamics = importlib.import_module("lunaris.core.dynamics")
    engine = importlib.import_module("lunaris.core.dynamics.engine")
    requirements = importlib.import_module("lunaris.core.dynamics.requirements")
    adaptive = importlib.import_module("lunaris.core.dynamics.adaptive_degree")
    packs = importlib.import_module("lunaris.core.dynamics.perturbation_packs")

    assert dynamics.DynamicsEngine is engine.DynamicsEngine
    assert callable(requirements.extract_gravity_strict)
    assert callable(adaptive._select_adaptive_sh_degree)
    assert hasattr(packs, "_AlbedoPack")


def test_point_mass_rhs_golden_after_dynamics_package_split() -> None:
    from lunaris.core.dynamics import DynamicsEngine

    flags = PerturbationFlags(
        enable_sh=False,
        enable_3rd_body_sun=False,
        enable_3rd_body_earth=False,
        enable_srp=False,
        enable_albedo=False,
        enable_relativity_1pn=False,
        enable_earth_j2=False,
    )
    engine = DynamicsEngine(
        sc_props=SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3),
        flags=flags,
        gravity_model=None,
        ephem_manager=None,
        surface_provider=None,
        earth_j2=None,
        allow_identity_rotation=True,
    )
    rhs = engine.build_rhs(force_rebuild=True)
    y = np.asarray([1_837_400.0, 0.0, 0.0, 0.0, 1_600.0, 0.0], dtype=np.float64)

    dy = rhs(0.0, y)
    expected_ax = -MU_MOON / (y[0] ** 2)

    np.testing.assert_allclose(dy[:3], y[3:], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(dy[3], expected_ax, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(dy[4:], [0.0, 0.0], rtol=0.0, atol=1e-15)
