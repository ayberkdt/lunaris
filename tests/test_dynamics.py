# tests/test_dynamics.py
from __future__ import annotations

import math
import os

import numpy as np
import pytest

# -----------------------------------------------------------------------------
# Imports (skip cleanly if the package layout isn't available in this context)
# -----------------------------------------------------------------------------
try:
    from lunaris.core.dynamics import DynamicsEngine, extract_ephem_tables_strict
except Exception as e:  # pragma: no cover
    pytest.skip(f"core.dynamics not importable: {e}", allow_module_level=True)

try:
    from lunaris.common.constants import AU, MU_MOON, R_MOON_MEAN
    from lunaris.common.math_utils import quat_rotate_np
    from lunaris.common.type_defs import PerturbationFlags, SpacecraftProps
    from lunaris.physics.solar_effects import SRPConfig
    from lunaris.physics.spherical_harmonics import GravityModel, compute_point_mass_acceleration
except Exception as e:  # pragma: no cover
    pytest.skip(f"common.type_defs not importable: {e}", allow_module_level=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _norm3(x: float, y: float, z: float) -> float:
    return float(math.sqrt(x * x + y * y + z * z))


def _build_default_state(*, r_km: float = 1837.4, v_ms: float = 1600.0) -> np.ndarray:
    """Simple planar state: r=[r,0,0], v=[0,v,0] in SI units."""
    r_m = float(r_km) * 1000.0
    y = np.zeros(6, dtype=np.float64)
    y[0] = r_m
    y[4] = float(v_ms)
    return y


@pytest.fixture(scope="module")
def engine_point_mass() -> tuple[DynamicsEngine, callable]:
    """
    Build a minimal engine configuration with point-mass gravity only.
    Keeps compilation cost to a minimum by reusing the same RHS for the module.
    """
    sc = SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3)

    # Everything disabled => ephemeris/gravity model not required.
    flags = PerturbationFlags(
        enable_sh=False,
        enable_3rd_body_sun=False,
        enable_3rd_body_earth=False,
        enable_srp=False,
        enable_albedo=False,
        enable_relativity_1pn=False,
        enable_earth_j2=False,
    )

    eng = DynamicsEngine(
        sc_props=sc,
        flags=flags,
        gravity_model=None,
        ephem_manager=None,
        surface_provider=None,
        earth_j2=None,
        allow_identity_rotation=True,  # OK since SH/albedo are disabled.
    )

    rhs = eng.build_rhs(force_rebuild=True)

    # Warm-up (Numba compile) once at module scope.
    y0 = _build_default_state()
    _ = rhs(0.0, y0)

    return eng, rhs


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_rhs_shape_and_inward_acceleration(engine_point_mass: tuple[DynamicsEngine, callable]) -> None:
    eng, rhs = engine_point_mass

    t0 = 0.0
    y0 = _build_default_state()

    dy0 = rhs(t0, y0)
    assert dy0.shape == y0.shape

    ax, ay, az = float(dy0[3]), float(dy0[4]), float(dy0[5])
    a_norm = _norm3(ax, ay, az)
    assert math.isfinite(a_norm) and a_norm > 0.0

    r = y0[0:3]
    r_norm = float(np.linalg.norm(r))
    assert r_norm > 0.0

    # For point-mass gravity only, acceleration should point roughly inward: a · r < 0
    dot_ar = float(ax * r[0] + ay * r[1] + az * r[2])
    assert dot_ar < 0.0


def test_one_step_consistency_smoke(engine_point_mass: tuple[DynamicsEngine, callable]) -> None:
    _, rhs = engine_point_mass

    t0 = 0.0
    dt = 1.0
    y0 = _build_default_state()

    dy0 = rhs(t0, y0)
    y1 = y0 + dt * dy0
    dy1 = rhs(t0 + dt, y1)

    assert dy1.shape == y0.shape
    assert np.all(np.isfinite(dy1))

    # Algebraic identity check (should be ~0, only roundoff left)
    dv_res = float(np.linalg.norm((y1[3:6] - y0[3:6]) - dt * dy0[3:6]))
    assert dv_res < 1e-10


def test_extract_ephem_tables_accepts_constant_vector_rows_with_full_quaternion_timeline() -> None:
    class _MockEphem:
        def get_data_provider(self):
            return {
                "dt_s": 60.0,
                "r_sun_tab_m": np.zeros((1, 3), dtype=np.float64),
                "r_earth_tab_m": np.zeros((1, 3), dtype=np.float64),
                "q_i2f_tab": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (8, 1)),
            }

    dt_s, sun_tab, earth_tab, q_tab = extract_ephem_tables_strict(_MockEphem())

    assert dt_s == 60.0
    assert sun_tab.shape == (1, 3)
    assert earth_tab.shape == (1, 3)
    assert q_tab.shape == (8, 4)


def test_acceleration_breakdown_smoke(engine_point_mass: tuple[DynamicsEngine, callable]) -> None:
    eng, _ = engine_point_mass

    if not hasattr(eng, "get_acceleration_breakdown"):
        pytest.skip("DynamicsEngine.get_acceleration_breakdown not available in this build.")

    y0 = _build_default_state()
    comp = eng.get_acceleration_breakdown(0.0, y0)

    assert isinstance(comp, dict)
    assert len(comp) >= 1

    # Expect at least a gravity term in minimal config.
    has_gravity = any("gravity" in k.lower() for k in comp.keys())
    assert has_gravity

    # All norms should be finite and non-negative.
    for _k, v in comp.items():
        assert math.isfinite(float(v))
        assert float(v) >= 0.0


def test_srp_config_controls_dynamics_eclipse() -> None:
    class _Ephem:
        def get_data_provider(self):
            return {
                "dt_s": 60.0,
                "r_sun_tab_m": np.tile(np.array([AU, 0.0, 0.0], dtype=np.float64), (2, 1)),
                "r_earth_tab_m": np.zeros((2, 3), dtype=np.float64),
                "q_i2f_tab": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (2, 1)),
            }

    flags = PerturbationFlags(enable_sh=False, enable_srp=True)
    sc = SpacecraftProps(mass_kg=12.0, area_m2=2.0, cr=1.8, cd=2.2)
    r = np.array([-(R_MOON_MEAN + 50e3), 0.0, 0.0], dtype=np.float64)
    y = np.r_[r, [0.0, 0.0, 0.0]]
    point_mass = np.array(
        compute_point_mass_acceleration(r[0], r[1], r[2], MU_MOON),
        dtype=np.float64,
    )

    default = DynamicsEngine(
        sc_props=sc,
        flags=flags,
        ephem_manager=_Ephem(),
        srp=SRPConfig(),
        allow_identity_rotation=True,
    )
    default_rhs = default.build_rhs(force_rebuild=True)
    default_srp = np.asarray(default_rhs(0.0, y), dtype=np.float64)[3:6] - point_mass

    np.testing.assert_allclose(default_srp, 0.0, rtol=0.0, atol=0.0)
    assert default.get_acceleration_breakdown(0.0, y)["SRP"] == pytest.approx(0.0)

    no_shadow = DynamicsEngine(
        sc_props=sc,
        flags=flags,
        ephem_manager=_Ephem(),
        srp=SRPConfig(enable_moon_eclipse=False),
        allow_identity_rotation=True,
    )
    no_shadow_rhs = no_shadow.build_rhs(force_rebuild=True)
    no_shadow_srp = np.asarray(no_shadow_rhs(0.0, y), dtype=np.float64)[3:6] - point_mass

    assert float(np.linalg.norm(no_shadow_srp)) > 0.0
    assert no_shadow.get_acceleration_breakdown(0.0, y)["SRP"] > 0.0


def test_surrogate_gravity_provider_can_drive_python_rhs() -> None:
    class _StubSurrogateGravity:
        model_kind = "st_lrps"
        R_ref_m = 1_737_400.0
        GM_m3s2 = 4.9048695e12

        def surrogate_forward(self, *args, **kwargs):
            return np.array([-1.25, 0.0, 0.0], dtype=np.float64)

        def acceleration_fixed(self, _r_fixed):
            return np.array([-1.25, 0.0, 0.0], dtype=np.float64)

    sc = SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3)
    flags = PerturbationFlags(enable_sh=True)
    eng = DynamicsEngine(
        sc_props=sc,
        flags=flags,
        gravity_model=_StubSurrogateGravity(),
        ephem_manager=None,
        surface_provider=None,
        earth_j2=None,
        allow_identity_rotation=True,
    )

    rhs = eng.build_rhs(force_rebuild=True)
    y0 = _build_default_state()
    dy0 = rhs(0.0, y0)

    assert dy0.shape == y0.shape
    assert float(dy0[3]) == pytest.approx(-1.25)
    assert float(dy0[4]) == pytest.approx(0.0)
    assert float(dy0[5]) == pytest.approx(0.0)

    comp = eng.get_acceleration_breakdown(0.0, y0)
    assert "Gravity (ST-LRPS)" in comp


def test_cpu_sh_rhs_uses_i2f_then_conjugate_frame_bridge() -> None:
    """Classical CPU SH must evaluate in fixed frame and rotate acceleration back."""

    class _ConstantQuaternionEphem:
        def __init__(self, q_i2f: np.ndarray) -> None:
            self.q_i2f = np.asarray(q_i2f, dtype=np.float64)

        def get_data_provider(self):
            zeros = np.zeros((1, 3), dtype=np.float64)
            return {
                "dt_s": 1.0,
                "r_sun_tab_m": zeros,
                "r_earth_tab_m": zeros,
                "q_i2f_tab": np.vstack([self.q_i2f, self.q_i2f]),
            }

    degree = 4
    r_ref = 1_737_400.0
    gm = 4.904_869_5e12
    c = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    s = np.zeros_like(c)
    c[2, 0] = -9.0e-5
    c[2, 2] = 1.5e-5
    s[3, 1] = -2.0e-6
    c[4, 3] = 7.5e-7
    gravity = GravityModel.from_arrays(degree, r_ref, gm, c, s)

    c45 = math.sqrt(0.5)
    q_i2f = np.array([c45, 0.0, 0.0, c45], dtype=np.float64)
    q_f2i = np.array([q_i2f[0], -q_i2f[1], -q_i2f[2], -q_i2f[3]], dtype=np.float64)
    state = np.array(
        [r_ref + 160_000.0, 120_000.0, -75_000.0, 12.0, 1550.0, -30.0],
        dtype=np.float64,
    )

    engine = DynamicsEngine(
        sc_props=SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3),
        flags=PerturbationFlags(enable_sh=True),
        gravity_model=gravity,
        ephem_manager=_ConstantQuaternionEphem(q_i2f),
        surface_provider=None,
        earth_j2=None,
        allow_identity_rotation=False,
    )
    rhs = engine.build_rhs(force_rebuild=True)
    dy = np.asarray(rhs(0.0, state), dtype=np.float64)

    fixed_position = quat_rotate_np(q_i2f, state[:3])
    fixed_accel = gravity.accel_fixed(fixed_position, degree=degree)
    expected_inertial_accel = quat_rotate_np(q_f2i, fixed_accel)

    np.testing.assert_allclose(dy[:3], state[3:], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(dy[3:6], expected_inertial_accel, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(os.getenv("RUN_SLOW") != "1", reason="Set RUN_SLOW=1 to run slow integration test.")
def test_solve_ivp_mini_run(engine_point_mass: tuple[DynamicsEngine, callable]) -> None:
    pytest.importorskip("scipy")
    from scipy.integrate import solve_ivp  # type: ignore

    _, rhs = engine_point_mass
    t0 = 0.0
    tf = 10.0
    y0 = _build_default_state()

    sol = solve_ivp(rhs, (t0, tf), y0, rtol=1e-9, atol=1e-12, max_step=1.0)

    assert sol.status in (0, 1)  # 0: success, 1: terminated (shouldn't happen here)
    assert sol.t.size >= 2

    y_end = sol.y[:, -1]
    assert y_end.shape[0] == 6
    assert np.all(np.isfinite(y_end))


# -----------------------------------------------------------------------------
# Fail-closed guard: degenerate (all-zero) Sun/Earth ephemeris tables
# -----------------------------------------------------------------------------
class _ZeroBodyEphem:
    """Ephemeris stub: valid quaternion timeline, all-zero Sun/Earth tables
    (what build_tables produces with include_third_body=False)."""

    def __init__(self, sun=None, earth=None, n: int = 4) -> None:
        z = np.zeros((1, 3), dtype=np.float64)
        self._d = {
            "dt_s": 60.0,
            "r_sun_tab_m": (np.tile(np.asarray(sun, dtype=np.float64), (n, 1)) if sun is not None else z),
            "r_earth_tab_m": (np.tile(np.asarray(earth, dtype=np.float64), (n, 1)) if earth is not None else z),
            "q_i2f_tab": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (n, 1)),
        }

    def get_data_provider(self):
        return self._d


def test_build_rhs_rejects_zero_sun_table_with_srp_enabled() -> None:
    eng = DynamicsEngine(
        sc_props=SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3),
        flags=PerturbationFlags(enable_sh=False, enable_srp=True),
        ephem_manager=_ZeroBodyEphem(),
        allow_identity_rotation=True,
    )
    with pytest.raises(ValueError, match="Sun table is all zeros"):
        eng.build_rhs(force_rebuild=True)


def test_build_rhs_rejects_zero_earth_table_with_third_body_earth() -> None:
    eng = DynamicsEngine(
        sc_props=SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3),
        flags=PerturbationFlags(enable_sh=False, enable_3rd_body_earth=True),
        ephem_manager=_ZeroBodyEphem(sun=(AU, 0.0, 0.0)),
        allow_identity_rotation=True,
    )
    with pytest.raises(ValueError, match="Earth table is all zeros"):
        eng.build_rhs(force_rebuild=True)


def test_build_rhs_allows_zero_tables_when_only_quaternion_needed() -> None:
    """SH-only dynamics need only q_i2f; zero Sun/Earth rows must stay legal."""
    degree = 2
    c = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    s = np.zeros_like(c)
    gravity = GravityModel.from_arrays(degree, 1_737_400.0, MU_MOON, c, s)
    eng = DynamicsEngine(
        sc_props=SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3),
        flags=PerturbationFlags(enable_sh=True),
        gravity_model=gravity,
        ephem_manager=_ZeroBodyEphem(),
        allow_identity_rotation=False,
    )
    rhs = eng.build_rhs(force_rebuild=True)
    dy = rhs(0.0, _build_default_state())
    assert np.all(np.isfinite(dy))


def test_build_rhs_degrades_external_1pn_on_zero_tables_with_warning() -> None:
    """Auto-enabled external 1PN terms degrade (documented policy) instead of
    failing the run when the ephemeris has no Sun/Earth vectors."""
    eng = DynamicsEngine(
        sc_props=SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3),
        flags=PerturbationFlags(enable_sh=False, enable_relativity_1pn=True),
        ephem_manager=_ZeroBodyEphem(),
        allow_identity_rotation=True,
    )
    with pytest.warns(RuntimeWarning, match="external-body relativity terms disabled"):
        rhs = eng.build_rhs(force_rebuild=True)
    assert eng._prep["req"]["use_rel_external"] is False
    dy = rhs(0.0, _build_default_state())
    assert np.all(np.isfinite(dy))


def test_body_fixed_gravity_without_ephemeris_fails_closed() -> None:
    """Phase 6 frame safety: body-fixed SH gravity needs q_i2f. Without an
    ephemeris AND without an explicit identity opt-in, construction must fail
    rather than silently evaluate Moon-fixed gravity in inertial coordinates."""
    degree = 2
    c = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    s = np.zeros_like(c)
    gravity = GravityModel.from_arrays(degree, 1_737_400.0, MU_MOON, c, s)
    with pytest.raises(ValueError, match="Ephemeris is required"):
        DynamicsEngine(
            sc_props=SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3),
            flags=PerturbationFlags(enable_sh=True),
            gravity_model=gravity,
            ephem_manager=None,
            allow_identity_rotation=False,
        )


def test_body_fixed_gravity_identity_rotation_requires_explicit_opt_in() -> None:
    """The identity-rotation smoke path is legal only when explicitly allowed."""
    degree = 2
    c = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    s = np.zeros_like(c)
    gravity = GravityModel.from_arrays(degree, 1_737_400.0, MU_MOON, c, s)
    eng = DynamicsEngine(
        sc_props=SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3),
        flags=PerturbationFlags(enable_sh=True),
        gravity_model=gravity,
        ephem_manager=None,
        allow_identity_rotation=True,
    )
    assert eng.allow_identity_rotation is True
    dy = eng.build_rhs(force_rebuild=True)(0.0, _build_default_state())
    assert np.all(np.isfinite(dy))
