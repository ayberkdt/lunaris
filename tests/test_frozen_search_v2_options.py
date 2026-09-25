"""Frozen-search v2 options: perilune sampling, discrete argp choices, frozen screening score."""
from __future__ import annotations

import numpy as np
import pytest

from lunaris.analysis.frozen.classify import FrozenClassificationConfig, frozen_score
from lunaris.analysis.frozen.domain_guard import DomainGuardResult
from lunaris.analysis.frozen.metrics import compute_frozen_metrics
from lunaris.analysis.frozen.search import (
    ElementBounds,
    FrozenSearchConfig,
    FrozenSearchDomainGuard,
    frozen_screening_scores,
    sobol_element_samples,
)

MU = 4902800306330.2
R_KM = 1737.4


def _bounds() -> ElementBounds:
    return ElementBounds(a_km=(1900.0, 2100.0), e=(0.005, 0.10), i_deg=(0.0, 180.0), argp_deg=(90.0, 270.0))


def test_hp_sampling_derives_semi_major_axis_inside_the_perilune_band() -> None:
    elements, prov = sobol_element_samples(
        _bounds(), 256, seed=3, method="sobol_scrambled", hp_km=(110.0, 150.0), r_ref_km=R_KM
    )
    a_km, e = elements[:, 0], elements[:, 1]
    hp = a_km * (1.0 - e) - R_KM
    assert hp.min() >= 110.0 - 1e-9 and hp.max() <= 150.0 + 1e-9
    assert e.min() >= 0.005 and e.max() <= 0.10
    assert prov["hp_km"] == [110.0, 150.0]
    assert prov["a_km_derived_range"][0] >= (R_KM + 110.0) / (1.0 - 0.005) - 1e-9


def test_argp_choices_land_exactly_on_the_set_with_equal_measure() -> None:
    elements, prov = sobol_element_samples(
        _bounds(), 512, seed=5, method="sobol", argp_choices=(90.0, 270.0), r_ref_km=R_KM
    )
    argp = elements[:, 4]
    assert set(np.unique(argp).tolist()) == {90.0, 270.0}
    frac = float(np.mean(argp == 90.0))
    assert 0.4 < frac < 0.6  # unscrambled Sobol splits a dimension evenly
    assert prov["argp_choices_deg"] == [90.0, 270.0]


def test_config_rejects_unknown_screening_score_and_bad_hp_band() -> None:
    kw = dict(bounds=_bounds(), n_samples=8, guard=FrozenSearchDomainGuard(altitude_min_km=100.0, altitude_max_km=1000.0))
    with pytest.raises(ValueError, match="screening_score"):
        FrozenSearchConfig(screening_score="v7", **kw)
    with pytest.raises(ValueError, match="hp_km"):
        FrozenSearchConfig(hp_km=(150.0, 110.0), **kw)
    cfg = FrozenSearchConfig(screening_score="frozen", hp_km=(110.0, 150.0), argp_choices=(90.0, 270.0), **kw)
    assert "frozen_score" in cfg.score_definition()
    assert cfg.screening_classification_config().mission_duration_s == cfg.screening_duration_s


def _kepler_block(n_orbits: float, n_t: int, a_m: float, e: float, inc: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-body (T, 1, 6) history from a Keplerian orbit, argp = 90 deg."""
    n = np.sqrt(MU / a_m**3)
    t = np.linspace(0.0, n_orbits * 2.0 * np.pi / n, n_t)
    M = n * t
    E = M.copy()
    for _ in range(30):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r = a_m * (1 - e * np.cos(E))
    p = a_m * (1 - e * e)
    r_pf = np.stack([r * np.cos(nu), r * np.sin(nu), np.zeros_like(nu)], 1)
    v_pf = np.sqrt(MU / p)[None] * np.stack([-np.sin(nu), e + np.cos(nu), np.zeros_like(nu)], 1)
    w, ci, si = np.pi / 2, np.cos(inc), np.sin(inc)
    Rm = np.array([[np.cos(w), -np.sin(w) * ci, np.sin(w) * si], [np.sin(w), np.cos(w) * ci, -np.cos(w) * si], [0.0, si, ci]])
    Y = np.concatenate([r_pf @ Rm.T, v_pf @ Rm.T], axis=1)[:, None, :]
    return t, Y


def test_frozen_screening_score_matches_direct_classification_score() -> None:
    t, Y = _kepler_block(20.0, 801, 1.95e6, 0.03, np.radians(85.0))
    guard = DomainGuardResult(
        domain_exit_flag=np.zeros(1, dtype=bool),
        t_domain_exit_s=np.full(1, np.nan),
        escape_flag=np.zeros(1, dtype=bool),
        nonfinite_flag=np.zeros(1, dtype=bool),
        reasons=np.array([""], dtype=np.str_),
    )
    cls = FrozenClassificationConfig.for_mission_duration(float(t[-1]), perilune_safety_min_m=50_000.0)
    scores, extra = frozen_screening_scores(
        t, Y, np.zeros(1), np.full(1, np.nan), guard, mu=MU, r_ref=R_KM * 1e3, cls_config=cls
    )
    from lunaris.common.math_utils import screening_orbital_elements_vec

    a_m, e, inc, argp = screening_orbital_elements_vec(Y[:, :, :3], Y[:, :, 3:], MU)
    direct = frozen_score(
        compute_frozen_metrics(t, eccentricity=e[:, 0], inclination_rad=inc[:, 0], omega_rad=argp[:, 0],
                               h_peri_m=a_m[:, 0] * (1 - e[:, 0]) - R_KM * 1e3),
        cls,
    )
    assert np.isfinite(scores[0])
    assert scores[0] == pytest.approx(direct, rel=1e-12)
    assert np.isfinite(extra["omega_span_rad"][0]) and extra["omega_span_rad"][0] < np.radians(5.0)  # two-body: argp fixed
    assert extra["inclination_range_rad"][0] < 1e-6


def test_frozen_screening_score_is_infinite_for_impact_or_domain_exit() -> None:
    t, Y = _kepler_block(5.0, 201, 1.95e6, 0.03, np.radians(85.0))
    Y2 = np.concatenate([Y, Y], axis=1)
    guard = DomainGuardResult(
        domain_exit_flag=np.array([False, True]),
        t_domain_exit_s=np.array([np.nan, float(t[100])]),
        escape_flag=np.zeros(2, dtype=bool),
        nonfinite_flag=np.zeros(2, dtype=bool),
        reasons=np.array(["", "altitude"], dtype=np.str_),
    )
    cls = FrozenClassificationConfig.for_mission_duration(float(t[-1]), perilune_safety_min_m=50_000.0)
    scores, _ = frozen_screening_scores(
        t, Y2, np.array([1.0, 0.0]), np.array([float(t[50]), np.nan]), guard, mu=MU, r_ref=R_KM * 1e3, cls_config=cls
    )
    assert not np.isfinite(scores[0])  # impacted
    assert not np.isfinite(scores[1])  # domain exit


def test_cli_parses_v2_flags() -> None:
    from lunaris.cli.frozen_search import build_parser

    args = build_parser().parse_args(
        ["--out", "x", "--hp-km", "110", "150", "--argp-choices", "90,270", "--screening-score", "frozen"]
    )
    assert args.hp_km == [110.0, 150.0]
    assert args.argp_choices == "90,270"
    assert args.screening_score == "frozen"
