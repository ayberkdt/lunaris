"""The Earth-J2 default is EGM2008's J2 with EGM2008's own reference radius.

History: the config factory, ``ensure_model_configs`` and ``EarthJ2Params``
hard-coded J2 = 1.08262668e-3, which is not EGM2008's: the ICGEM EGM2008.gfc
(fully normalized, tide-free) gives C20 = -0.484165143790815e-3, so
J2 = -sqrt(5) C20 = 1.0826261738522227e-3 (4.7e-7 relative). ``EarthJ2Params``
also paired its J2 with the WGS-84 radius (6378137.0 m) instead of EGM2008's
(6378136.3 m). The pair now lives once in ``common.constants``.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

from lunaris.common import constants as C
from lunaris.core import config as config_mod
from lunaris.physics.third_body_effects import EarthJ2Params

# C-bar(2,0) and the header radius of ICGEM EGM2008.gfc (lines "gfc 2 0" and "radius").
_EGM2008_C20 = -0.484165143790815e-3
_EGM2008_RADIUS_M = 6378136.3
_EGM2008_GFC = Path(__file__).resolve().parents[1] / "data" / "gravity_models" / "EGM2008.gfc"


def test_j2_earth_egm2008_is_minus_sqrt5_c20() -> None:
    # 1e-15 relative: one sqrt and one multiply of a 15-digit decimal; five decades
    # below the old 4.7e-7 discrepancy.
    assert pytest.approx(-math.sqrt(5.0) * _EGM2008_C20, rel=1e-15, abs=0.0) == C.J2_EARTH_EGM2008
    assert C.R_EARTH_EQUATORIAL_EGM2008_M == _EGM2008_RADIUS_M
    assert "J2_EARTH_EGM2008" in C.CONSTANT_SOURCES
    assert "R_EARTH_EQUATORIAL_EGM2008_M" in C.CONSTANT_SOURCES


@pytest.mark.requires_data
def test_j2_earth_egm2008_matches_the_coefficient_file_when_present() -> None:
    if not _EGM2008_GFC.is_file():
        pytest.skip(f"EGM2008 coefficient file not present: {_EGM2008_GFC}")
    text = _EGM2008_GFC.read_text(encoding="utf-8-sig")
    assert "fully_normalized" in text and "tide_free" in text
    c20 = re.search(r"^gfc\s+2\s+0\s+(\S+)", text, flags=re.MULTILINE)
    radius = re.search(r"^radius\s+(\S+)", text, flags=re.MULTILINE)
    assert c20 and radius
    assert C.J2_EARTH_EGM2008 == pytest.approx(-math.sqrt(5.0) * float(c20.group(1).replace("D", "E")), rel=1e-12)
    assert C.R_EARTH_EQUATORIAL_EGM2008_M == pytest.approx(float(radius.group(1).replace("D", "E")), rel=1e-12)


def test_earth_j2_params_default_is_the_egm2008_pair() -> None:
    params = EarthJ2Params()
    assert params.j2_coeff == C.J2_EARTH_EGM2008
    assert params.r_eq_m == C.R_EARTH_EQUATORIAL_EGM2008_M


def _placeholder_data_dir(root: Path) -> Path:
    """Empty placeholder assets: the factory checks existence only."""
    kernels = root / "ephemeris_models"
    gravity = root / "gravity_models"
    kernels.mkdir(parents=True)
    gravity.mkdir(parents=True)
    for name in ("naif0012.tls", "pck00011.tpc", "moon_pa_de440_200625.bpc", "de440.bsp"):
        (kernels / name).write_bytes(b"")
    (gravity / "jggrx_1800f_sha.tab").write_bytes(b"")
    return root


def test_ensure_model_configs_fills_the_egm2008_pair(tmp_path: Path) -> None:
    cfg = config_mod.load_default_config(data_dir=_placeholder_data_dir(tmp_path / "data"))
    flags = replace(cfg.flags, enable_earth_j2=True)
    filled = config_mod.ensure_model_configs(replace(cfg, flags=flags, earth_j2=None))
    assert filled.earth_j2 is not None
    assert filled.earth_j2.j2_coeff == C.J2_EARTH_EGM2008
    assert filled.earth_j2.r_eq_m == C.R_EARTH_EQUATORIAL_EGM2008_M
