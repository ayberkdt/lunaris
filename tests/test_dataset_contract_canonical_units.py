"""The dataset contract's unit block must describe the stored arrays.

``spatial_cloud_generator --canonical`` divides positions by DU, potentials by
VU^2 and accelerations by DU/TU^2 (DU = r_ref_m) before writing, but the
contract (``dataset_contract_json``, ``/metadata/contract_json``) and the
``units`` attr always claimed m, m^2/s^2 and m/s^2, and ``validate()`` refused
anything else. Only the separate ``unit_system`` attr was truthful.
"""

from __future__ import annotations

import pytest

pytest.importorskip("h5py")

from lunaris.surrogate.st_lrps.data.dataset_contract import (  # noqa: E402
    CANONICAL_UNITS,
    DEFAULT_UNITS,
    DatasetContract,
    DatasetContractError,
    build_contract_payload_for_generator,
)
from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI  # noqa: E402


def _payload(**kw):
    return build_contract_payload_for_generator(
        dataset_id="units", n_samples=4, degree_min=2, degree_max=4, target_mode="residual",
        baseline_kind="spherical_harmonics", mu_si=MU_MOON_SI, r_ref_m=R_MOON_SI,
        altitude_min_km=100.0, altitude_max_km=200.0, random_seed=1, sampling_policy={"name": "t"},
        source_gravity_model="toy", source_gravity_file_path="toy.gfc",
        source_gravity_file_sha256="a" * 64, generator_version="test",
        columns=["x", "y", "z", "dU", "dax", "day", "daz"], **kw,
    )


def test_canonical_generator_contract_declares_canonical_units() -> None:
    assert _payload(canonical=True)["units"] == CANONICAL_UNITS
    assert _payload()["units"] == DEFAULT_UNITS
    # Both validate on reload.
    assert DatasetContract.from_dict(_payload(canonical=True)).units == CANONICAL_UNITS


def test_legacy_si_claim_still_loads_and_other_units_are_refused() -> None:
    # Datasets written before the canonical block existed keep loading.
    assert DatasetContract.from_dict(_payload()).units == DEFAULT_UNITS
    bad = _payload()
    bad["units"] = {"position": "km", "potential": "km^2/s^2", "acceleration": "km/s^2"}
    with pytest.raises(DatasetContractError, match="units"):
        DatasetContract.from_dict(bad)


def test_generator_attrs_path_follows_unit_system() -> None:
    from lunaris.surrogate.st_lrps.data.spatial_cloud_generator import (
        _contract_from_current_generator_attrs,
    )

    attrs = {
        "degree_min": "2", "degree_max": "4", "target_mode": "residual",
        "mu_si": str(MU_MOON_SI), "r_ref_m": str(R_MOON_SI), "alt_min_km": "100", "alt_max_km": "200",
        "seed": "1", "source_gravity_model": "toy", "source_gravity_file_path": "toy.gfc",
        "columns": "[x,y,z,dU,dax,day,daz]",
    }
    canon = _contract_from_current_generator_attrs({**attrs, "unit_system": "canonical"},
                                                   n_samples=4, dataset_name="data")
    si = _contract_from_current_generator_attrs({**attrs, "unit_system": "si"},
                                                n_samples=4, dataset_name="data")
    assert canon.units == CANONICAL_UNITS
    assert si.units == DEFAULT_UNITS
