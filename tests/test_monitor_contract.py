"""Telemetry contract tests (Mission Monitor test group A).

Covers serialize/deserialize round-trips, schema-version validation, the
finite-value policy, and optional-channel handling (absent, never zero).
"""

from __future__ import annotations

import json
import math

import pytest

from lunaris.common.telemetry_contract import (
    TELEMETRY_META_PREFIX,
    TELEMETRY_SAMPLE_PREFIX,
    TELEMETRY_SCHEMA_VERSION,
    TelemetryDecodeError,
    TelemetryEvent,
    TelemetryProvenance,
    TelemetrySample,
    UnsupportedTelemetrySchemaError,
    decode_meta_line,
    decode_sample_line,
    encode_meta_line,
    encode_sample_line,
    sample_from_legacy_dict,
    sample_from_payload,
    sample_to_payload,
)


def make_sample(**overrides) -> TelemetrySample:
    base = dict(
        run_id="run_x",
        sequence_id=7,
        simulation_time_s=120.5,
        wall_time_s=1.25,
        altitude_m=52_000.0,
        radius_m=1_789_400.0,
        speed_m_s=1_633.2,
        state_inertial=(1.0e6, 2.0e6, 3.0e5, 100.0, -200.0, 30.0),
        frame_inertial="MOON_INERTIAL",
        orbital_elements={"sma_m": 1.8e6, "ecc": 0.01, "inc_rad": 1.5},
        diagnostics={"rhs_evals": 4200, "backend": "scipy", "impacted": False},
        events=(TelemetryEvent("periselene", 60.0, "periapsis pass"),),
    )
    base.update(overrides)
    return TelemetrySample(**base)


class TestRoundTrip:
    def test_sample_line_round_trip_preserves_every_field(self):
        sample = make_sample()
        line = encode_sample_line(sample)
        assert line.startswith(TELEMETRY_SAMPLE_PREFIX + " ")
        decoded = decode_sample_line(line)
        assert decoded == sample

    def test_meta_line_round_trip(self):
        prov = TelemetryProvenance(
            run_id="run_x",
            requested_backend="torch_cuda_sh",
            effective_backend="numba_cpu_sh",
            fallback_reason="CUDA unavailable",
            integrator="DOP853",
            sh_degree=120,
            adaptive_degree=False,
            reference_radius_m=1_737_400.0,
            mu_m3s2=4.9028e12,
            config_sha256="abc123",
            data_hashes={"gravity_model": "deadbeef"},
        )
        line = encode_meta_line(prov)
        assert line.startswith(TELEMETRY_META_PREFIX + " ")
        assert decode_meta_line(line) == prov

    def test_encoded_line_is_single_line_compact_json(self):
        line = encode_sample_line(make_sample())
        assert "\n" not in line
        payload = json.loads(line[len(TELEMETRY_SAMPLE_PREFIX):])
        assert payload["schema_version"] == TELEMETRY_SCHEMA_VERSION


class TestSchemaVersion:
    def test_unknown_schema_version_fails_closed(self):
        payload = sample_to_payload(make_sample())
        payload["schema_version"] = "lunaris_telemetry_v99"
        with pytest.raises(UnsupportedTelemetrySchemaError) as excinfo:
            sample_from_payload(payload)
        assert "v99" in str(excinfo.value)

    def test_missing_schema_version_fails_closed(self):
        payload = sample_to_payload(make_sample())
        del payload["schema_version"]
        with pytest.raises(UnsupportedTelemetrySchemaError):
            sample_from_payload(payload)

    def test_missing_required_fields_raise_decode_error(self):
        payload = sample_to_payload(make_sample())
        del payload["run_id"]
        with pytest.raises(TelemetryDecodeError):
            sample_from_payload(payload)


class TestFinitePolicy:
    def test_constructor_rejects_non_finite_scalars(self):
        with pytest.raises(ValueError):
            make_sample(altitude_m=float("nan"))
        with pytest.raises(ValueError):
            make_sample(simulation_time_s=float("inf"))

    def test_encoder_never_emits_nan_json(self):
        # allow_nan=False in the encoder is the hard guarantee; a NaN reaching
        # json.dumps would raise instead of producing invalid JSON.
        line = encode_sample_line(make_sample())
        json.loads(line[len(TELEMETRY_SAMPLE_PREFIX):])  # strict-parses cleanly

    def test_decoder_maps_non_finite_optionals_to_missing(self):
        payload = sample_to_payload(make_sample())
        # Simulate a foreign producer that let NaN through (json.loads accepts it).
        payload["altitude_m"] = float("nan")
        payload["orbital_elements"]["ecc"] = float("inf")
        decoded = sample_from_payload(payload)
        assert decoded.altitude_m is None  # missing, not zero
        assert "ecc" not in decoded.orbital_elements
        assert decoded.orbital_elements["sma_m"] == pytest.approx(1.8e6)

    def test_decoder_drops_incomplete_state_vector_entirely(self):
        payload = sample_to_payload(make_sample())
        payload["state_inertial"][3] = float("nan")
        assert sample_from_payload(payload).state_inertial is None
        payload["state_inertial"] = [1.0, 2.0, 3.0]  # wrong length
        assert sample_from_payload(payload).state_inertial is None


class TestOptionalChannels:
    def test_minimal_sample_has_no_fake_values(self):
        sample = TelemetrySample(run_id="r", sequence_id=0, simulation_time_s=0.0)
        payload = sample_to_payload(sample)
        for absent in ("altitude_m", "radius_m", "state_inertial",
                       "orbital_elements", "diagnostics", "events"):
            assert absent not in payload
        decoded = sample_from_payload(payload)
        assert decoded.altitude_m is None
        assert decoded.orbital_elements == {}
        assert decoded.events == ()

    def test_events_decode_leniently_but_never_invent_fields(self):
        payload = sample_to_payload(make_sample())
        payload["events"].append({"event_type": "", "simulation_time_s": 1.0})  # invalid
        payload["events"].append("not-a-dict")
        decoded = sample_from_payload(payload)
        assert [e.event_type for e in decoded.events] == ["periselene"]

    def test_sample_mappings_are_defensively_copied(self):
        elements = {"ecc": 0.1}
        sample = TelemetrySample(
            run_id="r", sequence_id=0, simulation_time_s=0.0, orbital_elements=elements
        )
        elements["ecc"] = 9.9
        assert sample.orbital_elements["ecc"] == pytest.approx(0.1)


class TestLegacyAdapter:
    def test_legacy_dict_maps_km_fields_to_si(self):
        sample = sample_from_legacy_dict(
            {"t_s": 30.0, "alt_km": 50.0, "v_km_s": 1.6, "ecc": 0.02,
             "surface_r_km": 1737.5, "terrain_clearance_km": 48.2},
            run_id="legacy_run", sequence_id=3,
        )
        assert sample is not None
        assert sample.simulation_time_s == pytest.approx(30.0)
        assert sample.altitude_m == pytest.approx(50_000.0)
        assert sample.speed_m_s == pytest.approx(1600.0)
        assert sample.surface_radius_m == pytest.approx(1_737_500.0)
        assert sample.terrain_clearance_m == pytest.approx(48_200.0)
        assert sample.orbital_elements["ecc"] == pytest.approx(0.02)
        assert sample.sequence_id == 3

    def test_legacy_t_with_hour_and_day_units(self):
        hours = sample_from_legacy_dict({"t": 2.0, "t_unit": "h"}, sequence_id=0)
        days = sample_from_legacy_dict({"t": 1.5, "t_unit": "days"}, sequence_id=1)
        assert hours is not None and hours.simulation_time_s == pytest.approx(7200.0)
        assert days is not None and days.simulation_time_s == pytest.approx(1.5 * 86400.0)

    def test_legacy_dict_without_time_is_not_telemetry(self):
        assert sample_from_legacy_dict({"alt_km": 10.0}, sequence_id=0) is None
        assert sample_from_legacy_dict({"t_s": float("nan")}, sequence_id=0) is None

    def test_legacy_missing_channels_stay_missing(self):
        sample = sample_from_legacy_dict({"t_s": 1.0}, sequence_id=0)
        assert sample is not None
        assert sample.altitude_m is None
        assert sample.speed_m_s is None
        assert sample.orbital_elements == {}


class TestEventValidation:
    def test_event_rejects_bad_severity_and_non_finite_time(self):
        with pytest.raises(ValueError):
            TelemetryEvent("impact", math.nan)
        with pytest.raises(ValueError):
            TelemetryEvent("impact", 0.0, severity="fatal")
