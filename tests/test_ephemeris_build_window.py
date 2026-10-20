"""SPICE table window, kernel-pool hygiene and start_date normalisation.

- ``build_spice_tables`` must cover the absolute solver window
  ``[0, t0_s + duration_s]`` (table t=0 is ``start_date``); building over
  ``duration_s`` froze the Sun/Earth vectors and attitude for the last ``t0_s``.
- The default build must not ``kclear`` kernels another caller furnished.
- A timezone-aware ``start_date`` must reach SPICE as a UTC ``Z`` string.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("spiceypy")

from lunaris.common.type_defs import TimeConfig  # noqa: E402
from lunaris.physics import ephemeris as eph  # noqa: E402


def _capture_build(monkeypatch) -> dict:
    captured: dict = {}

    def fake_build_tables(**kwargs):
        captured.update(kwargs)
        return "tables"

    monkeypatch.setattr(eph, "build_tables", fake_build_tables)
    return captured


def test_tables_cover_t0_plus_duration(monkeypatch):
    captured = _capture_build(monkeypatch)
    tc = TimeConfig(t0_s=86_400.0, duration_s=3_600.0, output_dt_s=60.0)
    eph.build_spice_tables(tc, eph.SpiceBuildConfig(kernels=("naif0012.tls",)))
    assert captured["duration_s"] == pytest.approx(90_000.0)


@pytest.mark.parametrize("output_dt_s, table_dt_s", [(3600.0, 60.0), (600.0, 60.0), (60.0, 60.0), (10.0, 10.0)])
def test_table_step_does_not_follow_a_coarse_output_step(monkeypatch, output_dt_s, table_dt_s):
    # The output grid must not coarsen the force-model tables: a 3600 s output step
    # moved a 100 km lunar orbit (SH 60, Sun + Earth, DOP853 rtol 1e-12) by 0.60 mm
    # after 1 day and 10.8 mm after 2 days against 60 s tables.
    captured = _capture_build(monkeypatch)
    tc = TimeConfig(duration_s=7_200.0, output_dt_s=output_dt_s)
    eph.build_spice_tables(tc, eph.SpiceBuildConfig(kernels=("naif0012.tls",)))
    assert captured["output_dt_s"] == table_dt_s


@pytest.mark.parametrize("output_dt_s", [60.0, 600.0, 3600.0])
def test_capped_tables_run_one_output_step_past_the_end(monkeypatch, output_dt_s):
    # A Yoshida composition samples up to 0.916 steps past tf and the samplers
    # extrapolate one table interval. With the 60 s cap the table runs on by
    # output_dt - 60 s, so a step up to output_dt stays covered; output_dt <= 60 s
    # builds exactly the same table as before.
    captured = _capture_build(monkeypatch)
    tc = TimeConfig(duration_s=7_200.0, output_dt_s=output_dt_s)
    eph.build_spice_tables(tc, eph.SpiceBuildConfig(kernels=("naif0012.tls",)))
    table_dt = captured["output_dt_s"]
    assert captured["duration_s"] == pytest.approx(7_200.0 + output_dt_s - table_dt)
    assert captured["duration_s"] + table_dt >= 7_200.0 + 0.9154595930203948 * output_dt_s


def test_negative_t0_is_refused(monkeypatch):
    _capture_build(monkeypatch)
    tc = TimeConfig(t0_s=-60.0, duration_s=3_600.0, output_dt_s=60.0)
    with pytest.raises(ValueError, match="precedes start_date"):
        eph.build_spice_tables(tc, eph.SpiceBuildConfig(kernels=("naif0012.tls",)))


@pytest.mark.parametrize(
    "start, expected",
    [
        ("2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
        ("2025-01-01T03:00:00+03:00", "2025-01-01T00:00:00Z"),
        (datetime(2025, 1, 1, 3, 0, 0, tzinfo=timezone(timedelta(hours=3))), "2025-01-01T00:00:00Z"),
        ("2025-01-01T00:00:00.250Z", "2025-01-01T00:00:00.250000Z"),
        ("2025 JAN 01 00:00:00 TDB", "2025 JAN 01 00:00:00 TDB"),
    ],
)
def test_start_date_is_normalised_to_utc(monkeypatch, start, expected):
    captured = _capture_build(monkeypatch)
    tc = TimeConfig(start_date=start, duration_s=3_600.0, output_dt_s=60.0)  # type: ignore[arg-type]
    eph.build_spice_tables(tc, eph.SpiceBuildConfig(kernels=("naif0012.tls",)))
    assert captured["start_utc"] == expected


class _FakeSpice:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def kclear(self) -> None:
        self.calls.append(("kclear", None))

    def furnsh(self, path: str) -> None:
        self.calls.append(("furnsh", path))

    def unload(self, path: str) -> None:
        self.calls.append(("unload", path))


def test_default_kernel_context_unloads_only_its_own_kernels(monkeypatch):
    fake = _FakeSpice()
    monkeypatch.setattr(eph, "spice", fake)
    with eph._spice_kernels_loaded(["a.tls", "b.bsp"]):
        pass
    assert fake.calls == [("furnsh", "a.tls"), ("furnsh", "b.bsp"), ("unload", "b.bsp"), ("unload", "a.tls")]


def test_kernel_context_unloads_partial_loads_on_failure(monkeypatch):
    fake = _FakeSpice()

    def failing_furnsh(path: str) -> None:
        if path == "b.bsp":
            raise RuntimeError("missing kernel")
        fake.calls.append(("furnsh", path))

    fake.furnsh = failing_furnsh  # type: ignore[method-assign]
    monkeypatch.setattr(eph, "spice", fake)
    with pytest.raises(RuntimeError), eph._spice_kernels_loaded(["a.tls", "b.bsp"]):
        pass
    assert fake.calls == [("furnsh", "a.tls"), ("unload", "a.tls")]


def test_explicit_kclear_flags_still_clear_the_pool(monkeypatch):
    fake = _FakeSpice()
    monkeypatch.setattr(eph, "spice", fake)
    with eph._spice_kernels_loaded(["a.tls"], clean_before=True, clear_after=True):
        pass
    assert fake.calls == [("kclear", None), ("furnsh", "a.tls"), ("kclear", None)]


def test_spice_build_config_default_does_not_clear_the_pool():
    assert eph.SpiceBuildConfig().clear_kernels_after is False
