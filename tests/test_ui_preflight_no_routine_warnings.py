"""Pre-flight: a default run raises no warning; real concerns still do.

Regression tests: the gravity and albedo checks returned an informational
message on success ("Using default gravity model", "Gravity file size: 190 MB",
"Gravity file validated: ...", "Using constant albedo model") and the worker
emitted every non-empty message as ``validation_warning``. Gravity is on by
default, so every run logged a ``[Warning]`` and lit the log's warning badge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from lunaris.ui.core.preflight_validation import PreFlightWorker  # noqa: E402


def _run(tmp_path: Path, **overrides) -> dict[str, list]:
    script = tmp_path / "engine.py"
    script.write_text("", encoding="utf-8")
    data = {
        "orbit_mode": "hp_ha", "hp_km": 100.0, "ha_km": 100.0,
        "inc_deg": 90.0, "raan_deg": 0.0, "argp_deg": 0.0, "ta_deg": 0.0,
        "gravity_enabled": True, "gravity_backend": "classic_sh", "gravity_file": "",
        "output_dir": str(tmp_path / "out"),
        "mass_kg": 1000.0, "area_m2": 5.0, "cd": 2.2, "cr": 1.5,
        "rtol": 1e-12, "atol": 1e-14, "duration_val": 10.0,
    }
    data.update(overrides)
    worker = PreFlightWorker(data, script)
    seen: dict[str, list] = {"warning": [], "error": [], "progress": [], "complete": []}
    worker.validation_warning.connect(seen["warning"].append)
    worker.validation_error.connect(seen["error"].append)
    worker.validation_progress.connect(seen["progress"].append)
    worker.validation_complete.connect(lambda ok, msg: seen["complete"].append((ok, msg)))
    worker.run()  # synchronously, on this thread
    return seen


def test_default_gravity_run_has_no_warning(tmp_path) -> None:
    seen = _run(tmp_path)
    assert seen["complete"] == [(True, "Pre-flight validation successful")], seen
    assert seen["warning"] == []
    assert "Using default gravity model" in seen["progress"]


def test_validated_gravity_file_is_progress_not_warning(tmp_path) -> None:
    gravity = tmp_path / "gravity.sha"
    gravity.write_bytes(b"0" * 1024)
    seen = _run(tmp_path, gravity_file=str(gravity))
    assert seen["warning"] == []
    assert any("gravity.sha" in p for p in seen["progress"])


def test_constant_albedo_is_progress_not_warning(tmp_path) -> None:
    seen = _run(tmp_path, albedo_enabled=True, albedo_source="constant_albedo")
    assert seen["warning"] == []
    assert "Using constant albedo model" in seen["progress"]


def test_swapped_apsides_still_warn(tmp_path) -> None:
    seen = _run(tmp_path, hp_km=200.0, ha_km=100.0)
    assert any("Values will be swapped" in w for w in seen["warning"])


def test_large_gravity_file_still_warns(tmp_path, monkeypatch) -> None:
    gravity = tmp_path / "huge.sha"
    gravity.write_bytes(b"0")
    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        result = real_stat(self, *args, **kwargs)
        if self.name == "huge.sha":
            class _S:
                st_size = 600 * 1024 * 1024
                st_mode = result.st_mode
            return _S()
        return result

    monkeypatch.setattr(Path, "stat", fake_stat)
    seen = _run(tmp_path, gravity_file=str(gravity))
    assert any("Large gravity file" in w for w in seen["warning"])
