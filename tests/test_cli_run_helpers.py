from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from lunaris.cli.run import build_run_meta, write_run_artifacts
from lunaris.core.config import load_default_config


def test_write_run_artifacts_writes_diagnostics_and_config(tmp_path, capsys, monkeypatch) -> None:
    import pathlib
    import lunaris.core.config
    monkeypatch.setattr(lunaris.core.config, "_resolve_default_kernel_paths", lambda: ("mock",))
    monkeypatch.setattr(lunaris.core.config, "_resolve_default_gravity_path", lambda: pathlib.Path("mock"))
    cfg = load_default_config()
    diag = {"method": "DOP853", "impacted": False}

    write_run_artifacts(tmp_path, cfg, diag)

    assert json.loads((tmp_path / "run_diagnostics.json").read_text()) == diag
    config_payload = json.loads((tmp_path / "run_config.json").read_text())
    assert config_payload["propagator"]["method"] == cfg.propagator.method
    assert "[DIAG]" in capsys.readouterr().out


def test_build_run_meta_uses_measured_output_spacing(monkeypatch) -> None:
    import pathlib
    import lunaris.core.config
    monkeypatch.setattr(lunaris.core.config, "_resolve_default_kernel_paths", lambda: ("mock",))
    monkeypatch.setattr(lunaris.core.config, "_resolve_default_gravity_path", lambda: pathlib.Path("mock"))
    cfg = load_default_config()
    result = SimpleNamespace(t=np.asarray([0.0, 10.0, 20.0, 30.0], dtype=np.float64))

    meta = build_run_meta(cfg, result, mu=4.9, propagation_time_s=1.25)

    assert meta["mu_m3s2"] == 4.9
    assert meta["propagation_time_s"] == 1.25
    assert meta["output_dt_s_measured"] == 10.0
    assert meta["output_epoch_count"] == 4
