from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from lunaris.cli.run import (
    CliStageError,
    build_run_meta,
    create_canonical_run_dir,
    main_entry,
    write_run_artifacts,
)
from lunaris.core.config import load_default_config
from lunaris.ui.core.results_index import index_runs


def test_write_run_artifacts_writes_diagnostics_and_config(tmp_path, capsys, monkeypatch) -> None:
    import pathlib

    import lunaris.core.config
    monkeypatch.setattr(lunaris.core.config, "_resolve_default_kernel_paths", lambda *a, **k: ("mock",))
    monkeypatch.setattr(lunaris.core.config, "_resolve_default_gravity_path", lambda *a, **k: pathlib.Path("mock"))
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
    monkeypatch.setattr(lunaris.core.config, "_resolve_default_kernel_paths", lambda *a, **k: ("mock",))
    monkeypatch.setattr(lunaris.core.config, "_resolve_default_gravity_path", lambda *a, **k: pathlib.Path("mock"))
    cfg = load_default_config()
    result = SimpleNamespace(t=np.asarray([0.0, 10.0, 20.0, 30.0], dtype=np.float64))

    meta = build_run_meta(cfg, result, mu=4.9, propagation_time_s=1.25)

    assert meta["mu_m3s2"] == 4.9
    assert meta["propagation_time_s"] == 1.25
    assert meta["output_dt_s_measured"] == 10.0
    assert meta["output_epoch_count"] == 4


def test_canonical_run_dir_colocates_artifacts_for_the_indexer(tmp_path, monkeypatch) -> None:
    """Config, diagnostics, and figures must share one leaf the indexer finds.

    Regression for the split-directory bug: config was written to the output root
    while figures went to a separate ``run_*`` subdir, so the Run History gallery
    came up empty and re-used roots overwrote the top-level config. The canonical
    run directory keeps everything together as one self-contained run.
    """
    import pathlib

    import lunaris.core.config
    monkeypatch.setattr(lunaris.core.config, "_resolve_default_kernel_paths", lambda *a, **k: ("mock",))
    monkeypatch.setattr(lunaris.core.config, "_resolve_default_gravity_path", lambda *a, **k: pathlib.Path("mock"))
    cfg = load_default_config()

    out_root = tmp_path / "missions"
    out_root.mkdir()

    run_dir = create_canonical_run_dir(out_root)
    assert run_dir.parent == out_root.resolve()
    assert run_dir.name.startswith("run_")

    write_run_artifacts(run_dir, cfg, {"method": "DOP853"})
    # Reporting writes figures/PDF into the same leaf (use_run_subdir=False).
    (run_dir / "orbit.png").write_bytes(b"\x89PNG\r\n")
    (run_dir / "report.pdf").write_bytes(b"%PDF-1.4")

    # The output root holds no config of its own; only the canonical leaf is a run.
    assert not (out_root / "run_config.json").exists()
    records = index_runs(out_root)
    assert len(records) == 1
    (record,) = records
    assert record.run_dir == run_dir
    assert len(record.figures) == 1
    assert len(record.reports) == 1


def test_main_entry_is_the_single_runtime_failure_boundary(monkeypatch, capsys) -> None:
    import lunaris.cli.run as run_module

    def fail_main() -> int:
        raise CliStageError("Config init failed", ValueError("bad config"))

    monkeypatch.setattr(run_module, "main", fail_main)

    assert main_entry() == 1
    assert "[FATAL] Config init failed: bad config" in capsys.readouterr().out
