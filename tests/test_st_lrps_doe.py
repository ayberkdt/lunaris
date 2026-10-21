"""Screening-campaign generator (tools/st_lrps_doe.py): design properties,
generated scenario files, real-parser acceptance and the effect analysis."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_doe():
    spec = importlib.util.spec_from_file_location("st_lrps_doe", _REPO_ROOT / "tools" / "st_lrps_doe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


doe = _load_doe()


def _matrix(runs):
    return [[r.levels[k] for k in doe.FACTOR_KEYS] for r in runs]


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v, strict=True))


def test_phase1_is_a_balanced_orthogonal_resolution_iii_fraction():
    rows = _matrix(doe.design_runs("phase1"))
    assert len(rows) == 8 and len({tuple(r) for r in rows}) == 8
    cols = list(zip(*rows, strict=True))
    for c in cols:
        assert sum(c) == 0
    for i, j in itertools.combinations(range(6), 2):
        assert _dot(cols[i], cols[j]) == 0


def test_phase1_plus_foldover_is_resolution_iv():
    runs = doe.design_runs("all")
    rows = _matrix(runs)
    assert len(rows) == 16 and len({tuple(r) for r in rows}) == 16
    cols = list(zip(*rows, strict=True))
    for main in range(6):
        for i, j in itertools.combinations(range(6), 2):
            interaction = [a * b for a, b in zip(cols[i], cols[j], strict=True)]
            assert _dot(cols[main], interaction) == 0, (main, i, j)


def test_foldover_reuses_the_corrected_anchor_instead_of_a_duplicate_run():
    new = doe.build_runs("foldover")
    assert len(new) == 7
    assert all(any(v > 0 for v in r.levels.values()) for r in new)
    names = [r.name for r in doe.build_runs("all")]
    assert len(names) == len(set(names)) == 17
    assert doe.ANCHOR_CORRECTED in {r.name for r in doe.design_runs("foldover")}


@pytest.mark.parametrize("phase", sorted(doe.SCENARIO_FILES))
def test_committed_scenario_files_match_the_generator(phase):
    path = doe.SCENARIO_DIR / doe.SCENARIO_FILES[phase]
    assert path.read_text(encoding="utf-8") == doe.render_scenarios(phase)


def test_every_run_is_accepted_by_the_real_training_parser(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from lunaris.surrogate.st_lrps.training.config import parse_args

    for run in doe.build_runs("all"):
        cmd = doe._command(run, tmp_path / "data", tmp_path / "out")
        monkeypatch.setattr(sys, "argv", ["lunaris-train", *cmd[3:]])
        cfg = parse_args()
        assert cfg.epochs == doe.FIDELITY_EPOCHS and cfg.lr_schedule_mode == "wsd"
        assert (cfg.max_train_batches, cfg.max_val_batches) == (488, 31)
        assert (cfg.seed, cfg.split_seed) == (doe.SEED, doe.SEED)
        if not run.levels:  # shipped anchor
            assert (cfg.u_center, cfg.gradnorm_mode, cfg.w_a) == ("mean", "ntk_init", 1.0)
            continue
        lv = run.levels
        assert (cfg.u_center, cfg.gradnorm_mode, cfg.w_a) == ("zero", "fixed", 2.0)
        assert cfg.w_u == (10.0 if lv["A"] > 0 else 1.0)
        assert cfg.direction_loss_weight == (0.02 if lv["B"] > 0 else 0.1)
        assert cfg.relative_loss_weight == (0.01 if lv["B"] > 0 else 0.05)
        assert cfg.low_degree_loss_weight == (10.0 if lv["C"] > 0 else 0.0)
        assert cfg.band_init == ("siren_first" if lv["D"] > 0 else "legacy")
        assert cfg.laplacian_mode == ("train" if lv["E"] > 0 else "diagnostic")
        assert cfg.use_altitude_balanced_loss is (lv["F"] < 0)


def _write_history(run_dir: Path, value: float, *, epochs: int) -> None:
    run_dir.mkdir(parents=True)
    rows = [
        {"epoch": e, "best_epoch": epochs, "val_u_low_rmse_m2_s2": value, "val_accel_rmse_m_s2": 1e-8,
         "val_loss_low_degree": 1e-9, "val_angular_mean_deg": 0.1, "val_wtz_frac": 0.0}
        for e in range(epochs)
    ]
    (run_dir / "history.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_analysis_recovers_planted_effects_and_recommends_their_signs(tmp_path):
    # log10 response = -3 + 0.3*A - 0.2*C  ->  effects A = +0.6, C = -0.4, rest 0.
    for run in doe.build_runs("all"):
        if not run.levels:
            continue
        y = -3.0 + 0.3 * run.levels["A"] - 0.2 * run.levels["C"]
        _write_history(tmp_path / run.name, 10.0**y, epochs=doe.FIDELITY_EPOCHS)
    report = doe.analyze(tmp_path, "all", "final")
    eff = report["effects"]["val_u_low_rmse_m2_s2"]["effects"]
    assert eff["A"] == pytest.approx(0.6) and eff["C"] == pytest.approx(-0.4)
    for k in "BDEF":
        assert eff[k] == pytest.approx(0.0, abs=1e-12)
    assert report["recommended_levels"]["A"] == -1 and report["recommended_levels"]["C"] == +1
    assert doe._write_report(report, tmp_path).is_file()


def test_incomplete_runs_are_left_out_of_the_effects(tmp_path):
    for run in doe.design_runs("phase1"):
        _write_history(tmp_path / run.name, 1e-3, epochs=doe.FIDELITY_EPOCHS - 1)
    report = doe.analyze(tmp_path, "phase1", "final")
    assert report["effects"] == {}
    assert {row["status"] for row in report["runs"] if row["name"].startswith("DoE_P1_R")} == {"partial"}


def test_lenth_pse_ignores_one_active_effect():
    effects = [0.01, -0.02, 0.015, -0.01, 0.02, 1.0]
    assert doe.lenth_pse(effects) < 0.05
