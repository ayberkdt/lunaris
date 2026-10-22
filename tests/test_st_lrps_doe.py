"""Screening-campaign generator (tools/st_lrps_doe.py): design properties,
generated scenario files, real-parser acceptance and the effect analysis."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
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
        assert cfg.low_degree_loss_weight == (100.0 if lv["C"] > 0 else 0.0)
        assert cfg.low_degree_lmax == 12
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


def _noise(name: str) -> float:
    """Deterministic +-0.01 pseudo-noise per run so Lenth's PSE is non-zero."""
    return 0.01 * ((sum(map(ord, name)) % 7) - 3) / 3.0


def test_analysis_recovers_planted_effects_and_recommends_only_supported_levels(tmp_path):
    # log10 response = -3 + 0.3*A - 0.2*C + noise  ->  effects A = +0.6, C = -0.4.
    for run in doe.build_runs("all"):
        if not run.levels:
            continue
        y = -3.0 + 0.3 * run.levels["A"] - 0.2 * run.levels["C"] + _noise(run.name)
        _write_history(tmp_path / run.name, 10.0**y, epochs=doe.FIDELITY_EPOCHS)
    report = doe.analyze(tmp_path, "all", "final")
    assert report["orthogonal"] is True
    block = report["effects"]["val_u_low_rmse_m2_s2"]
    assert block["effects"]["A"] == pytest.approx(0.6, abs=0.02)
    assert block["effects"]["C"] == pytest.approx(-0.4, abs=0.02)
    assert len(block["effects"]) + len(block["alias_contrasts"]) == 15
    rec = report["recommended_levels"]
    assert (rec["A"], rec["C"]) == (-1, +1)
    assert all(rec[k] == 0 for k in "BDEF")
    assert doe._write_report(report, tmp_path).is_file()


def test_phase1_analysis_uses_seven_orthogonal_contrasts(tmp_path):
    for run in doe.design_runs("phase1"):
        _write_history(tmp_path / run.name, 10.0 ** (-3.0 + _noise(run.name)), epochs=doe.FIDELITY_EPOCHS)
    block = doe.analyze(tmp_path, "phase1", "final")["effects"]["val_u_low_rmse_m2_s2"]
    assert len(block["effects"]) + len(block["alias_contrasts"]) == 7


def test_incomplete_designs_are_not_analysed(tmp_path):
    runs = doe.design_runs("phase1")
    for run in runs[:-1]:
        _write_history(tmp_path / run.name, 1e-3, epochs=doe.FIDELITY_EPOCHS)
    _write_history(tmp_path / runs[-1].name, 1e-3, epochs=doe.FIDELITY_EPOCHS - 1)
    report = doe.analyze(tmp_path, "phase1", "final")
    assert report["effects"] == {} and report["incomplete_design"] == [runs[-1].name]
    assert "recommended_levels" not in report
    assert "incomplete" in doe._write_report(report, tmp_path).read_text(encoding="utf-8")


def test_launch_plan_reuses_only_runs_with_identical_flags(tmp_path):
    run = doe.design_runs("phase1")[0]
    run_dir = tmp_path / run.name
    data = tmp_path / "data"
    assert doe.plan_launch(run, run_dir, data) == "start"
    run_dir.mkdir()
    assert doe.plan_launch(run, run_dir, data) == "stale"    # no launch record
    (run_dir / doe.FLAGS_FILE).write_text(json.dumps(doe._launch_record(run, data)), encoding="utf-8")
    assert doe.plan_launch(run, run_dir, data) == "stale"    # nothing to resume from
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "ckpt_last.pt").write_bytes(b"x")
    assert doe.plan_launch(run, run_dir, data) == "resume"
    assert doe.plan_launch(run, run_dir, tmp_path / "other_data") == "stale"
    (run_dir / "history.jsonl").write_text(
        "\n".join(json.dumps({"epoch": e}) for e in range(doe.FIDELITY_EPOCHS)), encoding="utf-8"
    )
    assert doe.plan_launch(run, run_dir, data) == "skip"
    record = doe._launch_record(run, data)
    record["flags"] = record["flags"] + ["--x"]
    (run_dir / doe.FLAGS_FILE).write_text(json.dumps(record), encoding="utf-8")
    assert doe.plan_launch(run, run_dir, data) == "stale"


def test_resume_repeats_every_design_flag(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from lunaris.surrogate.st_lrps.training.config import parse_args

    run = next(r for r in doe.design_runs("phase1") if r.levels["C"] > 0 and r.levels["E"] > 0)
    cmd = doe.resume_command(run, tmp_path / "data", tmp_path / run.name)
    assert "--out" not in cmd and cmd[cmd.index("--resume-from") + 1] == str(tmp_path / run.name)
    monkeypatch.setattr(sys, "argv", ["lunaris-train", *cmd[3:]])
    cfg = parse_args()
    assert cfg.low_degree_loss_weight == 100.0 and cfg.laplacian_mode == "train"
    assert (cfg.max_train_batches, cfg.lr_schedule_mode, cfg.epochs) == (488, "wsd", doe.FIDELITY_EPOCHS)


def test_lenth_margin_matches_the_published_critical_values():
    # Lenth (1989): t_0.975 on m/3 df; 7 contrasts -> ~3.76, 15 contrasts -> ~2.57.
    assert doe.lenth_margin_t(7) == pytest.approx(3.76, abs=0.02)
    assert doe.lenth_margin_t(15) == pytest.approx(2.571, abs=0.002)


def test_lenth_pse_ignores_one_active_effect():
    effects = [0.01, -0.02, 0.015, -0.01, 0.02, 1.0]
    assert doe.lenth_pse(effects) < 0.05


def test_run_env_puts_this_checkout_first(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "elsewhere")
    path = doe._train_env()["PYTHONPATH"].split(__import__("os").pathsep)
    assert path[0] == str(_REPO_ROOT / "src") and path[1] == "elsewhere"


@pytest.mark.parametrize(("stage", "epochs"), [("confirm", 200), ("final", 800)])
def test_promoted_runs_train_on_full_data_with_the_chosen_levels(monkeypatch, tmp_path, stage, epochs):
    pytest.importorskip("torch")
    from lunaris.surrogate.st_lrps.training.config import parse_args

    levels = doe.parse_levels("ABcDef")
    run = doe.promoted_run(levels, stage, seed=43)
    monkeypatch.setattr(sys, "argv", ["lunaris-train", *doe._command(run, tmp_path, tmp_path)[3:]])
    cfg = parse_args()
    assert cfg.epochs == epochs and cfg.max_train_batches is None and cfg.max_val_batches is None
    assert (cfg.seed, cfg.split_seed) == (43, doe.SEED)
    assert (cfg.best_metric, cfg.ema_decay, cfg.lr_schedule_mode) == ("potential_low", 0.999, "wsd")
    assert (cfg.w_u, cfg.low_degree_loss_weight, cfg.band_init) == (10.0, 0.0, "siren_first")


def test_levels_string_is_validated():
    assert doe.parse_levels("abcdef") == dict.fromkeys(doe.FACTOR_KEYS, -1)
    for bad in ("ABCDE", "ABCDEG", "AACDEF"):
        with pytest.raises(ValueError, match="levels"):
            doe.parse_levels(bad)


def test_arc_proxy_becomes_the_primary_response(monkeypatch, tmp_path):
    planted, rough = {}, {}
    for run in doe.build_runs("all"):
        if not run.levels:
            continue
        _write_history(tmp_path / run.name, 1e-3 * (1.0 + _noise(run.name)), epochs=doe.FIDELITY_EPOCHS)
        planted[run.name] = 10.0 ** (-1.0 - 0.25 * run.levels["B"] + _noise(run.name))
        rough[run.name] = 1.4 * 10.0 ** (0.1 * run.levels["E"] + _noise(run.name))
    seen = set()

    def fake(run_dir, proxy, device, prefer):
        seen.add(prefer)
        return {"gmean_abs_along_track_m": planted[Path(run_dir).name],
                "roughness_ratio_below_300km": rough[Path(run_dir).name]}

    monkeypatch.setattr(doe, "_arc_proxy_summary", fake)
    report = doe.analyze(tmp_path, "all", "final", arc_proxy=tmp_path / "proxy")
    assert report["primary_response"] == "arc_proxy_gmean_m"
    assert report["effects"]["arc_proxy_gmean_m"]["effects"]["B"] == pytest.approx(-0.5, abs=0.02)
    assert report["recommended_levels"]["B"] == +1
    assert seen == {"last"}       # --at final scores the final checkpoint
    assert report["effects"]["arc_proxy_roughness"]["effects"]["E"] == pytest.approx(0.2, abs=0.02)


def test_promotion_can_select_on_validation_orbits(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from lunaris.surrogate.st_lrps.training.config import parse_args

    run = doe.promoted_run(doe.parse_levels("ABcDef"), "final", arc_val_data=tmp_path / "val_orbits")
    monkeypatch.setattr(sys, "argv", ["lunaris-train", *doe._command(run, tmp_path, tmp_path)[3:]])
    cfg = parse_args()
    assert (cfg.best_metric, cfg.arc_val_every, cfg.arc_val_data) == ("arc_val", 1, str(tmp_path / "val_orbits"))



def test_response_surface_design_and_fit(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from lunaris.surrogate.st_lrps.training.config import parse_args

    factors = [doe.parse_rsm_factor("w_u:--w-u:1:30:log"), doe.parse_rsm_factor("ldl:--low-degree-loss-weight:10:1000:log")]
    runs = doe.rsm_runs(doe.parse_levels("ABcDef"), factors)
    assert len(runs) == 11 and len({r.name for r in runs}) == 11
    monkeypatch.setattr(sys, "argv", ["lunaris-train", *doe._command(runs[0], tmp_path, tmp_path)[3:]])
    cfg = parse_args()
    assert (cfg.w_u, cfg.low_degree_loss_weight) == pytest.approx((1.0, 10.0))   # (-1, -1) overrides A/C
    monkeypatch.setattr(sys, "argv", ["lunaris-train", *doe._command(runs[-1], tmp_path, tmp_path)[3:]])
    cfg = parse_args()
    assert cfg.seed == doe.SEED + 2 and cfg.w_u == pytest.approx(math.sqrt(30.0), rel=1e-5)
    # planted quadratic with its minimum at coded (0.4, -0.5)
    for r, (c1, c2, _seed) in zip(runs, doe._rsm_points(), strict=True):
        y = -2.0 + 0.5 * (c1 - 0.4) ** 2 + 0.3 * (c2 + 0.5) ** 2
        _write_history(tmp_path / r.name, 10.0**y, epochs=doe.FIDELITY_EPOCHS)
    fit = doe.rsm_analyze(runs, factors, tmp_path, None, "cpu")
    assert fit["optimum"]["coded"] == pytest.approx([0.4, -0.5], abs=0.011)
    assert fit["optimum"]["values"]["w_u"] == pytest.approx(factors[0].value(0.4), rel=0.02)
    with pytest.raises(ValueError, match="name:--flag"):
        doe.parse_rsm_factor("w_u:w-u:1:30")
