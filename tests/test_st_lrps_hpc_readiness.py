"""HPC-readiness guards for ST-LRPS training (plan: ST_LRPS_HPC_READINESS).

Covers:
* H2 — graceful stop-signal handling extends beyond SIGINT to SIGTERM (and
  SIGUSR1 where the platform has it), with best-effort install/restore.
* H3 — the RAM-preload budget respects cgroup and Slurm memory limits instead
  of trusting node-wide availability.
* H4 — preflight fails closed when the job context demands a GPU but training
  resolved to CPU (the silent CPU-fallback footgun).
* H5 — the scenario launcher's --resume mode relaunches with --resume-from
  when a checkpoint exists, starts fresh otherwise, and refuses ambiguity.
"""

from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

import pytest

# The readiness module exercises the full training engine, whose torch import is
# intentionally behind the optional ``ml``/``hpc`` extras. Keep core-only test
# runs green by skipping this optional suite when the extra is not installed.
pytest.importorskip("torch.nn")

from lunaris.surrogate.st_lrps.training.engine import (
    _available_ram_mb,
    _cgroup_available_mb,
    _decide_preload,
    _install_stop_signal_handlers,
    _restore_stop_signal_handlers,
    _slurm_mem_limit_mb,
    _stop_signal_list,
)
from lunaris.surrogate.st_lrps.training.preflight import (
    FAIL,
    PASS,
    SKIP,
    check_cuda_visibility,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "tools" / "hpc"))

import run_training_scenario as launcher  # noqa: E402

# --------------------------------------------------------------------------- #
# H2 — stop signals
# --------------------------------------------------------------------------- #


class TestStopSignals:
    def test_stop_signal_list_covers_sigint_and_sigterm(self):
        sigs = _stop_signal_list()
        assert int(signal.SIGINT) in sigs
        assert int(signal.SIGTERM) in sigs

    @pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 is POSIX-only")
    def test_stop_signal_list_includes_sigusr1_on_posix(self):
        assert int(signal.SIGUSR1) in _stop_signal_list()

    def test_install_and_restore_round_trip(self):
        calls: list[int] = []

        def handler(signum, _frame):
            calls.append(int(signum))

        originals = {sig: signal.getsignal(sig) for sig in _stop_signal_list()}
        installed = _install_stop_signal_handlers(handler)
        try:
            # Main-thread test process: every stop signal must be registered.
            assert set(installed) == set(_stop_signal_list())
            for sig in installed:
                assert signal.getsignal(sig) is handler
        finally:
            _restore_stop_signal_handlers(installed)
        for sig, orig in originals.items():
            assert signal.getsignal(sig) is orig

    def test_handler_receives_sigterm_semantics(self):
        # The loop's contract: first signal flips a flag, second raises.
        interrupt = {"flag": False}

        def on_stop(signum, frame):
            if interrupt["flag"]:
                raise KeyboardInterrupt
            interrupt["flag"] = True

        on_stop(int(signal.SIGTERM), None)
        assert interrupt["flag"] is True
        with pytest.raises(KeyboardInterrupt):
            on_stop(int(signal.SIGTERM), None)


# --------------------------------------------------------------------------- #
# H3 — RAM budget under cgroup / Slurm limits
# --------------------------------------------------------------------------- #


class TestRamBudget:
    def test_auto_preload_uses_streaming_when_ram_budget_is_unknown(self):
        should_preload, reason = _decide_preload(
            "auto",
            dataset_mb=10.0,
            auto_preload_mb=2048.0,
            est_ram_mb=20.0,
            avail_ram_mb=None,
        )
        assert should_preload is False
        assert "unknown RAM" in reason

    def test_explicit_preload_records_unknown_ram_warning(self):
        should_preload, reason = _decide_preload(
            "always",
            dataset_mb=10.0,
            auto_preload_mb=2048.0,
            est_ram_mb=20.0,
            avail_ram_mb=None,
        )
        assert should_preload is True
        assert "unknown RAM" in reason

    def test_auto_preload_vetoes_estimate_above_slurm_budget(self):
        should_preload, reason = _decide_preload(
            "auto",
            dataset_mb=100.0,
            auto_preload_mb=2048.0,
            est_ram_mb=20_000.0,
            avail_ram_mb=32_000.0,
        )
        assert should_preload is False
        assert "RAM safety veto" in reason

    def test_cgroup_v2_limit_minus_usage(self, tmp_path):
        limit = tmp_path / "memory.max"
        usage = tmp_path / "memory.current"
        limit.write_text(str(8 * 1024**3))   # 8 GiB
        usage.write_text(str(2 * 1024**3))   # 2 GiB used
        result = _cgroup_available_mb(limit_files=((str(limit), str(usage)),))
        assert result == pytest.approx(6 * 1024.0)

    def test_cgroup_v2_unlimited_is_none(self, tmp_path):
        limit = tmp_path / "memory.max"
        limit.write_text("max")
        assert _cgroup_available_mb(
            limit_files=((str(limit), str(tmp_path / "memory.current")),)
        ) is None

    def test_cgroup_v1_sentinel_is_none(self, tmp_path):
        limit = tmp_path / "memory.limit_in_bytes"
        limit.write_text(str(2**63))  # PAGE_COUNTER_MAX-style "no limit"
        assert _cgroup_available_mb(
            limit_files=((str(limit), str(tmp_path / "usage")),)
        ) is None

    def test_cgroup_missing_files_is_none(self, tmp_path):
        assert _cgroup_available_mb(
            limit_files=((str(tmp_path / "nope"), str(tmp_path / "nope2")),)
        ) is None

    def test_slurm_mem_per_node(self):
        assert _slurm_mem_limit_mb({"SLURM_MEM_PER_NODE": "32000"}) == pytest.approx(32000.0)

    def test_slurm_mem_per_node_with_suffix(self):
        assert _slurm_mem_limit_mb({"SLURM_MEM_PER_NODE": "32G"}) == pytest.approx(32 * 1024.0)

    def test_slurm_mem_per_cpu_times_cpus(self):
        env = {"SLURM_MEM_PER_CPU": "4000", "SLURM_CPUS_ON_NODE": "8"}
        assert _slurm_mem_limit_mb(env) == pytest.approx(32000.0)

    def test_slurm_absent_is_none(self):
        assert _slurm_mem_limit_mb({}) is None

    def test_available_ram_takes_min_of_signals(self, monkeypatch):
        import lunaris.surrogate.st_lrps.training.engine as eng

        monkeypatch.setattr(eng, "_cgroup_available_mb", lambda *a, **k: 4096.0)
        monkeypatch.setattr(eng, "_slurm_mem_limit_mb", lambda *a, **k: 32000.0)
        result = eng._available_ram_mb()
        assert result is not None
        # Never above the tightest (cgroup) limit, whatever psutil reports.
        assert result <= 4096.0

    def test_available_ram_none_when_no_signal(self, monkeypatch):
        import builtins

        import lunaris.surrogate.st_lrps.training.engine as eng

        monkeypatch.setattr(eng, "_cgroup_available_mb", lambda *a, **k: None)
        monkeypatch.setattr(eng, "_slurm_mem_limit_mb", lambda *a, **k: None)
        real_import = builtins.__import__

        def _no_psutil(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("psutil disabled for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_psutil)
        assert eng._available_ram_mb() is None

    def test_psutil_is_an_hpc_extra(self):
        text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        hpc_block = text.split("hpc = [", 1)[1].split("]", 1)[0]
        assert "psutil" in hpc_block


# --------------------------------------------------------------------------- #
# H4 — CUDA visibility preflight
# --------------------------------------------------------------------------- #


class TestCudaVisibility:
    def test_cuda_device_passes(self):
        check = check_cuda_visibility(device_type="cuda", environ={})
        assert check.status == PASS

    def test_cpu_without_demand_skips(self):
        check = check_cuda_visibility(device_type="cpu", environ={})
        assert check.status == SKIP

    def test_require_cuda_env_fails_on_cpu(self):
        check = check_cuda_visibility(
            device_type="cpu", environ={"LUNARIS_REQUIRE_CUDA": "1"}
        )
        assert check.status == FAIL

    def test_slurm_gpu_allocation_fails_on_cpu(self):
        check = check_cuda_visibility(
            device_type="cpu", environ={"SLURM_JOB_GPUS": "0"}
        )
        assert check.status == FAIL
        assert "SLURM_JOB_GPUS" in check.detail

    def test_cuda_visible_devices_disabled_does_not_demand(self):
        for value in ("", "-1"):
            check = check_cuda_visibility(
                device_type="cpu", environ={"CUDA_VISIBLE_DEVICES": value}
            )
            assert check.status == SKIP, value

    def test_check_is_wired_into_run_preflight(self, tmp_path, monkeypatch):
        from lunaris.surrogate.st_lrps.training.preflight import run_preflight

        monkeypatch.setenv("LUNARIS_REQUIRE_CUDA", "1")
        report = run_preflight(out_dir=tmp_path, device_type="cpu")
        by_name = {c.name: c for c in report.checks}
        assert by_name["cuda_visibility"].status == FAIL
        assert report.go is False

    def test_shell_preflight_has_the_same_fail_closed_gpu_gate(self):
        text = (_REPO_ROOT / "hpc" / "preflight.sh").read_text(encoding="utf-8")
        assert "LUNARIS_REQUIRE_CUDA" in text
        assert "sys.exit(4)" in text
        assert "${torch_rc} -eq 4" in text


# --------------------------------------------------------------------------- #
# H5 — scenario launcher --resume
# --------------------------------------------------------------------------- #


def _write_scenario_file(tmp_path: Path) -> Path:
    scenario = {
        "name": "ResumeDemo_Scenario",
        "entrypoint": "lunaris-train",
        "description": "resume-mode launcher test scenario",
        "runtime_model_kind": "potential_autograd",
        "tags": ["test"],
        "flags": ["--seed", "42"],
    }
    path = tmp_path / "scenarios.jsonl"
    path.write_text(json.dumps(scenario) + "\n", encoding="utf-8")
    return path


class TestLauncherResume:
    def test_find_resumable_checkpoint_prefers_last(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "ckpt_best.pt").write_bytes(b"x")
        (ckpt_dir / "ckpt_last.pt").write_bytes(b"x")
        found = launcher.find_resumable_checkpoint(tmp_path)
        assert found is not None and found.name == "ckpt_last.pt"

    def test_find_resumable_checkpoint_none_without_files(self, tmp_path):
        assert launcher.find_resumable_checkpoint(tmp_path) is None

    def test_dry_run_resume_uses_resume_from(self, tmp_path, capsys):
        scenario_file = _write_scenario_file(tmp_path)
        out_root = tmp_path / "training"
        run_dir = out_root / "ResumeDemo_Scenario"
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "checkpoints" / "ckpt_last.pt").write_bytes(b"x")

        rc = launcher.main([
            str(scenario_file), "--index", "0", "--dry-run", "--resume",
            "--output-root", str(out_root), "--epochs", "100",
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "mode        : resume (from ckpt_last.pt)" in out
        assert "--resume-from" in out
        assert "--out " not in out  # --out must be replaced, not duplicated

    def test_dry_run_resume_without_prior_run_is_fresh(self, tmp_path, capsys):
        scenario_file = _write_scenario_file(tmp_path)
        rc = launcher.main([
            str(scenario_file), "--index", "0", "--dry-run", "--resume",
            "--output-root", str(tmp_path / "training"),
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "mode        : fresh" in out
        assert "--out" in out

    def test_resume_with_force_is_an_error(self, tmp_path, capsys):
        scenario_file = _write_scenario_file(tmp_path)
        rc = launcher.main([
            str(scenario_file), "--index", "0", "--dry-run",
            "--resume", "--force", "--output-root", str(tmp_path / "training"),
        ])
        err = capsys.readouterr().err
        assert rc == 2
        assert "mutually exclusive" in err

    def test_resume_on_nonempty_dir_without_checkpoint_is_an_error(self, tmp_path, capsys):
        scenario_file = _write_scenario_file(tmp_path)
        out_root = tmp_path / "training"
        run_dir = out_root / "ResumeDemo_Scenario"
        run_dir.mkdir(parents=True)
        (run_dir / "stale.txt").write_text("leftover", encoding="utf-8")

        rc = launcher.main([
            str(scenario_file), "--index", "0", "--dry-run", "--resume",
            "--output-root", str(out_root),
        ])
        err = capsys.readouterr().err
        assert rc == 2
        assert "no resumable checkpoint" in err

    def test_production_scenario_keeps_trainconfig_default_architecture(self):
        """H6 invariant: the strong-model scenario must not override any
        architecture-critical field — TrainConfig defaults ARE the profile."""
        path = _REPO_ROOT / "hpc" / "scenarios" / "st_lrps_strong_model_production.jsonl"
        scenarios = launcher.load_scenarios(path)
        assert len(scenarios) == 1
        scenario = scenarios[0]
        launcher.validate_scenario(scenario)
        flag_heads = {tok.split("=", 1)[0] for tok in scenario["flags"] if tok.startswith("--")}
        arch_flags = {
            "--model-preset", "--hidden", "--depth", "--activation", "--dropout",
            "--w0-first", "--w0-hidden", "--n-bands", "--multiscale-mode",
            "--use-residual-blocks", "--no-residual-blocks", "--use-fourier",
            "--use-sh-encoding", "--use-radial-separation",
            "--use-radial-decay-encoding", "--use-physical-radial-decay-encoding",
            "--use-real-sh-basis", "--output-dim",
        }
        assert not (flag_heads & arch_flags), (
            f"production scenario overrides architecture flags: {flag_heads & arch_flags}"
        )
        assert "--run-preset" in flag_heads  # paper posture is the point
        assert scenario["flags"][scenario["flags"].index("--run-preset") + 1] == "paper"

        from lunaris.surrogate.st_lrps.networks.models import compute_architecture_signature
        from lunaris.surrogate.st_lrps.training.config import (
            TrainConfig,
            apply_model_preset,
            apply_run_preset,
        )

        default_cfg = TrainConfig(data="train.h5", out="run-default")
        apply_model_preset(default_cfg)
        production_cfg = TrainConfig(data="train.h5", out="run-production", seed=42, run_preset="paper")
        apply_run_preset(production_cfg)
        apply_model_preset(production_cfg)
        assert compute_architecture_signature(production_cfg) == compute_architecture_signature(default_cfg)

    def test_resume_denemesi_reproduction_scenario_pins_historical_model(self):
        """The reproduction recipe must not drift with TrainConfig defaults."""
        path = (
            _REPO_ROOT
            / "hpc"
            / "scenarios"
            / "st_lrps_resume_denemesi_reproduction.jsonl"
        )
        scenarios = launcher.load_scenarios(path)
        assert len(scenarios) == 1
        scenario = scenarios[0]
        launcher.validate_scenario(scenario)

        flags = scenario["flags"]

        def value(name: str) -> str:
            return flags[flags.index(name) + 1]

        assert value("--runtime-model-kind") == "potential_autograd"
        assert value("--model-preset") == "recommended_physical_radial_decay"
        assert value("--hidden") == "512"
        assert value("--depth") == "5"
        assert value("--n-bands") == "2"
        assert value("--epochs") == "400"
        assert value("--batch-size") == "8192"
        assert value("--lr") == "7e-5"
        assert value("--fit-rows") == "2000000"
        assert value("--direction-loss-ramp-epochs") == "70"
        assert value("--n-hutchinson-samples") == "2"
        assert "--deterministic" in flags
        assert "--no-amp" in flags

        from lunaris.surrogate.st_lrps.networks.models import build_model_from_config
        from lunaris.surrogate.st_lrps.training.config import TrainConfig, apply_model_preset

        cfg = TrainConfig(
            data="train.h5",
            out="run-reproduction",
            hidden=int(value("--hidden")),
            depth=int(value("--depth")),
            activation=value("--activation"),
            dropout=float(value("--dropout")),
            w0_first=float(value("--w0-first")),
            w0_hidden=float(value("--w0-hidden")),
            model_preset=value("--model-preset"),
            runtime_model_kind=value("--runtime-model-kind"),
            use_residual_blocks=True,
            n_bands=int(value("--n-bands")),
            multiscale_mode=value("--multiscale-mode"),
            degree_min=20,
            degree_max=200,
            x_scale_m=2_738_000.0,
            resolved_r_ref_m=1_738_000.0,
        )
        apply_model_preset(cfg)
        model = build_model_from_config(cfg)
        assert sum(p.numel() for p in model.parameters()) == 1_848_321
        assert model.backbone.w0_bands == pytest.approx([13.7, 42.4])

    def test_submit_helper_supports_cluster_resource_overrides(self):
        text = (_REPO_ROOT / "hpc" / "submit.sh").read_text(encoding="utf-8")
        for variable, flag in (
            ("LUNARIS_CPUS_PER_TASK", "--cpus-per-task="),
            ("LUNARIS_MEM", "--mem="),
            ("LUNARIS_TIME", "--time="),
            ("LUNARIS_SIGNAL", "--signal="),
        ):
            assert variable in text
            assert flag in text

    def test_fresh_collision_error_mentions_resume(self, tmp_path, capsys, monkeypatch):
        scenario_file = _write_scenario_file(tmp_path)
        out_root = tmp_path / "training"
        run_dir = out_root / "ResumeDemo_Scenario"
        run_dir.mkdir(parents=True)
        (run_dir / "stale.txt").write_text("leftover", encoding="utf-8")

        rc = launcher.main([
            str(scenario_file), "--index", "0",
            "--output-root", str(out_root),
        ])
        err = capsys.readouterr().err
        assert rc == 3
        assert "--resume" in err
