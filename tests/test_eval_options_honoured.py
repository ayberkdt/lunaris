"""Evaluation CLI options must do what the help says, or refuse loudly.

Each test pins one silent trap found in the pre-run audit of
``lunaris.surrogate.st_lrps.evaluation.cli``: an option that was accepted but
dropped, shadowed, remapped or overwritten, so the report described a different
evaluation than the one that was asked for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI  # noqa: E402
from lunaris.surrogate.st_lrps.evaluation import cli as eval_cli  # noqa: E402
from lunaris.surrogate.st_lrps.evaluation.cli import evaluate  # noqa: E402


def _make_h5(path: Path, n: int = 40, *, dset: str = "data") -> Path:
    rng = np.random.default_rng(5)
    r = R_MOON_SI + rng.uniform(200e3, 500e3, n)
    theta = rng.uniform(0.1, np.pi - 0.1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    u = 1e2 * np.cos(theta)
    a = 1e-4 * np.stack([np.cos(phi), np.sin(phi), np.cos(theta)], axis=1)
    data = np.column_stack([x, y, z, u, a]).astype(np.float64)
    with h5py.File(str(path), "w") as hf:
        hf.create_dataset(dset, data=data)
        hf.attrs["unit_system"] = "SI"
        hf.attrs["mu_si"] = float(MU_MOON_SI)
        hf.attrs["r_ref_m"] = float(R_MOON_SI)
        hf.attrs["central_body"] = "moon"
        hf.attrs["target_mode"] = "residual"
        hf.attrs["degree_min"] = 20
        hf.attrs["degree_max"] = 60
        hf.attrs["requested_degree"] = 60
        hf.attrs["alt_min_km"] = 200.0
        hf.attrs["alt_max_km"] = 500.0
        hf.attrs["columns"] = "[x,y,z,U,ax,ay,az]"
    return path


def _make_run(root: Path) -> Path:
    """Run dir with BOTH checkpoints; ``ckpt_last`` is distinguishable by kind/epoch/EMA."""
    from st_lrps_contract_test_utils import make_contract_run

    run_dir = Path(make_contract_run(root, alt_min_km=100.0, alt_max_km=1000.0)["run_dir"])
    best = run_dir / "checkpoints" / "ckpt_best.pt"
    payload = torch.load(best, map_location="cpu", weights_only=False)
    payload["ema_state_dict"] = {
        k: (v + 0.01 if v.is_floating_point() else v) for k, v in payload["model_state_dict"].items()
    }
    torch.save(payload, best)
    last = dict(payload)
    last["kind"] = "last"
    last["epoch"] = 7
    last["epoch_display"] = 8
    torch.save(last, run_dir / "checkpoints" / "ckpt_last.pt")
    return run_dir


def _evaluate(run_dir: Path, data: Path, out: Path, **kw: Any):
    kw.setdefault("alt_bin_km", 50.0)
    return evaluate(
        model_dir=run_dir, data_path=data, out_dir=out, device=torch.device("cpu"),
        batch_size=8, a_sign=1.0, r_ref_m=R_MOON_SI, **kw,
    )


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["lunaris-eval", "--device", "cpu", "--batch-size", "8", *argv])
    eval_cli.main()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. --model-dir <file>.pt evaluates exactly that file
# ---------------------------------------------------------------------------

def test_model_dir_checkpoint_file_evaluates_that_file(tmp_path: Path, monkeypatch) -> None:
    run_dir = _make_run(tmp_path / "run")
    data = _make_h5(tmp_path / "d.h5")
    out = tmp_path / "out"
    last = run_dir / "checkpoints" / "ckpt_last.pt"
    _main(monkeypatch, "--model-dir", str(last), "--data", str(data), "--out", str(out))
    report = _read(out / "eval_report.json")
    assert Path(report["artifacts"]["checkpoint"]).name == "ckpt_last.pt"
    assert report["metrics"]["checkpoint_kind"] == "last"
    assert report["metrics"]["checkpoint_epoch"] == 7
    manifest = _read(out / "eval_manifest.json")
    assert Path(manifest["checkpoint_path"]).name == "ckpt_last.pt"
    assert manifest["checkpoint_kind"] == "last"
    assert manifest["checkpoint_epoch"] == 7


def test_model_dir_run_dir_still_prefers_best(tmp_path: Path, monkeypatch) -> None:
    run_dir = _make_run(tmp_path / "run")
    data = _make_h5(tmp_path / "d.h5")
    out = tmp_path / "out"
    _main(monkeypatch, "--model-dir", str(run_dir), "--data", str(data), "--out", str(out))
    assert Path(_read(out / "eval_manifest.json")["checkpoint_path"]).name == "ckpt_best.pt"


# ---------------------------------------------------------------------------
# 2. --use-ema is recorded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("streaming", [False, True])
def test_weights_source_is_recorded(tmp_path: Path, streaming: bool) -> None:
    run_dir = _make_run(tmp_path / "run")
    data = _make_h5(tmp_path / "d.h5")
    _evaluate(run_dir, data, tmp_path / "raw", streaming=streaming)
    _evaluate(run_dir, data, tmp_path / "ema", streaming=streaming, use_ema=True)
    for name, expected in (("raw", "raw"), ("ema", "ema")):
        assert _read(tmp_path / name / "eval_report.json")["weights_source"] == expected
        assert _read(tmp_path / name / "eval_manifest.json")["weights_source"] == expected


# ---------------------------------------------------------------------------
# 3. A mistyped --dataset-name is an error, not a silent fallback
# ---------------------------------------------------------------------------

def test_mistyped_dataset_name_raises_and_lists_available(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path / "run")
    data = _make_h5(tmp_path / "d.h5", dset="cloud")
    with pytest.raises((KeyError, ValueError), match="datsa") as excinfo:
        _evaluate(run_dir, data, tmp_path / "out", dataset_name="datsa")
    assert "cloud" in str(excinfo.value)


def test_no_dataset_name_still_auto_picks_first_2d_dataset(tmp_path: Path, monkeypatch) -> None:
    run_dir = _make_run(tmp_path / "run")
    data = _make_h5(tmp_path / "d.h5", dset="cloud")
    out = tmp_path / "out"
    _main(monkeypatch, "--model-dir", str(run_dir), "--data", str(data), "--out", str(out))
    assert _read(out / "eval_report.json")["n_evaluated"] == 40


# ---------------------------------------------------------------------------
# 4. --ood-data and --ood-high-data are both evaluated, to their own dirs
# ---------------------------------------------------------------------------

def test_ood_data_and_ood_high_data_are_both_evaluated(tmp_path: Path, monkeypatch) -> None:
    run_dir = _make_run(tmp_path / "run")
    data = _make_h5(tmp_path / "d.h5")
    ood = _make_h5(tmp_path / "ood.h5", n=24)
    ood_high = _make_h5(tmp_path / "ood_high.h5", n=16)
    out = tmp_path / "out"
    _main(
        monkeypatch, "--model-dir", str(run_dir), "--data", str(data), "--out", str(out),
        "--ood-data", str(ood), "--ood-high-data", str(ood_high),
    )
    assert _read(out / "ood" / "eval_report.json")["artifacts"]["data_path"] == str(ood.resolve())
    assert _read(out / "ood_high" / "eval_report.json")["artifacts"]["data_path"] == str(ood_high.resolve())
    splits = {row["split"] for row in _read(out / "summary_metrics.json")}
    assert {"ood", "ood_high"} <= splits


def test_ood_data_alone_goes_to_ood_dir(tmp_path: Path, monkeypatch) -> None:
    run_dir = _make_run(tmp_path / "run")
    data = _make_h5(tmp_path / "d.h5")
    ood = _make_h5(tmp_path / "ood.h5", n=24)
    out = tmp_path / "out"
    _main(monkeypatch, "--model-dir", str(run_dir), "--data", str(data), "--out", str(out),
          "--ood-data", str(ood))
    assert (out / "ood" / "eval_report.json").is_file()
    assert not (out / "ood_high").exists()


# ---------------------------------------------------------------------------
# 5. --alt-bin-km is honoured under --streaming (and the width used is recorded)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("width_km", "n_bins"), [(100.0, 3), (40.0, 8)])
def test_streaming_honours_alt_bin_width(tmp_path: Path, width_km: float, n_bins: int) -> None:
    run_dir = _make_run(tmp_path / "run")
    data = _make_h5(tmp_path / "d.h5")
    metrics = _evaluate(run_dir, data, tmp_path / "out", streaming=True, alt_bin_km=width_km)
    assert metrics["altitude_bin_width_km"] == pytest.approx(width_km)
    assert len(metrics["altitude_bin_count"]) == n_bins
    assert sum(metrics["altitude_bin_count"]) == 40
    lo, hi = metrics["altitude_bin_range_km"]
    assert hi - lo == pytest.approx(n_bins * width_km)
    assert metrics["altitude_underflow_count"] == 0
    assert metrics["altitude_overflow_count"] == 0


# ---------------------------------------------------------------------------
# 6. The suite writer must not clobber the primary split's own artifacts
# ---------------------------------------------------------------------------

def test_out_suite_does_not_overwrite_primary_manifest_and_altitude_csv(tmp_path: Path, monkeypatch) -> None:
    run_dir = _make_run(tmp_path / "run")
    data = _make_h5(tmp_path / "d.h5")
    out = tmp_path / "out"
    _main(monkeypatch, "--model-dir", str(run_dir), "--data", str(data), "--out", str(out))

    primary = _read(out / "eval_manifest.json")
    assert primary["schema_version"] == "st_lrps_eval_manifest_v1"
    assert primary["checkpoint_hash"]
    assert primary["checkpoint_kind"] == "best"

    header = (out / "altitude_binned_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "rmse_a_mag" in header
    assert "mape" in header.lower()

    suite = _read(out / "eval_suite_manifest.json")
    assert suite["schema_version"] == "st_lrps_eval_suite_v1"
    assert suite["splits"][0]["split"] == "primary"
    assert Path(suite["altitude_binned_metrics_csv"]).name == "altitude_binned_metrics_suite.csv"
    assert Path(suite["altitude_binned_metrics_csv"]).is_file()


# ---------------------------------------------------------------------------
# 7. --topk-errors 0 means 0; plot limits do not shadow each other
# ---------------------------------------------------------------------------

def test_topk_errors_zero_disables_and_n_is_honoured(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path / "run")
    data = _make_h5(tmp_path / "d.h5")
    _evaluate(run_dir, data, tmp_path / "k5", topk_errors=5)
    assert len((tmp_path / "k5" / "topk_worst.csv").read_text().strip().splitlines()) == 1 + 5
    _evaluate(run_dir, data, tmp_path / "k0", topk_errors=0)
    assert not (tmp_path / "k0" / "topk_worst.csv").exists()
    assert _read(tmp_path / "k0" / "eval_report.json")["topk_export_path"] is None


def test_topk_errors_default_still_exports_worst_cases(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["lunaris-eval"])
    assert eval_cli.parse_args().topk_errors == 100


def test_save_error_points_requires_positive_topk(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path / "run")
    data = _make_h5(tmp_path / "d.h5")
    with pytest.raises(ValueError, match="topk"):
        _evaluate(run_dir, data, tmp_path / "out", topk_errors=0, save_error_points=tmp_path / "e.csv")


def _plot_points_seen(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []

    def _capture(*args: Any, **kwargs: Any) -> None:
        seen.append(int(sum(a.shape[0] for a in args[10])))  # ang_deg_plot buffers

    monkeypatch.setattr(eval_cli, "_save_evaluation_plots", _capture)
    return seen


def test_max_points_for_plots_is_not_shadowed_by_plot_sample_limit(tmp_path: Path, monkeypatch) -> None:
    run_dir = _make_run(tmp_path / "run")
    data = _make_h5(tmp_path / "d.h5")
    seen = _plot_points_seen(monkeypatch)
    _evaluate(run_dir, data, tmp_path / "cap", max_points_for_plots=10, plot_sample_limit=None)
    _evaluate(run_dir, data, tmp_path / "all")
    assert seen[0] == 16  # batches of 8 are kept until >= 10 points are buffered
    assert seen[1] == 40


def test_conflicting_explicit_plot_limits_are_refused(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path / "run")
    data = _make_h5(tmp_path / "d.h5")
    with pytest.raises(ValueError, match="plot"):
        _evaluate(run_dir, data, tmp_path / "out", max_points_for_plots=10, plot_sample_limit=20)
