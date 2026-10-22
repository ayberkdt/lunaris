"""Adjoint-weighted arc loss (training/arc_loss.py): exact error, unbiased
gradient estimate, contract checks and trainer wiring through the CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.training.arc_loss import ArcLossTerm  # noqa: E402


def _write_proxy(root: Path, n_orbits: int = 3, m: int = 200, seed: int = 0) -> Path:
    g = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for k in range(n_orbits):
        np.savez(
            root / f"orbit_{k:06d}.npz",
            r_fix=g.normal(size=(m, 3)),
            a_true=g.normal(size=(m, 3)),
            weight=g.normal(size=(m, 3)) / m,
            hp_km=100.0,
        )
    (root / "proxy_manifest.json").write_text(json.dumps({"base_degree": 25, "ref_degree": 200}), encoding="utf-8")
    return root


class _Linear(torch.nn.Module):
    """Acceleration model a(x) = A x + b with learnable A, b (float64)."""

    def __init__(self) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.A = torch.nn.Parameter(torch.randn(3, 3, generator=g, dtype=torch.float64))
        self.b = torch.nn.Parameter(torch.randn(3, generator=g, dtype=torch.float64))

    def accel(self, x: torch.Tensor, create_graph: bool) -> torch.Tensor:
        out = x @ self.A.T + self.b
        return out if create_graph else out.detach()


def _exact_s(model: _Linear, orbit) -> torch.Tensor:
    r, a_true, w = orbit
    return (w * (model.accel(r, True) - a_true)).sum()


def test_exact_error_matches_the_direct_sum(tmp_path):
    term = ArcLossTerm(_write_proxy(tmp_path / "p"), chunk=37)
    model = _Linear()
    for k in range(3):
        expected = float(_exact_s(model, term.orbits[k]))
        assert term.exact_error(model.accel, k, torch.device("cpu"), torch.float64) == pytest.approx(expected, rel=1e-12)


def test_subsampled_gradient_is_unbiased_for_half_mean_square_error(tmp_path):
    term = ArcLossTerm(_write_proxy(tmp_path / "p", n_orbits=1, m=200), subsample=50, orbits_per_step=1, seed=3)
    model = _Linear()
    # Reference gradient of 1/2 s^2 with the full orbit.
    model.zero_grad()
    (0.5 * _exact_s(model, term.orbits[0]) ** 2).backward()
    g_ref = torch.cat([model.A.grad.flatten(), model.b.grad.flatten()]).clone()
    # Average of the estimator's gradients over many subsamples.
    acc = torch.zeros_like(g_ref)
    n = 3000
    for _ in range(n):
        model.zero_grad()
        loss, stats = term.term(model.accel, torch.device("cpu"), torch.float64)
        loss.backward()
        acc += torch.cat([model.A.grad.flatten(), model.b.grad.flatten()])
    g_est = acc / n
    assert torch.linalg.norm(g_est - g_ref) < 0.05 * torch.linalg.norm(g_ref)
    assert stats["arc_s_sq_mean_m2"] == pytest.approx(float(_exact_s(model, term.orbits[0])) ** 2, rel=1e-10)


def test_degree_mismatch_and_empty_dirs_are_rejected(tmp_path):
    p = _write_proxy(tmp_path / "p")
    with pytest.raises(ValueError, match="base_degree"):
        ArcLossTerm(p, expected_base_degree=20)
    with pytest.raises(ValueError, match="ref_degree"):
        ArcLossTerm(p, expected_ref_degree=150)
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "proxy_manifest.json").write_text(json.dumps({"base_degree": 25, "ref_degree": 200}))
    with pytest.raises(ValueError, match="no orbit"):
        ArcLossTerm(tmp_path / "empty")


def test_arc_flags_reach_the_train_config(monkeypatch, tmp_path):
    from lunaris.surrogate.st_lrps.training.config import parse_args

    monkeypatch.setattr(sys, "argv", [
        "lunaris-train", "--train-data", "t.h5", "--val-data", "v.h5",
        "--arc-loss-data", str(tmp_path), "--arc-loss-weight", "1e-6", "--arc-loss-every", "25",
        "--arc-loss-orbits", "3", "--arc-loss-subsample", "2048", "--arc-loss-s-ref-m", "0.5",
    ])
    cfg = parse_args()
    assert (cfg.arc_loss_data, cfg.arc_loss_weight, cfg.arc_loss_every) == (str(tmp_path), 1e-6, 25)
    assert (cfg.arc_loss_orbits, cfg.arc_loss_subsample, cfg.arc_loss_s_ref_m) == (3, 2048, 0.5)
    monkeypatch.setattr(sys, "argv", ["lunaris-train", "--train-data", "t.h5", "--val-data", "v.h5"])
    assert parse_args().arc_loss_weight == 0.0


def test_evaluate_all_reports_geometric_mean_and_rms(tmp_path):
    term = ArcLossTerm(_write_proxy(tmp_path / "p", n_orbits=3))
    model = _Linear()
    s = np.array([float(_exact_s(model, o)) for o in term.orbits])
    out = term.evaluate_all(model.accel, torch.device("cpu"), torch.float64)
    assert out["arc_rms_m"] == pytest.approx(float(np.sqrt(np.mean(s**2))), rel=1e-10)
    assert out["arc_gmean_m"] == pytest.approx(float(np.exp(np.mean(np.log(np.abs(s))))), rel=1e-6)


def test_arc_val_checkpoint_metric():
    from lunaris.surrogate.st_lrps.training.metrics import compute_checkpoint_score

    stats = {"val_total_loss": 1.0, "val_loss_u": 0.1, "val_loss_a": 0.2, "arc_gmean_m": 0.02}
    score, report = compute_checkpoint_score(stats, {"best_metric": "arc_val"})
    assert score == 0.02 and report["formula"] == "val_arc_gmean_m"
    with pytest.raises(ValueError, match="arc_val"):
        compute_checkpoint_score({**stats, "arc_gmean_m": None}, {"best_metric": "arc_val"})
