# -*- coding: utf-8 -*-
"""
Regression tests for the surrogate gravity runtime wrapper.

These tests build a tiny synthetic checkpoint on the fly so we can verify the
desktop/runtime integration without depending on the large experimental runs
stored in the repository.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.physics.surrogate_gravity import (
    SurrogateGravityModel,
    _build_model_from_config,
    _extract_degree_metadata,
    _is_valid_surrogate_run,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_tiny_run(
    tmp_path: Path,
    run_name: str,
    extra_config: dict | None = None,
    ckpt_name: str = "ckpt_best.pt",
) -> Path:
    """Build a minimal synthetic ST-LRPS run directory."""
    run_dir = tmp_path / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    config = {
        "hidden": 8,
        "depth": 1,
        "activation": "tanh",
        "dropout": 0.0,
        "resolved_mu_si": float(MU_MOON),
        "resolved_a_sign": 1.0,
        "scaler_kind": "isometric",
        "degree_min": 0,
        "degree_max": 50,
    }
    if extra_config:
        config.update(extra_config)

    scaler = {
        "x": {"mean": [0.0, 0.0, 0.0], "scale": 2_000_000.0},
        "u": {"mean": [0.0], "scale": 1.0},
        "a": {"mean": [0.0, 0.0, 0.0], "scale": 1.0},
    }

    model = _build_model_from_config(config)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    torch.save(
        {"model": model.state_dict(), "config": config, "scaler": scaler},
        ckpt_dir / ckpt_name,
    )
    return run_dir


# ---------------------------------------------------------------------------
# Existing regression tests (updated to include required degree_max)
# ---------------------------------------------------------------------------

def test_surrogate_gravity_residual_mode_reduces_to_point_mass_when_network_is_zero(tmp_path: Path) -> None:
    run_dir = _make_tiny_run(tmp_path, "run_residual")

    runtime = SurrogateGravityModel.from_model_dir(
        run_dir,
        mu_override=float(MU_MOON),
        r_ref_override=float(R_MOON),
        device_preference="cpu",
    )

    r = float(R_MOON + 100_000.0)
    accel = runtime.acceleration_fixed(np.array([r, 0.0, 0.0], dtype=np.float64))
    expected = np.array([-float(MU_MOON) / (r * r), 0.0, 0.0], dtype=np.float64)

    assert runtime.training_mode == "residual_potential"
    assert np.allclose(accel, expected, rtol=1e-5, atol=1e-10)


def test_surrogate_gravity_rejects_obviously_wrong_absolute_body_scale(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_absolute"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    config = {
        "hidden": 8,
        "depth": 1,
        "activation": "tanh",
        "dropout": 0.0,
        "a_sign": 1.0,
        "degree_min": 10,
        "degree_max": 50,
    }
    scaler = {
        "x": {"mean": [0.0, 0.0, 0.0], "std": [2_000_000.0, 2_000_000.0, 2_000_000.0]},
        "u": {"mean": [6.0e7], "std": [1.0]},
        "a": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
    }

    model = _build_model_from_config(config)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "config": config, "scaler": scaler}, ckpt_dir / "ckpt_best.pt")

    with pytest.raises(ValueError, match="incompatible"):
        SurrogateGravityModel.from_model_dir(
            run_dir,
            mu_override=float(MU_MOON),
            r_ref_override=float(R_MOON),
            device_preference="cpu",
        )


# ---------------------------------------------------------------------------
# Degree metadata contract
# ---------------------------------------------------------------------------

class TestExtractDegreeMetadata:
    """Unit tests for _extract_degree_metadata()."""

    def test_top_level_keys(self) -> None:
        cfg = {"degree_min": 10, "degree_max": 50}
        deg_min, deg_max = _extract_degree_metadata(cfg)
        assert deg_min == 10
        assert deg_max == 50

    def test_dataset_meta_fallback(self) -> None:
        cfg = {"dataset_meta": {"degree_min": 5, "degree_max": 100}}
        deg_min, deg_max = _extract_degree_metadata(cfg)
        assert deg_min == 5
        assert deg_max == 100

    def test_requested_degree_last_resort(self) -> None:
        cfg = {"dataset_meta": {"requested_degree": 200}}
        _, deg_max = _extract_degree_metadata(cfg)
        assert deg_max == 200

    def test_top_level_wins_over_dataset_meta(self) -> None:
        cfg = {
            "degree_min": 10,
            "degree_max": 50,
            "dataset_meta": {"degree_min": 1, "degree_max": 999},
        }
        deg_min, deg_max = _extract_degree_metadata(cfg)
        assert deg_min == 10
        assert deg_max == 50

    def test_missing_degree_max_raises(self) -> None:
        with pytest.raises(ValueError, match="degree_max"):
            _extract_degree_metadata({"degree_min": 10})

    def test_empty_config_raises(self) -> None:
        with pytest.raises(ValueError, match="degree_max"):
            _extract_degree_metadata({})

    def test_degree_min_defaults_to_zero_when_absent(self) -> None:
        cfg = {"degree_max": 50}
        deg_min, deg_max = _extract_degree_metadata(cfg)
        assert deg_min == 0
        assert deg_max == 50


class TestSurrogateGravityModelDegreeAttributes:
    """Verify that the loaded runtime wrapper exposes all degree metadata."""

    def test_degree_attributes_exposed(self, tmp_path: Path) -> None:
        run_dir = _make_tiny_run(tmp_path, "run_deg", extra_config={"degree_min": 20, "degree_max": 100})
        model = SurrogateGravityModel.from_model_dir(
            run_dir,
            mu_override=float(MU_MOON),
            r_ref_override=float(R_MOON),
            device_preference="cpu",
        )
        assert model.degree_min == 20
        assert model.degree_max == 100
        assert model.base_degree == 20
        assert model.target_degree == 100
        assert model.effective_degree_max == 100

    def test_degree_max_satisfies_propagator_contract(self, tmp_path: Path) -> None:
        """Confirm _get_sh_degree() in core.propagator no longer raises for ST-LRPS."""
        run_dir = _make_tiny_run(tmp_path, "run_propagator_compat")
        sgm = SurrogateGravityModel.from_model_dir(
            run_dir,
            mu_override=float(MU_MOON),
            r_ref_override=float(R_MOON),
            device_preference="cpu",
        )

        # Simulate what core.propagator._get_sh_degree() does.
        assert hasattr(sgm, "degree_max"), "degree_max must be present for propagator"
        assert int(sgm.degree_max) > 0

    def test_missing_degree_max_raises_at_load_time(self, tmp_path: Path) -> None:
        """degree_max missing → ValueError at from_model_dir(), not inside MC loop."""
        run_dir = _make_tiny_run(
            tmp_path, "run_no_degree",
            extra_config={"degree_min": None, "degree_max": None},
        )
        # Overwrite config.json without degree keys.
        config_no_degree = {
            "hidden": 8, "depth": 1, "activation": "tanh", "dropout": 0.0,
            "resolved_mu_si": float(MU_MOON), "resolved_a_sign": 1.0,
            "scaler_kind": "isometric",
        }
        (run_dir / "config.json").write_text(
            json.dumps(config_no_degree, indent=2), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="degree_max"):
            SurrogateGravityModel.from_model_dir(
                run_dir,
                mu_override=float(MU_MOON),
                r_ref_override=float(R_MOON),
                device_preference="cpu",
            )


# ---------------------------------------------------------------------------
# ckpt_last.pt fallback
# ---------------------------------------------------------------------------

class TestCkptLastFallback:
    """Verify that ckpt_last.pt is accepted as a fallback with a warning."""

    def test_ckpt_last_accepted_with_warning(self, tmp_path: Path) -> None:
        run_dir = _make_tiny_run(tmp_path, "run_last", ckpt_name="ckpt_last.pt")
        assert (run_dir / "checkpoints" / "ckpt_last.pt").exists()
        assert not (run_dir / "checkpoints" / "ckpt_best.pt").exists()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = SurrogateGravityModel.from_model_dir(
                run_dir,
                mu_override=float(MU_MOON),
                r_ref_override=float(R_MOON),
                device_preference="cpu",
            )

        warning_msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        assert any("ckpt_last.pt" in m for m in warning_msgs), (
            "Expected a RuntimeWarning mentioning ckpt_last.pt"
        )
        assert model.degree_max == 50  # from _make_tiny_run default

    def test_is_valid_surrogate_run_accepts_ckpt_last(self, tmp_path: Path) -> None:
        run_dir = _make_tiny_run(tmp_path, "run_valid_last", ckpt_name="ckpt_last.pt")
        assert _is_valid_surrogate_run(run_dir)

    def test_is_valid_surrogate_run_requires_config_json(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "incomplete"
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "checkpoints" / "ckpt_best.pt").touch()
        assert not _is_valid_surrogate_run(run_dir)

    def test_is_valid_surrogate_run_requires_checkpoint(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "no_ckpt"
        run_dir.mkdir()
        (run_dir / "config.json").write_text("{}", encoding="utf-8")
        assert not _is_valid_surrogate_run(run_dir)


# ---------------------------------------------------------------------------
# Naming: no HNN references in active MC state
# ---------------------------------------------------------------------------

def test_no_hnn_naming_in_surrogate_gravity_module() -> None:
    """ST-LRPS module must not re-export HNN names in runtime paths."""
    import lunaris.physics.surrogate_gravity as sgm_mod
    public_names = [n for n in dir(sgm_mod) if not n.startswith("_")]
    hnn_names = [n for n in public_names if "hnn" in n.lower()]
    assert hnn_names == [], f"HNN names found in models.surrogate_gravity: {hnn_names}"
