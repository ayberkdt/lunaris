from pathlib import Path

import pytest

from lunaris.common import GravityConfig
from lunaris.common.montecarlo_defs import MonteCarloConfig, validate_st_lrps_model_dir


def test_gravity_config_backend_aware():
    # Valid classic_sh
    cfg = GravityConfig(file_path="dummy.txt", backend="classic_sh")
    assert cfg.file_path == "dummy.txt"

    # Invalid classic_sh
    with pytest.raises(ValueError, match="file_path cannot be empty when backend='classic_sh'"):
        GravityConfig(file_path="", backend="classic_sh")

    # Valid st_lrps
    cfg = GravityConfig(file_path="", backend="st_lrps", st_lrps_model_dir="mock_dir")
    assert cfg.st_lrps_model_dir == "mock_dir"

    # Invalid st_lrps (missing dir)
    with pytest.raises(ValueError, match="st_lrps_model_dir cannot be empty when backend='st_lrps'"):
        GravityConfig(file_path="dummy.txt", backend="st_lrps", st_lrps_model_dir="")


def test_monte_carlo_config_no_fs_check():
    # Should not raise any filesystem errors on construction
    cfg = MonteCarloConfig(
        n_samples=10,
        gravity_mode_override="st_lrps",
        st_lrps_model_dir="some_nonexistent_dir"
    )
    assert cfg.st_lrps_model_dir == "some_nonexistent_dir"

    # Should raise if st_lrps_model_dir is empty when backend is st_lrps
    with pytest.raises(ValueError, match="st_lrps_model_dir cannot be empty"):
        MonteCarloConfig(n_samples=10, gravity_mode_override="st_lrps", st_lrps_model_dir="")

    # High-degree GPU SH requests are accepted at config time so backend policy
    # can record an explicit CPU fallback instead of silently clipping degree.
    cfg = MonteCarloConfig(n_samples=10, gpu_sh_degree=80)
    assert cfg.gpu_sh_degree == 80


def test_monte_carlo_impact_and_storage_contracts() -> None:
    cfg = MonteCarloConfig(
        n_samples=10,
        detect_impact=False,
        compute_impact_statistics=True,
        result_storage_mode="auto",
        max_result_memory_gb=0.5,
    )
    assert cfg.impact_detection_enabled is False
    assert cfg.impact_statistics_enabled is True
    assert cfg.estimated_result_bytes(20) == 20 * 10 * 6 * 8

    with pytest.raises(ValueError, match="requires output_format='hdf5'"):
        MonteCarloConfig(
            n_samples=10,
            output_format="npz",
            result_storage_mode="disk",
        )


def test_legacy_compute_impact_probability_maps_to_both_flags() -> None:
    with pytest.warns(DeprecationWarning):
        cfg = MonteCarloConfig(n_samples=10, compute_impact_probability=False)
    assert cfg.impact_detection_enabled is False
    assert cfg.impact_statistics_enabled is False


def test_validate_st_lrps_model_dir(tmp_path: Path):
    model_dir = tmp_path / "mock_model"

    # 1. Directory does not exist
    with pytest.raises(ValueError, match="must point to an existing trained ST-LRPS run directory"):
        validate_st_lrps_model_dir(model_dir)

    model_dir.mkdir()

    # 2. config.json missing
    with pytest.raises(ValueError, match="must contain config.json"):
        validate_st_lrps_model_dir(model_dir)

    (model_dir / "config.json").write_text("{}")

    # 3. Checkpoint missing
    with pytest.raises(ValueError, match="must contain checkpoints/ckpt_best.pt or checkpoints/ckpt_last.pt"):
        validate_st_lrps_model_dir(model_dir)

    ckpt_dir = model_dir / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "ckpt_best.pt").write_text("dummy")

    # 4. Valid
    validated = validate_st_lrps_model_dir(model_dir)
    assert validated == model_dir.resolve()


def test_lazy_imports():
    import lunaris.common as common
    assert hasattr(common, "math_utils")
    assert hasattr(common, "time_utils")

    # Accessing them triggers the lazy import
    from lunaris.common import math_utils, time_utils
    assert math_utils is not None
    assert time_utils is not None

    # Check that Numba requirement is mentioned in their docstrings
    assert "Numba is a required dependency" in math_utils.__doc__
    assert "Numba is required" in time_utils.__doc__
