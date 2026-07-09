"""Package metadata regression tests."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _expected_version() -> str:
    try:
        return version("lunaris")
    except PackageNotFoundError:
        return "0+unknown"


def test_public_package_versions_share_metadata_source() -> None:
    import lunaris
    import lunaris.core as core
    import lunaris.surrogate.st_lrps as st_lrps

    expected = _expected_version()

    assert lunaris.__version__ == expected
    assert core.__version__ == expected
    assert st_lrps.__version__ == expected


def test_lunaris_api_facade_exports_stable_names() -> None:
    import lunaris.api as api

    assert set(api.__all__) == {
        "BatchPropagationConfig",
        "BatchPropagationEngine",
        "BatchPropagationResult",
        "DynamicsEngine",
        "load_default_config",
        "propagate",
        "replace_sim_config",
    }
    assert api.load_default_config.__name__ == "load_default_config"
    assert api.replace_sim_config.__name__ == "replace_sim_config"
    assert api.BatchPropagationConfig is api.BatchPropagationConfig
    assert api.BatchPropagationEngine is api.BatchPropagationEngine
    assert api.BatchPropagationResult.__name__ == "BatchPropagationResult"
    assert api.BatchPropagationConfig.__name__ == "BatchPropagationConfig"
    assert api.BatchPropagationEngine.__name__ == "BatchPropagationEngine"


def test_distribution_status_classifier_matches_beta_readme() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert "Development Status :: 4 - Beta" in text
    assert "Development Status :: 3 - Alpha" not in text
