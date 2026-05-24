import warnings
from pathlib import Path

import pytest

from loaders.io_helpers import (
    project_root_from_path,
    autodetect_repository_data_roots,
    DataRootHints,
    CANONICAL_PROJECT_MARKERS,
    LEGACY_PROJECT_MARKERS,
)
from loaders.io_surface import InMemorySurfaceProvider, FileBackedSurfaceProvider
from loaders.spice_builder import resolve_kernel_paths


def test_project_root_discovery(tmp_path):
    # Setup mock structure
    (tmp_path / "ST_LRPS" / "data").mkdir(parents=True)
    deep_path = tmp_path / "ST_LRPS" / "data" / "assets" / "deep"
    deep_path.mkdir(parents=True)
    
    # Test strict=False (fallback)
    root = project_root_from_path(deep_path)
    assert root.name == "ST_LRPS"
    
    # Test strict=True
    root_strict = project_root_from_path(deep_path, strict=True)
    assert root_strict.name == "ST_LRPS"
    
    # Test strict=True missing
    outside_path = tmp_path / "outside" / "dir"
    outside_path.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        project_root_from_path(outside_path, strict=True)
        
    # Test strict=False missing
    root_fallback = project_root_from_path(outside_path, strict=False)
    assert root_fallback.is_dir()


def test_deprecation_warnings():
    provider = InMemorySurfaceProvider()
    with pytest.warns(DeprecationWarning, match="get_provider.. is deprecated"):
        provider.get_provider()
        
    # We can't easily test FileBackedSurfaceProvider without real files,
    # but the logic is identical.


def test_spice_absolute_paths(tmp_path, monkeypatch):
    kernel_file = tmp_path / "test.bsp"
    kernel_file.write_text("dummy")
    
    # Change current working directory to something else so relative path resolution can be tested
    monkeypatch.chdir(tmp_path)
    
    kernels = ["test.bsp"]
    resolved = resolve_kernel_paths(kernels)
    
    assert len(resolved) == 1
    assert Path(resolved[0]).is_absolute()
    assert resolved[0] == str(kernel_file.resolve())
