import os
import pytest

def get_banned_strings():
    # Use dynamic strings to avoid this test file breaking its own checks
    return [
        "surrogate" + "_" + "gravity" + "_" + "model",
        "Lunar" + "Sim",
        "LUNAR" + "_" + "SIMULATION",
        "LUNARSIM" + "_",
        "gececi" + "_kod",
    ]

def get_required_fields():
    return [
        "scenario_id",
        "model",
        "runtime_s",
        "rms_pos_err_km",
        "final_pos_err_km",
        "radial_rms_km",
        "along_rms_km",
        "cross_rms_km",
        "status",
        "failure_reason"
    ]

def get_boundary_layers():
    return [
        "analysis/",
        "st_lrps/evaluation/",
        "validation/"
    ]

@pytest.fixture
def doc_paths():
    # Paths are relative to the tests/ directory
    base_dir = os.path.dirname(os.path.dirname(__file__))
    return [
        os.path.join(base_dir, "validation", "README.md"),
        os.path.join(base_dir, "validation", "gravity", "README.md"),
        os.path.join(base_dir, "validation", "gravity", "output_schema.md"),
    ]

def test_file_existence(doc_paths):
    """Assert validation documentation files exist."""
    for path in doc_paths:
        assert os.path.exists(path), f"File {path} does not exist."

def test_no_stale_paths(doc_paths):
    """Scan the docs to assert they don't contain stale paths or legacy project names."""
    banned_strings = get_banned_strings()
    for path in doc_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            for banned in banned_strings:
                assert banned not in content, f"Found stale string '{banned}' in {path}"

def test_gravity_readme_command_sanity(doc_paths):
    """Assert the gravity README points at the relocated harness module path.

    The harness was moved into the ST-LRPS package
    (``st_lrps/evaluation/compare_gravity_models.py``); the README must document
    the canonical ``python -m st_lrps.evaluation.compare_gravity_models`` command.
    """
    gravity_readme = doc_paths[1]
    if os.path.exists(gravity_readme):
        with open(gravity_readme, "r", encoding="utf-8") as f:
            content = f.read()
            assert "python -m st_lrps.evaluation.compare_gravity_models" in content, "Missing expected command in gravity README"

def test_output_schema_field_sanity(doc_paths):
    """Assert output_schema.md contains key fields."""
    output_schema = doc_paths[2]
    if os.path.exists(output_schema):
        with open(output_schema, "r", encoding="utf-8") as f:
            content = f.read()
            required_fields = get_required_fields()
            for field in required_fields:
                assert field in content, f"Missing required field '{field}' in output schema"

def test_boundary_statement_sanity(doc_paths):
    """Assert validation/README.md mentions all three layers."""
    top_readme = doc_paths[0]
    if os.path.exists(top_readme):
        with open(top_readme, "r", encoding="utf-8") as f:
            content = f.read()
            boundary_layers = get_boundary_layers()
            for layer in boundary_layers:
                assert layer in content, f"Missing boundary layer '{layer}' in top README"
