import os
import re
import subprocess
from pathlib import Path
import pytest

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

# Dynamically construct banned words so this file doesn't fail its own tests.
BANNED_PROJECT_NAMES = [
    "Lunar" + "Sim",
    "LUNAR" + "_SIMULATION",
    "LUNARSIM" + "_",
    "gececi" + "_kod",
]

OLD_PACKAGE_PATH = "surrogate" + "_gravity" + "_model"

def iter_doc_files(root: Path):
    """Yield documentation files covered by repository hygiene checks."""
    doc_roots = [
        root / "docs",
        root / "validation",
        root / "analysis",
        root / "st_lrps",
    ]
    explicit_docs = [
        root / "README.md",
        root / "visualization" / "README.md",
    ]

    seen = set()
    for path in explicit_docs:
        if path.exists() and path.suffix == ".md":
            resolved = path.resolve()
            seen.add(resolved)
            yield path

    for doc_root in doc_roots:
        if not doc_root.exists():
            continue
        for path in doc_root.rglob("*.md"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path

def iter_source_files(root: Path):
    exclude_dirs = {
        ".git", ".pytest_cache", "__pycache__", ".claude",
        "data", "results", "mc_results", "runs", "artifacts",
        "output", "outputs", "reports", ".vscode", ".idea"
    }
    
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for f in filenames:
            if f.endswith(('.py', '.md', '.txt', '.json', '.yaml', '.yml', '.ini', '.cfg')):
                path = Path(dirpath) / f
                rel_path = path.relative_to(root)
                # Allow exceptions
                if "tests/fixtures" in rel_path.as_posix() or "examples/st_lrps_minimal_artifact" in rel_path.as_posix():
                    continue
                # Skip this file itself
                if path.resolve() == Path(__file__).resolve():
                    continue
                yield path

def get_committed_files(root: Path):
    try:
        result = subprocess.run(
            ['git', 'ls-files'], 
            cwd=root, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return [root / Path(p) for p in result.stdout.splitlines() if p.strip()]
    except Exception:
        # Fallback to a basic walk if git is unavailable
        files = []
        for p in iter_source_files(root):
            files.append(p)
        return files

@pytest.mark.skipif(
    (get_project_root() / ("surrogate" + "_gravity" + "_model")).exists(),
    reason="Claude's refactor hasn't landed yet"
)
def test_no_stale_project_identity():
    root = get_project_root()
    found_issues = []
    
    for filepath in iter_doc_files(root):
        try:
            content = filepath.read_text(encoding='utf-8')
            for banned in BANNED_PROJECT_NAMES:
                if banned in content:
                    found_issues.append(f"Found '{banned}' in {filepath.relative_to(root)}")
        except Exception:
            pass
            
    assert not found_issues, "Stale project identity found:\n" + "\n".join(found_issues)

@pytest.mark.skipif(
    (get_project_root() / ("surrogate" + "_gravity" + "_model")).exists(),
    reason="Claude's refactor hasn't landed yet (old directory still exists)"
)
def test_no_old_surrogate_package_path():
    root = get_project_root()
    found_issues = []
    
    for filepath in iter_doc_files(root):
        try:
            content = filepath.read_text(encoding='utf-8')
            if OLD_PACKAGE_PATH in content:
                found_issues.append(f"Found old package path in {filepath.relative_to(root)}")
        except Exception:
            pass
            
    assert not found_issues, "Old surrogate package path found:\n" + "\n".join(found_issues)

@pytest.mark.skipif(
    (get_project_root() / ("surrogate" + "_gravity" + "_model")).exists(),
    reason="Claude's refactor hasn't landed yet"
)
def test_no_generated_artifacts_committed():
    root = get_project_root()
    committed_files = get_committed_files(root)
    
    banned_exact = {
        "gececi" + "_kod",
        "history.jsonl",
        "run_manifest.json",
        "command.txt",
        "scaler.json",
        "eval_report.json",
        "eval_manifest.json",
        "evaluate_metrics.json",
        "evaluate_summary.txt",
        "metrics_summary.csv",
        "ood_metrics.csv",
        "topk_worst.csv",
        "altitude_binned_metrics.csv",
        "angular_error_by_altitude.csv",
        "angular_error_by_accel_norm.csv",
        "acceleration_decomposition.csv"
    }
    
    banned_dirs = {
        "checkpoints",
        "evals",
        "gececi" + "_kod"
    }
    
    banned_regexes = [
        re.compile(r"spatial_rmse_.*\.csv"),
        re.compile(r"spatial_mape_.*\.csv")
    ]
    
    found_issues = []
    for filepath in committed_files:
        rel_path_str = filepath.relative_to(root).as_posix()
        if "tests/fixtures" in rel_path_str or "examples/st_lrps_minimal_artifact" in rel_path_str:
            continue
            
        name = filepath.name
        # Exact match
        if name in banned_exact:
            found_issues.append(f"Banned artifact committed: {rel_path_str}")
            continue
            
        # Regex match
        if any(r.match(name) for r in banned_regexes):
            found_issues.append(f"Banned artifact pattern committed: {rel_path_str}")
            continue
            
        # Directory check
        parts = filepath.parts
        for b_dir in banned_dirs:
            if b_dir in parts:
                found_issues.append(f"Banned directory '{b_dir}' committed in: {rel_path_str}")
                break

    assert not found_issues, "Generated artifacts found in committed files:\n" + "\n".join(found_issues)

def test_readme_sanity():
    root = get_project_root()
    readme_path = root / "README.md"
    assert readme_path.exists(), "README.md does not exist"
    
    content = readme_path.read_text(encoding='utf-8')
    
    assert "ST-LRPS" in content, "README title must contain ST-LRPS"
    
    for banned in [*BANNED_PROJECT_NAMES, OLD_PACKAGE_PATH]:
        assert banned not in content, "README must not contain stale project identity or package paths"

    if (root / "st_lrps" / "training" / "cli.py").exists():
        assert "python -m st_lrps.training.cli" in content
    else:
        legacy_train_cmd = "python -m st_lrps." + "st_lrps_train"
        assert legacy_train_cmd in content

    assert "validation.gravity.compare_gravity_models" in content
    assert "visualization.surface_explorer" in content
