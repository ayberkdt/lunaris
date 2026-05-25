import argparse
import pytest
from unittest.mock import patch
from st_lrps.run_ablation_matrix import parse_args, build_matrix

def test_ablation_run_eval_flags():
    """Verify that --run-eval-after-training and --eval-streaming flags are parsed."""
    args = parse_args(["--run-eval-after-training", "--eval-streaming", "--dry-run", "--train-data", "train.h5"])
    assert args.run_eval_after_training is True
    assert args.eval_streaming is True

def test_ablation_eval_commands_generation():
    """Verify that eval_commands are generated correctly when flags are set."""
    args = parse_args([
        "--run-eval-after-training",
        "--eval-streaming",
        "--dry-run",
        "--train-data", "train.h5",
        "--test-data", "test.h5",
        "--ood-data", "ood.h5",
        "--only", "baseline_single_siren"
    ])
    entries = build_matrix(args)
    assert len(entries) == 1
    entry = entries[0]
    assert "eval_commands" in entry
    
    # Expecting two eval commands: one for test-data, one for ood-data
    assert len(entry["eval_commands"]) == 2
    
    cmd_test = entry["eval_commands"][0]
    assert "--streaming" in cmd_test
    assert "--data" in cmd_test
    assert "test.h5" in cmd_test
    
    cmd_ood = entry["eval_commands"][1]
    assert "--streaming" in cmd_ood
    assert "--data" in cmd_ood
    assert "ood.h5" in cmd_ood

def test_ui_import_safety():
    """Verify ui_st_lrps can be imported without PySide6 immediately failing (using try-except logic)."""
    try:
        import st_lrps.ui_st_lrps
    except ImportError:
        pytest.fail("ui_st_lrps should handle missing PySide6 gracefully or import successfully.")
