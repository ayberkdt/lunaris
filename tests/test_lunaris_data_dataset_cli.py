from __future__ import annotations

import json
import subprocess
import sys

from dataset_pipeline_test_utils import write_toy_contract_h5


def _cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "lunaris.cli.data", *args]


def test_lunaris_data_inspect_validate_and_report(tmp_path):
    data_path = write_toy_contract_h5(tmp_path / "toy.h5", n=18)

    inspect = subprocess.run(
        _cmd("inspect", "--data", str(data_path)),
        capture_output=True,
        text=True,
        check=True,
    )
    inspect_payload = json.loads(inspect.stdout)
    assert inspect_payload["target_mode"] == "residual"
    assert inspect_payload["n_samples"] == 18

    validate_out = tmp_path / "validate"
    validate = subprocess.run(
        _cmd("validate", "--data", str(data_path), "--out", str(validate_out), "--n-check", "18"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(validate.stdout)["passed"] is True
    assert (validate_out / "dataset_validation_report.json").exists()

    report_out = tmp_path / "quality"
    report = subprocess.run(
        _cmd("report", "--data", str(data_path), "--out", str(report_out), "--bins", "4"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(report.stdout)["n_samples"] == 18
    assert (report_out / "dataset_quality_report.json").exists()
    assert (report_out / "dataset_quality_summary.md").exists()
