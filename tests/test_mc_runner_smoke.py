# -*- coding: utf-8 -*-
"""
Smoke tests for the Monte Carlo runner entry point.

These tests exercise the real MC bootstrap path with a tiny CPU ensemble so we
catch contract drift between the UI command builder, ``mc_runner.py``, and the
runtime bootstrap helpers used by the rest of the project.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import lunaris.core.mc_runner as mc_runner


def test_mc_runner_cpu_smoke_completes_and_writes_output(tmp_path: Path) -> None:
    output_path = tmp_path / "mc_smoke.npz"

    exit_code = mc_runner.main(
        [
            "--start-date", "2027-03-02T23:32:37",
            "--days", "0.0002",
            "--output-dt-s", "5",
            "--hp-km", "100",
            "--ha-km", "100",
            "--inc-deg", "90",
            "--raan-deg", "0",
            "--argp-deg", "0",
            "--ta-deg", "0",
            "--degree", "12",
            "--enable-sh", "on",
            "--enable-3rd-body-sun", "off",
            "--enable-3rd-body-earth", "off",
            "--enable-earth-j2", "off",
            "--enable-srp", "off",
            "--enable-albedo", "off",
            "--enable-thermal", "off",
            "--enable-tides", "off",
            "--enable-relativity-1pn", "off",
            "--mass-kg", "1000",
            "--area-m2", "5",
            "--cd", "2.2",
            "--cr", "1.5",
            "--method", "DOP853",
            "--user-max-step-s", "15",
            "--rtol", "1e-10",
            "--atol", "1e-12",
            "--n-samples", "2",
            "--seed", "11",
            "--use-gpu", "off",
            "--mc-dt-s", "10",
            "--mc-output-format", "npz",
            "--mc-output-path", str(output_path),
            "--impact-alt-km", "0",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()

    data = np.load(str(output_path), allow_pickle=False)
    assert data["Y"].ndim == 3
    assert data["Y"].shape[1] == 2
    assert data["impact_flags"].shape == (2,)


def test_mc_runner_emits_structured_progress_payloads(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "mc_progress.npz"

    exit_code = mc_runner.main(
        [
            "--start-date", "2027-03-02T23:32:37",
            "--days", "0.0002",
            "--output-dt-s", "5",
            "--hp-km", "100",
            "--ha-km", "100",
            "--inc-deg", "90",
            "--raan-deg", "0",
            "--argp-deg", "0",
            "--ta-deg", "0",
            "--degree", "12",
            "--enable-sh", "on",
            "--enable-3rd-body-sun", "off",
            "--enable-3rd-body-earth", "off",
            "--enable-earth-j2", "off",
            "--enable-srp", "off",
            "--enable-albedo", "off",
            "--enable-thermal", "off",
            "--enable-tides", "off",
            "--enable-relativity-1pn", "off",
            "--mass-kg", "1000",
            "--area-m2", "5",
            "--cd", "2.2",
            "--cr", "1.5",
            "--method", "DOP853",
            "--user-max-step-s", "15",
            "--rtol", "1e-10",
            "--atol", "1e-12",
            "--n-samples", "2",
            "--seed", "11",
            "--use-gpu", "off",
            "--mc-dt-s", "10",
            "--mc-output-format", "npz",
            "--mc-output-path", str(output_path),
            "--impact-alt-km", "0",
        ]
    )

    captured = capsys.readouterr().out.splitlines()
    progress_lines = [
        line for line in captured
        if line.startswith("[MC_PROGRESS]")
    ]

    assert exit_code == 0
    assert progress_lines

    payloads = [
        json.loads(line[len("[MC_PROGRESS]"):].strip())
        for line in progress_lines
    ]

    assert payloads[0]["stage"] == "sampling"
    assert payloads[-1]["stage"] == "finalizing"
    assert any(p.get("stage") == "propagating" for p in payloads)
    assert any(p.get("stage") == "writing" for p in payloads)
    assert payloads[-1]["percent"] >= 99.0
    assert payloads[-1]["done_samples"] == payloads[-1]["total_samples"]
