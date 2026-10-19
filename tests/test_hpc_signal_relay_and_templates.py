"""HPC launch path: the walltime signal must reach the trainer.

- Every ``hpc/*.sbatch`` sourced ``$(dirname "$0")/env_template.sh``; Slurm runs
  a spooled copy of the script, so ``$0`` is in the spool directory and every
  job died at its first line.
- ``--signal=B:TERM@<lead>`` reaches only the batch shell: the shell relays it
  to the trainer (or the scenario launcher) through ``lunaris_run_stoppable``
  and keeps the exit status (75 = stopped with ckpt_last saved), and the
  launcher relays the signal to its child.
- An array job with unscrambled ``sobol`` runs the same ensemble in every
  task (the design ignores the seed); the batch template refuses it.
"""

from __future__ import annotations

import importlib.util
import re
import signal
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_HPC = _REPO / "hpc"


def _launcher():
    spec = importlib.util.spec_from_file_location(
        "run_training_scenario_relay", _REPO / "tools" / "hpc" / "run_training_scenario.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_launcher_relays_sigterm_to_the_child(monkeypatch):
    launcher = _launcher()
    sent: list[int] = []

    class FakeProc:
        def send_signal(self, signum):
            sent.append(int(signum))

        def wait(self):
            # The walltime signal arrives while the launcher waits.
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)
            return 0

    monkeypatch.setattr(launcher.subprocess, "Popen", lambda cmd, cwd: FakeProc())
    before = signal.getsignal(signal.SIGTERM)
    assert launcher.run_forwarding_signals(["trainer"], cwd=".") == 0
    assert sent == [int(signal.SIGTERM)]
    assert signal.getsignal(signal.SIGTERM) == before


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
def test_launcher_relays_a_real_sigterm(tmp_path):
    import os
    import threading
    import time

    launcher = _launcher()
    marker = tmp_path / "got_term"
    child = (
        "import signal, sys, time, pathlib\n"
        f"p = pathlib.Path({str(marker)!r})\n"
        "signal.signal(signal.SIGTERM, lambda *a: (p.write_text('1'), sys.exit(7)))\n"
        "pathlib.Path(str(p) + '.ready').write_text('1')\n"
        "time.sleep(30)\n"
    )
    ready = Path(str(marker) + ".ready")

    def _signal_self():
        deadline = time.time() + 20
        while not ready.exists() and time.time() < deadline:
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_signal_self, daemon=True).start()
    rc = launcher.run_forwarding_signals([sys.executable, "-c", child], cwd=str(tmp_path))
    assert rc == 7
    assert marker.exists()


def _templates() -> list[Path]:
    return sorted(_HPC.glob("*.sbatch"))


@pytest.mark.parametrize("template", _templates(), ids=lambda p: p.name)
def test_templates_do_not_source_relative_to_the_spooled_script(template):
    text = template.read_text(encoding="utf-8")
    assert 'source "$(dirname "$0")/env_template.sh"' not in text
    assert '_hpc_dir="${LUNARIS_HPC_DIR:-${SLURM_SUBMIT_DIR:-.}/hpc}"' in text
    assert 'source "${_hpc_dir}/env_template.sh"' in text


def test_signal_templates_relay_to_their_program_and_keep_its_status():
    for name, program in (
        ("slurm_train_stlrps.sbatch", "lunaris-train"),
        ("slurm_train_scenario_array.sbatch", "python tools/hpc/run_training_scenario.py"),
    ):
        text = (_HPC / name).read_text(encoding="utf-8")
        assert "--signal=B:TERM" in text
        # Not exec: the shell must see the exit status to act on 75 (resumable stop).
        assert not re.search(rf"^exec {re.escape(program)}", text, re.MULTILINE), name
        assert re.search(
            rf"^lunaris_run_stoppable {re.escape(program)} .*\|\| rc=\$\?$", text, re.MULTILINE
        ), name
        assert re.search(r'^lunaris_after_run "\$\{rc\}"', text, re.MULTILINE), name


def test_submit_helper_exports_the_hpc_dir():
    text = (_HPC / "submit.sh").read_text(encoding="utf-8")
    assert '--export=ALL,LUNARIS_HPC_DIR=${_HPC_DIR}' in text


def test_batch_array_refuses_unscrambled_sobol():
    text = (_HPC / "slurm_batch_array.sbatch").read_text(encoding="utf-8")
    assert "use sobol_scrambled for array jobs" in text
    assert "SLURM_ARRAY_TASK_COUNT" in text
