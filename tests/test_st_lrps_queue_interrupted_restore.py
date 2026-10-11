"""ST-LRPS Studio training queue: a job running when the Studio died is not relaunched fresh.

Regression test: ``TrainingQueue._restore_persisted`` turned a persisted
``Running…`` job into ``Pending``; Start Queue then relaunched the same
arguments without ``--resume-from`` and the trainer started a new history in
the same output folder, deleting the interrupted run's history and
checkpoints. It is restored as ``Interrupted`` and is not picked by the queue.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")


class _FakeSettings:
    def __init__(self, store: dict) -> None:
        self._store = store

    def value(self, key, default=None):
        return self._store.get(key, default)

    def setValue(self, key, value) -> None:  # noqa: N802 - Qt API
        self._store[key] = value

    def sync(self) -> None:
        pass


def test_running_job_is_restored_as_interrupted(monkeypatch) -> None:
    from tests.ui_qt_helpers import QtWidgets

    from lunaris.surrogate.st_lrps.ui.studio_parts import training_pages

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    jobs = [
        {"label": "a", "args": ["--out", "runs/a"], "out_dir": "runs/a", "config": {}, "status": "Running…"},
        {"label": "b", "args": ["--out", "runs/b"], "out_dir": "runs/b", "config": {}, "status": "Pending"},
    ]
    store = {"training_queue/jobs": json.dumps(jobs, ensure_ascii=False)}
    monkeypatch.setattr(training_pages, "_settings", lambda: _FakeSettings(store))

    queue = training_pages.TrainingQueue()
    try:
        statuses = [job["status"] for job in queue._queue]
        assert statuses == ["Interrupted", "Pending"]
        # Only the genuinely pending job is eligible to run.
        assert [i for i, q in enumerate(queue._queue) if q["status"] == "Pending"] == [1]
        assert "Interrupted" in queue._list.item(0).text()
    finally:
        queue.deleteLater()
