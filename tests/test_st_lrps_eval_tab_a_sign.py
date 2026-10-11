"""ST-LRPS Studio evaluation tab: the acceleration sign is only ever +1 or -1.

Regression tests: ``a_sign`` was a ``QDoubleSpinBox`` with range -10..10, so
the tab could build ``--a-sign 2.5``. ``a_sign`` is a sign convention (U vs
-U), not a scale, and the evaluation CLI now refuses anything but +1 / -1
(``_a_sign_arg``). The control is a two-option combo; a value saved by the
old spin box restores as +1 unless it was exactly -1.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from lunaris.surrogate.st_lrps.ui.studio_parts import evaluation_pages  # noqa: E402
from lunaris.surrogate.st_lrps.ui.studio_parts.common_widgets import _settings  # noqa: E402


@pytest.fixture()
def make_tab():
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tabs = []

    def _make():
        tab = evaluation_pages.STLRPSEvalTab()
        tabs.append(tab)
        return tab

    yield _make
    for tab in tabs:
        tab.deleteLater()


def _saved_sign(value) -> None:
    s = _settings()
    s.beginGroup("eval")
    s.setValue("a_sign", value)
    s.endGroup()
    s.sync()


def test_control_offers_exactly_plus_and_minus_one(make_tab) -> None:
    tab = make_tab()
    combo = tab.a_sign
    assert isinstance(combo, QtWidgets.QComboBox)
    assert sorted(float(combo.itemData(i)) for i in range(combo.count())) == [-1.0, 1.0]


@pytest.mark.parametrize(("saved", "expected"), [(2.5, 1.0), (-1.0, -1.0), (-3.0, 1.0), ("junk", 1.0)])
def test_old_saved_values_restore_to_a_unit_sign(make_tab, saved, expected) -> None:
    _saved_sign(saved)
    tab = make_tab()
    assert float(tab.a_sign.currentData()) == expected


@pytest.mark.parametrize("index", [0, 1])
def test_command_passes_a_sign_the_cli_accepts(make_tab, monkeypatch, index) -> None:
    tab = make_tab()
    tab.a_sign.setCurrentIndex(index)
    captured: list[list[str]] = []
    monkeypatch.setattr(tab.runner, "start", lambda program, args, workdir=None: captured.append(list(args)))
    tab._start()
    assert captured, "the tab did not build a command"
    args = captured[0]
    value = args[args.index("--a-sign") + 1]
    assert float(value) in (1.0, -1.0)
    cli = pytest.importorskip("lunaris.surrogate.st_lrps.evaluation.cli")
    if hasattr(cli, "_a_sign_arg"):  # the core's own validator (WP13)
        assert cli._a_sign_arg(value) == float(value)
