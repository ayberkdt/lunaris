"""Both desktop apps keep test runs out of the user's real settings store.

Regression tests: ``QSettings(organization, application)`` always opens the
native store (the Windows registry), and ``QSettings.setPath`` cannot redirect
it. Every test that built a Mission Studio window wrote ``ui/geometry`` and
the density/motion keys into ``HKCU\\Software\\Lunaris\\MissionStudio``, and an
autouse fixture in ``test_st_lrps_orbit_benchmark_ui.py`` cleared
``QSettings("ST_LRPS_Project", "ST_LRPS_Dashboard")``, wiping the developer's
real ST-LRPS Studio preferences on every run. With ``LUNARIS_APP_DATA_DIR``
set, which ``tests/conftest.py`` does for the whole session, every store is an
.ini file under that folder.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402

from lunaris.ui_foundation.settings import APP_DATA_ENV, persistent_settings  # noqa: E402


def _factories():
    from lunaris.surrogate.st_lrps.ui.studio_parts.common_widgets import _settings
    from lunaris.ui.app import MainWindow
    from lunaris.ui.core.ui_commons import mission_studio_settings

    return {
        "st_lrps_studio": _settings,
        "mission_studio": mission_studio_settings,
        "mission_density": MainWindow._density_settings,
    }


@pytest.mark.parametrize("name", ["st_lrps_studio", "mission_studio", "mission_density"])
def test_app_settings_are_a_file_under_the_override(name, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(APP_DATA_ENV, str(tmp_path))
    settings = _factories()[name]()
    assert settings.format() == QSettings.Format.IniFormat
    assert settings.fileName().replace("\\", "/").startswith(tmp_path.as_posix())


def test_reduced_motion_reads_the_isolated_store(tmp_path, monkeypatch) -> None:
    from lunaris.ui.core.ui_commons import mission_studio_settings, prefers_reduced_motion

    monkeypatch.setenv(APP_DATA_ENV, str(tmp_path))
    settings = mission_studio_settings()
    settings.setValue("ui/reduce_motion", True)
    settings.sync()
    assert prefers_reduced_motion() is True


def test_native_store_without_the_override(monkeypatch) -> None:
    monkeypatch.delenv(APP_DATA_ENV, raising=False)
    settings = persistent_settings("Org", "App", "unused.ini")
    assert settings.format() == QSettings.Format.NativeFormat
    assert settings.organizationName() == "Org"
    assert settings.applicationName() == "App"


def test_the_suite_runs_with_the_override_set() -> None:
    # tests/conftest.py sets it for the session; without it every UI test would
    # read and write the real store and the real session/layout files.
    assert os.environ.get(APP_DATA_ENV, "").strip()
