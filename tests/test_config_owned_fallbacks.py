"""Settings come from their config, not from fallbacks that disagree with it.

The SciPy runner fell back to rtol 1e-9 (PropagatorConfig: 1e-10), the 2-body
baseline to 1e-9 / 1e-12 (PropagatorConfig: 1e-12 / 1e-14), the Nyquist step
helper defaulted to safety_div 5.0 (PropagatorConfig: 4.0), and plot_all to
save_pdf True / dpi 300 (VisualConfig: False / 150).
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from lunaris.analysis.reporting import manager
from lunaris.common.math_utils import nyquist_max_step_s
from lunaris.common.type_defs import PropagatorConfig
from lunaris.core.config import VisualConfig


def _field_default(cls, name: str):
    return next(f.default for f in dataclasses.fields(cls) if f.name == name)


def test_nyquist_helper_defaults_are_the_propagator_config_defaults() -> None:
    params = inspect.signature(nyquist_max_step_s).parameters
    assert params["safety_div"].default == _field_default(PropagatorConfig, "nyquist_safety_div")
    assert params["v_margin"].default == _field_default(PropagatorConfig, "nyquist_v_margin")


@pytest.mark.parametrize(
    ("visual_cfg", "expect_pdf", "expect_dpi"),
    [
        (None, VisualConfig().save_pdf, VisualConfig().default_dpi),
        (VisualConfig(save_pdf=True, save_pngs=True, default_dpi=200), True, 200),
    ],
)
def test_plot_all_takes_unset_choices_from_visual_config(
    tmp_path, monkeypatch, visual_cfg, expect_pdf, expect_dpi
) -> None:
    calls: dict[str, object] = {}

    def fake_pngs(run_dir, history, *, prefix, meta, dpi, ctx):
        calls["png_dpi"] = dpi
        return {}

    def fake_pdf(pdf_path, history, *, meta, ctx):
        calls["pdf"] = pdf_path
        return pdf_path

    monkeypatch.setattr(manager, "save_quicklook_pngs", fake_pngs)
    monkeypatch.setattr(manager, "make_report_pdf", fake_pdf)
    out = manager.plot_all({}, str(tmp_path), use_run_subdir=False, visual_cfg=visual_cfg)
    assert out["status"] == "success"
    assert calls["png_dpi"] == expect_dpi
    assert ("pdf" in calls) is expect_pdf


def test_explicit_plot_all_arguments_override_visual_config(tmp_path, monkeypatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        manager, "save_quicklook_pngs", lambda *a, dpi, **k: calls.setdefault("png_dpi", dpi) and {}
    )
    monkeypatch.setattr(manager, "make_report_pdf", lambda path, *a, **k: calls.setdefault("pdf", path))
    out = manager.plot_all(
        {}, str(tmp_path), use_run_subdir=False, save_pdf=False, save_png=True, dpi=72,
        visual_cfg=VisualConfig(save_pdf=True, default_dpi=200),
    )
    assert out["status"] == "success"
    assert calls == {"png_dpi": 72}
