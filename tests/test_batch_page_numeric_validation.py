"""Batch page: a numeric field that does not hold a usable number blocks the run.

Regression tests: ``get_data`` parsed every field with ``_parse_float`` /
``_parse_int``, which return a hard-coded default on any parse error, and
``validate_page_inputs`` only range-checked some fields. Text such as ``1,5``
in the VRAM budget, an empty threads-per-block field, ``nan`` in a dispersion
or ``12.5`` samples therefore ran with a value the field did not show and
nothing said so. Session text restored with ``setText`` is never re-committed
by the widget, so the page itself has to refuse it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from lunaris.ui.pages.batch_propagation_page import (  # noqa: E402
    BatchPropagationPage,
    UIBatchPropagationConfig,
)


@pytest.fixture()
def page():
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    p = BatchPropagationPage(batch_cfg=UIBatchPropagationConfig())
    yield p
    p.deleteLater()


@pytest.mark.parametrize(
    ("field", "text", "fragment"),
    [
        ("ent_vram", "1,5", "VRAM budget"),
        ("ent_tpb", "", "GPU threads per block"),
        ("ent_gpu_dev", "abc", "GPU device id"),
        ("ent_torch_chunk", "1e", "Torch SH chunk size"),
        ("ent_impact_alt", "inf", "Impact altitude"),
        ("ent_sh_degree", "ten", "SH degree"),
        ("ent_sigma_r", "nan", "σ_r"),
        ("ent_n_samples", "12.5", "Sample count"),
        ("ent_seed", "4.2", "Sampling seed"),
    ],
)
def test_unusable_numeric_text_is_a_validation_error(page, field, text, fragment) -> None:
    assert page.validate_page_inputs()[0] is True
    getattr(page, field).setText(text)
    ok, errors, _warnings = page.validate_page_inputs()
    assert ok is False
    matching = [e for e in errors if fragment in e]
    assert len(matching) == 1, errors  # one message per bad field, no duplicate range error
    assert not page.btn_run_batch.isEnabled()


def test_run_click_with_unusable_text_does_not_emit(page) -> None:
    fired: list[bool] = []
    page.run_requested.connect(lambda: fired.append(True))
    page.ent_vram.setText("1,5")
    page.btn_run_batch.setEnabled(True)  # e.g. a keyboard path that bypasses the button state
    page._on_run_clicked()
    assert fired == []
    assert "VRAM budget" in page.lbl_validation.text()


def test_valid_numbers_still_pass(page) -> None:
    page.ent_vram.setText("8.5")
    page.ent_n_samples.setText("1e3")
    page.ent_impact_alt.setText("2")
    ok, errors, _ = page.validate_page_inputs()
    assert ok, errors
    data = page.get_data()
    assert data["max_vram_gb"] == 8.5
    assert data["n_samples"] == 1000
