"""ST-LRPS evaluation must accept only a_sign = +1 / -1.

``evaluate_model`` scales the base potential and the residual acceleration by
``a_sign`` (``Δa = a_sign · ∇ΔU``). The ``--a-sign`` flag was ``type=float`` and
``resolved_a_sign`` from config.json was taken as-is, so 0.5 or 0 silently
rescaled (or zeroed) the evaluated field instead of being rejected.
"""

from __future__ import annotations

import argparse

import pytest

pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.evaluation import cli as eval_cli  # noqa: E402


@pytest.mark.parametrize("text", ["0.5", "0", "2", "-1.5"])
def test_cli_flag_rejects_non_unit_sign(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="must be"):
        eval_cli._a_sign_arg(text)


@pytest.mark.parametrize(("text", "want"), [("1", 1.0), ("+1", 1.0), ("-1", -1.0), ("-1.0", -1.0)])
def test_cli_flag_accepts_unit_sign(text: str, want: float) -> None:
    assert eval_cli._a_sign_arg(text) == want


def test_config_value_is_validated() -> None:
    with pytest.raises(ValueError, match="must be"):
        eval_cli._require_unit_a_sign(0.5, "a_sign")
    assert eval_cli._require_unit_a_sign(-1, "a_sign") == -1.0
