"""Pin compatibility aliases introduced by the API-boundary rename pass."""

from __future__ import annotations

import pytest


def test_relativity_kernel_rename_aliases_are_preserved() -> None:
    from lunaris.physics import relativity_effects as module

    aliases = {
        "_schwarzschild_components": "schwarzschild_components",
        "_external_schwarzschild_diff_components": (
            "external_schwarzschild_diff_components"
        ),
        "_de_sitter_components": "de_sitter_components",
        "_external_1pn_components": "external_1pn_components",
    }
    for old, new in aliases.items():
        assert getattr(module, old) is getattr(module, new)
        assert old in module.__all__
        assert new in module.__all__


def test_network_helper_rename_aliases_are_preserved() -> None:
    pytest.importorskip("torch")
    from lunaris.surrogate.st_lrps.networks import models

    aliases = {
        "_compute_harmonic_w0_bands": "compute_harmonic_w0_bands",
        "_get_output_head_params": "get_output_head_params",
    }
    for old, new in aliases.items():
        assert getattr(models, old) is getattr(models, new)
        assert old in models.__all__
        assert new in models.__all__
