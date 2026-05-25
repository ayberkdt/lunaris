# st_lrps/__init__.py
"""
ST-LRPS — Sobolev-Trained Lunar Residual Potential Surrogate.

Standalone package for ST-LRPS surrogate gravity data generation, training,
evaluation, and the runtime force model used by the propagator.

This package initializer is intentionally lightweight and import-safe: importing
``st_lrps`` must NOT pull in torch, h5py, the training engine, the models, the
evaluator, or the runtime force model, and must not read data or scan files.
Import the specific submodule you need (e.g. ``st_lrps.training.cli``) to pay
those costs explicitly.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
