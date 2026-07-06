"""Import-safe console entry points for optional Lunaris subsystems."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from collections.abc import Callable
from typing import Any

_OPTIONAL_ROOTS = {"torch", "h5py", "PySide6", "PyQt6", "spiceypy"}


def _help_requested() -> bool:
    return any(arg in {"-h", "--help"} for arg in sys.argv[1:])


def _fallback_help(*, description: str, extra: str) -> None:
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description=description,
        epilog=f"Install the optional dependencies with: pip install 'lunaris[{extra}]'",
    )
    parser.parse_args()


def _load_callable(module_name: str, callable_name: str) -> Callable[..., Any]:
    module = importlib.import_module(module_name)
    target = getattr(module, callable_name)
    if not callable(target):
        raise TypeError(f"{module_name}:{callable_name} is not callable")
    return target


def _run_optional(
    *,
    module_name: str,
    callable_name: str = "main",
    description: str,
    extra: str,
) -> Any:
    try:
        target = _load_callable(module_name, callable_name)
    except ModuleNotFoundError as exc:
        missing_root = str(exc.name or "").split(".", 1)[0]
        if missing_root not in _OPTIONAL_ROOTS:
            raise
        if _help_requested():
            _fallback_help(description=description, extra=extra)
            return 0
        raise RuntimeError(
            f"{description} requires the optional dependency {missing_root!r}. "
            f"Install it with: pip install 'lunaris[{extra}]'"
        ) from exc
    return target()


def ui_main() -> Any:
    if _help_requested():
        _fallback_help(description="Launch the Lunaris mission desktop UI.", extra="ui")
        return 0
    return _run_optional(
        module_name="lunaris.ui.app",
        description="Lunaris mission desktop UI",
        extra="ui",
    )


def launcher_main() -> Any:
    if _help_requested():
        _fallback_help(description="Launch the Lunaris application chooser.", extra="ui")
        return 0
    return _run_optional(
        module_name="lunaris.ui.launcher",
        description="Lunaris application launcher",
        extra="ui",
    )


def studio_main() -> Any:
    if _help_requested():
        _fallback_help(description="Launch the ST-LRPS Studio desktop UI.", extra="ui")
        return 0
    return _run_optional(
        module_name="lunaris.surrogate.st_lrps.ui.studio",
        description="ST-LRPS Studio desktop UI",
        extra="ui",
    )


def train_main() -> Any:
    return _run_optional(
        module_name="lunaris.surrogate.st_lrps.training.cli",
        description="Train an ST-LRPS potential surrogate",
        extra="ml",
    )


def eval_main() -> Any:
    return _run_optional(
        module_name="lunaris.surrogate.st_lrps.evaluation.cli",
        description="Evaluate an ST-LRPS potential surrogate",
        extra="ml",
    )


def validate_main() -> Any:
    if len(sys.argv) > 1 and sys.argv[1] in {"gravity-field", "gravity-trajectory", "-h", "--help"}:
        from lunaris.validation.gravity_reference.cli import main as gravity_reference_main

        return gravity_reference_main(sys.argv[1:])
    return _run_optional(
        module_name="lunaris.surrogate.st_lrps.evaluation.validation_suite",
        description="Run the ST-LRPS validation suite",
        extra="ml",
    )


def benchmark_main() -> Any:
    if _help_requested():
        try:
            spice_available = importlib.util.find_spec("spiceypy") is not None
        except ModuleNotFoundError:
            spice_available = False
        if not spice_available:
            _fallback_help(description="Benchmark Lunaris gravity models.", extra="all")
            return 0
    return _run_optional(
        module_name="lunaris.surrogate.st_lrps.evaluation.compare_gravity_models",
        description="Benchmark Lunaris gravity models",
        extra="all",
    )


__all__ = [
    "ui_main",
    "launcher_main",
    "studio_main",
    "train_main",
    "eval_main",
    "validate_main",
    "benchmark_main",
]
