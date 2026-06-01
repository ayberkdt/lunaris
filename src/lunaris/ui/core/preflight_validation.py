# ST_LRPS/ui_parts/preflight_validation.py
# -*- coding: utf-8 -*-
"""
Asynchronous pre-flight validation helpers for the desktop UI.

The validation worker performs lightweight checks before the backend process is
spawned. It intentionally focuses on fast feedback:
- invalid orbit values
- missing required files/directories
- obviously impossible numeric ranges
- output directory writeability

This module keeps the thread implementation out of the main window so the
window can focus on orchestration rather than background validation mechanics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Tuple

from PySide6 import QtCore

try:
    from lunaris.ui.core.surrogate_artifacts import validate_surrogate_run_preflight
except ImportError:
    if __name__ == "__main__":
        raise SystemExit(2)
    raise


class PreFlightWorker(QtCore.QThread):
    """
    Run pre-launch validation in a background Qt thread.

    The worker emits progress/warning/error signals so the host window can keep
    the user informed without blocking the GUI event loop.
    """

    validation_complete = QtCore.Signal(bool, str)
    validation_progress = QtCore.Signal(str)
    validation_warning = QtCore.Signal(str)
    validation_error = QtCore.Signal(str)

    def __init__(self, command_data: Mapping[str, Any], main_script_path: Path) -> None:
        super().__init__()
        self.command_data = dict(command_data)
        self.main_script_path = Path(main_script_path)
        self._stop_requested = False

    def run(self) -> None:
        """
        Execute the validation sequence.

        Each step reports progress before running. When a step fails, the worker
        stops immediately and reports the reason back to the host.
        """

        try:
            if self._is_cancelled():
                return

            self.validation_progress.emit("Validating orbit parameters...")
            if not self._validate_orbit():
                self.validation_error.emit("Invalid orbit parameters")
                self.validation_complete.emit(False, "Orbit validation failed")
                return

            if self._is_cancelled():
                return

            self.validation_progress.emit("Checking simulation engine...")
            if not self.main_script_path.exists():
                self.validation_error.emit(f"Main script not found: {self.main_script_path}")
                self.validation_complete.emit(False, "Simulation engine not found")
                return

            if self.command_data.get("gravity_enabled", True):
                if self._is_cancelled():
                    return
                self.validation_progress.emit("Validating gravity model files...")
                success, message = self._validate_gravity_files()
                if not success:
                    self.validation_error.emit(message)
                    self.validation_complete.emit(False, "Gravity file validation failed")
                    return
                if message:
                    self.validation_warning.emit(message)

            if self.command_data.get("albedo_enabled", False):
                if self._is_cancelled():
                    return
                self.validation_progress.emit("Validating albedo model files...")
                success, message = self._validate_albedo_files()
                if not success:
                    self.validation_error.emit(message)
                    self.validation_complete.emit(False, "Albedo file validation failed")
                    return
                if message:
                    self.validation_warning.emit(message)

            if self._is_cancelled():
                return

            self.validation_progress.emit("Checking output directory...")
            success, message = self._validate_output_directory()
            if not success:
                self.validation_error.emit(message)
                self.validation_complete.emit(False, "Output directory validation failed")
                return

            if self._is_cancelled():
                return

            self.validation_progress.emit("Validating numeric ranges...")
            success, message = self._validate_numeric_ranges()
            if not success:
                self.validation_error.emit(message)
                self.validation_complete.emit(False, "Numeric validation failed")
                return

            self.validation_progress.emit("All validation checks passed.")
            self.validation_complete.emit(True, "Pre-flight validation successful")

        except Exception as exc:
            error_text = f"Validation error: {exc}"
            self.validation_error.emit(error_text)
            self.validation_complete.emit(False, error_text)

    def stop(self) -> None:
        """
        Request cancellation.

        The current validation step is allowed to finish, after which the worker
        exits and reports a cancelled state to the host.
        """

        self._stop_requested = True

    def _is_cancelled(self) -> bool:
        """
        Short-circuit helper used between validation phases.

        Returning early keeps the host responsive when the application is
        closing or when the user changes their mind during validation.
        """

        if not self._stop_requested:
            return False
        self.validation_complete.emit(False, "Pre-flight validation cancelled")
        return True

    def _validate_orbit(self) -> bool:
        """
        Check whether the user-provided orbit parameters are physically sane.

        This is intentionally conservative: it catches obvious data entry
        mistakes without trying to fully reimplement backend orbit validation.
        """

        try:
            orbit_mode = self.command_data.get("orbit_mode", "hp_ha")
            if orbit_mode == "circular":
                alt_km = float(self.command_data.get("alt_km", 0.0))
                if alt_km < 0.0:
                    return False
            elif orbit_mode == "hp_ha":
                hp_km = float(self.command_data.get("hp_km", 0.0))
                ha_km = float(self.command_data.get("ha_km", hp_km))
                if hp_km < 0.0 or ha_km < 0.0:
                    return False
                if ha_km < hp_km:
                    self.validation_warning.emit(
                        f"Periselene ({hp_km} km) is above aposelene ({ha_km} km). Values will be swapped by the backend."
                    )
            else:
                semi_major_axis_km = float(self.command_data.get("a_km", 0.0))
                eccentricity = float(self.command_data.get("e", 0.0))
                if semi_major_axis_km <= 0.0:
                    return False
                if eccentricity < 0.0 or eccentricity >= 1.0:
                    return False

            for angular_key in ("inc_deg", "raan_deg", "argp_deg", "ta_deg"):
                angle = float(self.command_data.get(angular_key, 0.0))
                if not (0.0 <= angle <= 360.0):
                    return False
            return True
        except (TypeError, ValueError):
            return False

    def _validate_gravity_files(self) -> Tuple[bool, str]:
        """
        Verify that the chosen gravity model file exists and looks usable.

        Large files are not an error, but they are surfaced as warnings so the
        user understands why startup may take longer than expected.
        """

        backend = str(self.command_data.get("gravity_backend", "classic_sh") or "classic_sh").strip().lower()
        if backend == "st_lrps":
            model_dir = str(self.command_data.get("st_lrps_model_dir", "") or "").strip()
            ok, summary, warnings = validate_surrogate_run_preflight(model_dir)
            for w in warnings:
                self.validation_warning.emit(w)
            if not ok:
                return False, summary

            # Additionally verify that degree metadata is present so the
            # MC propagator does not fail inside the sample loop.
            if model_dir:
                cfg_path = Path(model_dir) / "config.json"
                if cfg_path.is_file():
                    try:
                        import json as _json
                        _cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
                        _dm = _cfg.get("dataset_meta") or {}
                        _deg_max = (
                            _cfg.get("degree_max")
                            or _dm.get("degree_max")
                            or _dm.get("requested_degree")
                        )
                        if _deg_max is None:
                            return (
                                False,
                                "ST-LRPS config.json is missing 'degree_max'. "
                                "Re-generate the dataset with spatial_cloud_generator.py >= v2.0 "
                                "and retrain, or manually add degree_max to config.json.",
                            )
                        _deg_min = _cfg.get("degree_min") or _dm.get("degree_min") or 0
                        summary = f"{summary} | degree {_deg_min}→{_deg_max}"
                    except Exception:
                        self.validation_warning.emit(
                            "Could not read degree metadata from ST-LRPS config.json."
                        )

            return True, summary

        file_path = str(self.command_data.get("gravity_file", "") or "").strip()
        if not file_path:
            return True, "Using default gravity model"

        path = Path(file_path)
        if not path.exists():
            return False, f"Gravity file not found: {path}"
        if not path.is_file():
            return False, f"Gravity path is not a file: {path}"

        try:
            file_size_mb = path.stat().st_size / (1024 * 1024)
        except OSError:
            return True, f"Gravity file validated: {path.name}"

        if file_size_mb > 500.0:
            return True, f"Large gravity file detected ({file_size_mb:.1f} MB). Loading may be slow."
        if file_size_mb > 100.0:
            return True, f"Gravity file size: {file_size_mb:.1f} MB"
        return True, f"Gravity file validated: {path.name}"

    def _validate_albedo_files(self) -> Tuple[bool, str]:
        """
        Check albedo label/image files when the albedo force model is enabled.

        The backend can fall back to defaults when explicit file paths are not
        set, so "missing path configuration" is not treated as an error here.
        """

        label_path = str(self.command_data.get("albedo_label", "") or "").strip()
        image_path = str(self.command_data.get("albedo_img", "") or "").strip()

        if not label_path:
            return True, "Using default albedo model"

        label_file = Path(label_path)
        if not label_file.exists():
            return False, f"Albedo label file not found: {label_path}"

        if image_path:
            image_file = Path(image_path)
            if not image_file.exists():
                return False, f"Albedo image file not found: {image_path}"

        return True, f"Albedo files validated: {label_file.name}"

    def _validate_output_directory(self) -> Tuple[bool, str]:
        """
        Ensure the selected output directory can be created and written to.

        This catches permission problems before the backend starts, which makes
        failures much easier to understand from the user's perspective.
        """

        output_dir = str(self.command_data.get("output_dir", "") or "").strip()
        if not output_dir:
            return False, "Output directory not specified"

        path = Path(output_dir)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return False, f"Cannot create output directory: {exc}"

        test_file = path / ".write_test"
        try:
            test_file.touch()
            test_file.unlink(missing_ok=True)
        except OSError as exc:
            return False, f"No write permission in output directory: {exc}"

        return True, f"Output directory ready: {path}"

    def _validate_numeric_ranges(self) -> Tuple[bool, str]:
        """
        Validate the non-orbit numeric parameters needed for a stable run.

        The checks are intentionally simple and focus on values that are clearly
        invalid regardless of the downstream dynamics model.
        """

        try:
            mass_kg = float(self.command_data.get("mass_kg", 1000.0))
            area_m2 = float(self.command_data.get("area_m2", 5.0))
            cd = float(self.command_data.get("cd", 2.2))
            cr = float(self.command_data.get("cr", 1.5))
            rtol = float(self.command_data.get("rtol", 1e-12))
            atol = float(self.command_data.get("atol", 1e-14))
            duration_value = float(self.command_data.get("duration_val", 10.0))

            if mass_kg <= 0.0:
                return False, "Spacecraft mass must be positive"
            if area_m2 <= 0.0:
                return False, "Cross-sectional area must be positive"
            if cd <= 0.0 or cr < 0.0:
                return False, "Drag and reflectivity coefficients must be positive"
            if rtol <= 0.0 or atol <= 0.0:
                return False, "Integration tolerances must be positive"
            if duration_value <= 0.0:
                return False, "Propagation duration must be positive"

            max_step = self.command_data.get("max_step", None)
            if max_step is not None and float(max_step) <= 0.0:
                return False, "Maximum solver step must be positive"

            return True, "Numeric ranges validated"
        except (TypeError, ValueError) as exc:
            return False, f"Invalid numeric value: {exc}"
