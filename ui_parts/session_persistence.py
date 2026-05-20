# LUNAR_SIMULATION/ui_parts/session_persistence.py
# -*- coding: utf-8 -*-
"""
Session capture, restore, and data-path auto-detection helpers.

The main window has to coordinate several independently owned pages. Rather
than letting `ui.py` manually reach into every widget for save/restore
operations, this module defines a small persistence layer that works against
page-level APIs (`get_data`, `to_dict`, `get_state`, `apply_state`, etc.).

Design goals
------------
1. Keep serialization rules in one place.
2. Preserve backward compatibility with previously saved mission profile files.
3. Keep path auto-detection testable and repository-aware.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from loaders.io_helpers import (
    DataRootHints,
    autodetect_repository_data_roots,
    prefer_dedicated_albedo_root,
)

from .data_files_page import DataFilesState
from .result_exports_page import OutputPageState
from .solver_policy import (
    DEFAULT_SOLVER_METHOD,
    coerce_positive_float,
    normalize_solver_config_object,
)


def _prefer_dedicated_albedo_root(project_root: Path, state: DataFilesState) -> DataFilesState:
    """
    UI-facing adapter around the loader-layer albedo-root migration policy.

    The actual repository-aware path logic now lives in
    `loaders.io_helpers.prefer_dedicated_albedo_root(...)`. This wrapper keeps
    the existing UI API stable while ensuring the path policy is owned by the
    loader layer instead of the widget/persistence layer.
    """

    migrated = prefer_dedicated_albedo_root(
        project_root,
        DataRootHints(
            ldem_root=state.ldem_root,
            albedo_root=state.albedo_root,
            kernel_dir="",
            use_ldem_for_albedo=bool(state.use_ldem_for_albedo),
        ),
    )
    state.ldem_root = migrated.ldem_root
    state.albedo_root = migrated.albedo_root
    state.use_ldem_for_albedo = bool(migrated.use_ldem_for_albedo)
    return state


def _safe_call(default: Any, fn: Callable[[], Any]) -> Any:
    """
    Execute a small page/config accessor without letting restore/save fail hard.

    Persistence should be resilient. If one page is not fully initialized yet,
    the caller still deserves the best-effort snapshot instead of a fatal
    exception.
    """

    try:
        return fn()
    except Exception:
        return default


def collect_session_snapshot(
    *,
    orbit_page: Any,
    propagation_page: Any,
    force_page: Any,
    output_page: Any,
    data_page: Any,
    gravity_cfg: Any,
    albedo_cfg: Any,
    solver_cfg: Any,
    spacecraft_cfg: Any,
    app_version: str,
    mc_page: Optional[Any] = None,
    surrogate_page: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Collect a full UI session payload suitable for JSON persistence.

    The output schema intentionally mirrors earlier project saves where
    possible. That keeps older user profiles loadable while allowing the newer,
    more modular UI pages to own their state internally.
    """

    orbit_payload = _safe_call(
        {
            "mode": "hp_ha",
            "hp_km": "",
            "ha_km": "",
            "a_km": "",
            "e": "",
            "alt_km": "",
            "inc_deg": "",
            "raan_deg": "",
            "argp_deg": "",
            "ta_deg": "",
        },
        lambda: orbit_page.get_data() or {},
    )

    propagation_payload = _safe_call({"timeline": {}, "integrator": {}}, lambda: propagation_page.to_dict() or {})
    forces_payload = _safe_call({}, lambda: force_page.get_data() or {})
    output_state = _safe_call(
        OutputPageState(output_dir="", generate_3d_plots=False, downsample_3d=1),
        lambda: output_page.get_state(),
    )
    data_state = _safe_call(DataFilesState(), lambda: data_page.get_state())

    return {
        "meta": {
            "version": app_version,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "orbit": {
            "mode": orbit_payload.get("mode", "hp_ha"),
            "hp_km": str(orbit_payload.get("hp_km", "")),
            "ha_km": str(orbit_payload.get("ha_km", "")),
            "a_km": str(orbit_payload.get("a_km", "")),
            "e": str(orbit_payload.get("e", "")),
            "alt_km": str(orbit_payload.get("alt_km", "")),
            "inc_deg": str(orbit_payload.get("inc_deg", "")),
            "raan_deg": str(orbit_payload.get("raan_deg", "")),
            "argp_deg": str(orbit_payload.get("argp_deg", "")),
            "ta_deg": str(orbit_payload.get("ta_deg", "")),
        },
        "timeline": propagation_payload.get("timeline", {}) or {},
        "integrator": propagation_payload.get("integrator", {}) or {},
        "forces": forces_payload,
        "solver_config": dataclasses.asdict(solver_cfg),
        "spacecraft_config": dataclasses.asdict(spacecraft_cfg),
        "output": {
            "dir": output_state.output_dir,
            "anim3d": bool(output_state.generate_3d_plots),
            "downsample_3d": int(output_state.downsample_3d),
            # Preserved for compatibility with older session files, even though
            # the dedicated CSV toggle was removed from the UI.
            "csv": True,
        },
        "albedo_config": dataclasses.asdict(albedo_cfg),
        "data_config": {
            "ldem_root": data_state.ldem_root,
            "albedo_root": data_state.albedo_root,
            "kernel_dir": data_state.kernel_dir,
            "ldem_ppd": int(data_state.ldem_ppd),
            "use_ldem_for_albedo": bool(data_state.use_ldem_for_albedo),
        },
        # This top-level copy keeps advanced gravity settings recoverable even if
        # older consumers ignore the nested forces payload.
        "gravity_config": gravity_cfg.to_dict(),
        # Monte Carlo configuration (absent when mc_page is not wired in)
        "monte_carlo": _safe_call({}, lambda: mc_page.get_data()) if mc_page is not None else {},
        # Surrogate Studio page state (runs root + selected run).  Absent in
        # older session files; restore code below tolerates that.
        "surrogate_studio": (
            _safe_call({"runs_root": "", "selected_run": ""}, lambda: surrogate_page.get_state())
            if surrogate_page is not None
            else {"runs_root": "", "selected_run": ""}
        ),
        # Visual workspace state — absent in old sessions; restore tolerates absence.
        "visual_state": {},   # populated via collect_visual_state() if caller fills it
    }


def apply_session_snapshot(
    payload: dict[str, Any],
    *,
    orbit_page: Any,
    propagation_page: Any,
    force_page: Any,
    output_page: Any,
    data_page: Any,
    gravity_cfg: Any,
    albedo_cfg: Any,
    solver_cfg: Any,
    spacecraft_cfg: Any,
    project_root: Path,
    log_warning: Optional[Callable[[str], None]] = None,
    mc_page: Optional[Any] = None,
    surrogate_page: Optional[Any] = None,
) -> None:
    """
    Restore a previously saved payload back into the modular UI.

    Parameters
    ----------
    payload:
        Parsed JSON-like dictionary from disk.
    log_warning:
        Optional callback used when part of the restore fails. The caller keeps
        control over whether warnings land in a log panel, stdout, or tests.
    """

    def warn(message: str) -> None:
        if log_warning is not None:
            log_warning(message)

    orbit_payload = payload.get("orbit", {}) or {}
    if orbit_payload:
        try:
            orbit_page.load_data(orbit_payload)
        except Exception as exc:
            warn(f"[Warning] Orbit state restore failed: {exc}")

    propagation_payload = {
        "timeline": payload.get("timeline", {}) or {},
        "integrator": payload.get("integrator", {}) or {},
    }

    solver_payload = payload.get("solver_config", {}) or {}
    method_label = (propagation_payload.get("integrator", {}) or {}).get("method", DEFAULT_SOLVER_METHOD)
    for field_name in ("rtol", "atol", "max_step"):
        if field_name in solver_payload:
            setattr(solver_cfg, field_name, solver_payload[field_name])
    normalize_solver_config_object(
        solver_cfg,
        method_label=method_label,
        upgrade_legacy_defaults=(coerce_positive_float((propagation_payload.get("integrator", {}) or {}).get("rtol")) is None),
    )

    try:
        propagation_page.apply_dict(propagation_payload)
    except Exception as exc:
        warn(f"[Warning] Propagation page restore failed: {exc}")

    forces_payload = payload.get("forces", {}) or {}
    gravity_payload = payload.get("gravity_config", {}) or {}
    if gravity_payload and "gravity" not in forces_payload:
        forces_payload = dict(forces_payload)
        forces_payload["gravity"] = {
            "enabled": True,
            "config": gravity_payload,
        }

    if forces_payload:
        try:
            force_page.load_data(forces_payload)
        except Exception as exc:
            warn(f"[Warning] Force page restore failed: {exc}")

    spacecraft_payload = payload.get("spacecraft_config", {}) or {}
    for field_name in ("mass_kg", "area_m2", "cd", "cr"):
        if field_name in spacecraft_payload:
            setattr(spacecraft_cfg, field_name, spacecraft_payload[field_name])

    output_payload = payload.get("output", {}) or {}
    output_page.apply_state(
        OutputPageState(
            output_dir=str(output_payload.get("dir", str(project_root / "mission_results"))),
            generate_3d_plots=bool(output_payload.get("anim3d", False)),
            downsample_3d=max(1, int(output_payload.get("downsample_3d", 1) or 1)),
        )
    )

    albedo_payload = payload.get("albedo_config", {}) or {}
    for field_name in (
        "label_path",
        "img_path",
        "model",
        "use_ls",
        "sampling",
        "normal_mult",
        "update_interval",
    ):
        if field_name in albedo_payload:
            setattr(albedo_cfg, field_name, albedo_payload[field_name])

    data_payload = payload.get("data_config", {}) or {}
    if data_payload:
        try:
            data_page.apply_state(
                _prefer_dedicated_albedo_root(
                    project_root,
                    DataFilesState(
                    ldem_root=str(data_payload.get("ldem_root", "")) or "",
                    albedo_root=str(data_payload.get("albedo_root", "")) or "",
                    kernel_dir=str(data_payload.get("kernel_dir", "")) or "",
                    ldem_ppd=max(1, int(data_payload.get("ldem_ppd", 4) or 4)),
                    use_ldem_for_albedo=bool(data_payload.get("use_ldem_for_albedo", False)),
                    ),
                )
            )
        except Exception as exc:
            warn(f"[Warning] Data page restore failed: {exc}")

    mc_payload = payload.get("monte_carlo", {}) or {}
    if mc_payload and mc_page is not None:
        try:
            mc_page.load_data(mc_payload)
        except Exception as exc:
            warn(f"[Warning] Monte Carlo page restore failed: {exc}")

    # Surrogate Studio page restore (optional / tolerant)
    surrogate_payload = payload.get("surrogate_studio", {}) or {}
    if surrogate_page is not None and surrogate_payload:
        try:
            surrogate_page.apply_state(surrogate_payload)
        except Exception as exc:
            warn(f"[Warning] Surrogate page restore failed: {exc}")


def collect_visual_state(
    *,
    active_page_key: str = "",
    splitter_sizes: Optional[list[int]] = None,
    log_collapsed: bool = False,
    telemetry_plot_type: str = "",
    telemetry_time_unit: str = "",
    artifact_filter: str = "",
    artifact_recursive: bool = False,
    mc_active_tab: int = 0,
) -> dict[str, Any]:
    """
    Build the ``visual_state`` sub-dict for session persistence.

    All parameters default to safe sentinel values so callers can pass only
    the fields they actually know about.
    """
    return {
        "active_page_key":    active_page_key,
        "splitter_sizes":     list(splitter_sizes) if splitter_sizes else [],
        "log_collapsed":      bool(log_collapsed),
        "telemetry_plot_type": telemetry_plot_type,
        "telemetry_time_unit": telemetry_time_unit,
        "artifact_filter":     artifact_filter,
        "artifact_recursive":  bool(artifact_recursive),
        "mc_active_tab":       int(mc_active_tab),
    }


def apply_visual_state(
    visual: dict[str, Any],
    *,
    main_window: Any = None,
) -> None:
    """
    Restore visual workspace state captured by :py:func:`collect_visual_state`.

    All keys are optional and missing values are silently ignored so old session
    files without a ``visual_state`` block remain loadable.
    """
    if not isinstance(visual, dict) or not visual:
        return

    # Active page
    active_page_key = str(visual.get("active_page_key", "") or "")
    if active_page_key and main_window is not None:
        _safe_call(None, lambda: main_window._switch_page(active_page_key))

    # Main splitter sizes
    splitter_sizes = visual.get("splitter_sizes")
    if splitter_sizes and main_window is not None:
        _safe_call(None, lambda: main_window.main_splitter.setSizes(list(splitter_sizes)))

    # Log collapsed state
    log_collapsed = bool(visual.get("log_collapsed", False))
    if main_window is not None and hasattr(main_window, "is_log_collapsed"):
        _safe_call(None, lambda: _restore_log_collapsed(main_window, log_collapsed))

    # Telemetry page
    plot_type = str(visual.get("telemetry_plot_type", "") or "")
    time_unit = str(visual.get("telemetry_time_unit", "") or "")
    if main_window is not None:
        telem = getattr(main_window, "page_telemetry", None)
        if telem is not None and plot_type:
            _safe_call(None, lambda: _restore_telemetry_visual(telem, plot_type, time_unit))

    # Artifact browser
    artifact_filter = str(visual.get("artifact_filter", "") or "")
    artifact_recursive = bool(visual.get("artifact_recursive", False))
    if main_window is not None:
        output_page = getattr(main_window, "page_output", None)
        if output_page is not None:
            _safe_call(None, lambda: _restore_artifact_visual(output_page, artifact_filter, artifact_recursive))

    # MC active tab
    mc_tab = int(visual.get("mc_active_tab", 0) or 0)
    if main_window is not None:
        mc_page = getattr(main_window, "page_mc", None)
        if mc_page is not None and hasattr(mc_page, "tabs"):
            _safe_call(None, lambda: mc_page.tabs.setCurrentIndex(mc_tab))


def _restore_log_collapsed(mw: Any, collapsed: bool) -> None:
    if mw.is_log_collapsed != collapsed:
        try:
            mw._toggle_log_collapsed()
        except Exception:
            pass


def _restore_telemetry_visual(telem: Any, plot_type: str, time_unit: str) -> None:
    try:
        mp = getattr(telem, "multi_plot", None)
        if mp is None:
            mp = telem
        combo = getattr(mp, "plot_type_combo", None)
        if combo is not None and plot_type:
            idx = combo.findText(plot_type)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        tu_combo = getattr(mp, "time_axis_combo", None)
        if tu_combo is not None and time_unit:
            idx = tu_combo.findText(time_unit)
            if idx >= 0:
                tu_combo.setCurrentIndex(idx)
    except Exception:
        pass


def _restore_artifact_visual(output_page: Any, artifact_filter: str, recursive: bool) -> None:
    try:
        cb = getattr(output_page, "cb_artifact_filter", None)
        if cb is not None and artifact_filter:
            idx = cb.findText(artifact_filter)
            if idx >= 0:
                cb.setCurrentIndex(idx)
        chk = getattr(output_page, "chk_recursive_scan", None)
        if chk is not None:
            chk.setChecked(recursive)
    except Exception:
        pass


def autodetect_data_state(project_root: Path, current_state: DataFilesState) -> tuple[DataFilesState, list[str]]:
    """
    UI adapter for repository data-root auto-discovery.

    The real discovery rules now live in `loaders.io_helpers`, which is the
    correct layer for repository-aware path scanning. This wrapper converts the
    UI page state into loader-side hints, then maps the normalized result back
    into `DataFilesState`.
    """
    detected, messages = autodetect_repository_data_roots(
        project_root,
        current=DataRootHints(
            ldem_root=current_state.ldem_root,
            albedo_root=current_state.albedo_root,
            kernel_dir=current_state.kernel_dir,
            use_ldem_for_albedo=bool(current_state.use_ldem_for_albedo),
        ),
    )

    return (
        DataFilesState(
            ldem_root=detected.ldem_root,
            albedo_root=detected.albedo_root,
            kernel_dir=detected.kernel_dir,
            ldem_ppd=max(1, int(current_state.ldem_ppd or 4)),
            use_ldem_for_albedo=bool(detected.use_ldem_for_albedo),
        ),
        messages,
    )
