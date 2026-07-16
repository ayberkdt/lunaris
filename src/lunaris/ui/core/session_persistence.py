# ST_LRPS/ui_parts/session_persistence.py
"""
Session capture, restore, and data-path auto-detection helpers.

The main window has to coordinate several independently owned pages. Rather
than letting `ui.py` manually reach into every widget for save/restore
operations, this module defines a small persistence layer that works against
page-level APIs (`get_data`, `to_dict`, `get_state`, `apply_state`, etc.).

Design goals
------------
1. Keep serialization rules in one place.
2. Upgrade older saved profiles through a single migration boundary
   (`migrate_session_payload`) so the restore path can assume the canonical
   schema instead of scattering legacy handling everywhere.
3. Keep path auto-detection testable and repository-aware.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from lunaris.ui.core.ui_commons import APP_NAME

# Canonical session schema. Bump SESSION_SCHEMA_VERSION whenever the on-disk
# layout changes in a way that requires migration in migrate_session_payload().
SESSION_SCHEMA_VERSION = 2
# Saved-profile metadata tracks the visible app name (single source of truth in
# ui_commons.APP_NAME). Older profiles written as "ST-LRPS Studio" stay readable:
# migration only keys off ``meta.schema_version`` and never rejects on app name.
SESSION_APP_NAME = APP_NAME


def migrate_session_payload(
    payload: dict[str, Any],
    *,
    log_warning: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Upgrade any saved session payload to the current canonical schema.

    This is the single, explicit migration boundary for session files. All
    backward-compatibility handling for older saves lives here so the rest of
    :func:`apply_session_snapshot` can operate on the canonical schema.

    Parameters
    ----------
    payload:
        Parsed JSON-like dict loaded from disk.
    log_warning:
        Optional callback invoked once when a legacy payload is migrated.

    Returns
    -------
    dict
        A new dict guaranteed to carry ``meta.schema_version`` ==
        :data:`SESSION_SCHEMA_VERSION` and ``meta.app`` ==
        :data:`SESSION_APP_NAME`.
    """

    if not isinstance(payload, dict):
        return {
            "meta": {
                "schema_version": SESSION_SCHEMA_VERSION,
                "app": SESSION_APP_NAME,
            }
        }

    migrated = dict(payload)
    meta = dict(migrated.get("meta", {}) or {})
    try:
        schema_version = int(meta.get("schema_version", 1) or 1)
    except (TypeError, ValueError):
        schema_version = 1

    if schema_version < SESSION_SCHEMA_VERSION and log_warning is not None:
        log_warning(
            f"[Session] Migrating legacy session (schema v{schema_version}) "
            f"to v{SESSION_SCHEMA_VERSION}."
        )

    # Legacy saves stored advanced gravity settings only at the top level
    # (`gravity_config`).  Fold them into the forces payload so the restore path
    # sees one canonical shape.  Idempotent: only injects when absent.
    forces_payload = dict(migrated.get("forces", {}) or {})
    gravity_payload = migrated.get("gravity_config", {}) or {}
    if gravity_payload and "gravity" not in forces_payload:
        forces_payload["gravity"] = {"enabled": True, "config": gravity_payload}
        migrated["forces"] = forces_payload

    meta["schema_version"] = SESSION_SCHEMA_VERSION
    meta.setdefault("app", SESSION_APP_NAME)
    migrated["meta"] = meta
    return migrated

import contextlib

from lunaris.loaders.io_helpers import (
    DataRootHints,
    autodetect_repository_data_roots,
    prefer_dedicated_albedo_root,
)
from lunaris.ui.core.solver_policy import (
    DEFAULT_SOLVER_METHOD,
    coerce_positive_float,
    normalize_solver_config_object,
)
from lunaris.ui.pages.data_files_page import DataFilesState
from lunaris.ui.pages.result_exports_page import OutputPageState


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
    batch_page: Any | None = None,
    thermal_cfg: Any | None = None,
) -> dict[str, Any]:
    """
    Collect a full UI session payload suitable for JSON persistence.

    The payload is written in the canonical schema (``meta.schema_version`` ==
    :data:`SESSION_SCHEMA_VERSION`, ``meta.app`` == :data:`SESSION_APP_NAME`).
    Older saved profiles remain loadable because :func:`apply_session_snapshot`
    runs them through :func:`migrate_session_payload` first.
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
        OutputPageState(output_dir="", generate_3d_plots=False, downsample_3d=1, report_preset="standard"),
        lambda: output_page.get_state(),
    )
    data_state = _safe_call(DataFilesState(), lambda: data_page.get_state())

    return {
        "meta": {
            "schema_version": SESSION_SCHEMA_VERSION,
            "app": SESSION_APP_NAME,
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
            "report_preset": str(getattr(output_state, "report_preset", "standard")),
            # Preserved for compatibility with older session files, even though
            # the dedicated CSV toggle was removed from the UI.
            "csv": True,
        },
        "albedo_config": dataclasses.asdict(albedo_cfg),
        "thermal_config": (
            dataclasses.asdict(thermal_cfg) if thermal_cfg is not None else {}
        ),
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
        # Batch propagation configuration (absent when batch_page is not wired in)
        "batch_propagation": _safe_call({}, lambda: batch_page.get_data()) if batch_page is not None else {},
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
    log_warning: Callable[[str], None] | None = None,
    batch_page: Any | None = None,
    thermal_cfg: Any | None = None,
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

    # Single migration boundary: everything below assumes the canonical schema.
    payload = migrate_session_payload(payload, log_warning=log_warning)

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

    # migrate_session_payload() has already folded any legacy top-level
    # gravity_config into forces["gravity"], so we can read it directly here.
    forces_payload = payload.get("forces", {}) or {}
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
            output_dir=str(output_payload.get("dir", str(project_root / "outputs" / "missions"))),
            generate_3d_plots=bool(output_payload.get("anim3d", False)),
            downsample_3d=max(1, int(output_payload.get("downsample_3d", 1) or 1)),
            report_preset=str(output_payload.get("report_preset", "standard") or "standard"),
        )
    )

    albedo_payload = payload.get("albedo_config", {}) or {}
    for field_name in (
        "model",
        "source",
        "albedo_const",
        "pressure_coefficient",
        "facet_lat_count",
        "facet_lon_count",
        "enable_eclipse",
        "require_provider",
    ):
        if field_name in albedo_payload and hasattr(albedo_cfg, field_name):
            setattr(albedo_cfg, field_name, albedo_payload[field_name])

    if thermal_cfg is not None:
        thermal_payload = payload.get("thermal_config", {}) or {}
        for field_name in (
            "mode",
            "temperature_k",
            "night_temperature_k",
            "emissivity",
            "surface_albedo",
            "ir_coefficient",
            "floor_flux_w_m2",
            "facet_lat_count",
            "facet_lon_count",
        ):
            if field_name in thermal_payload and hasattr(thermal_cfg, field_name):
                setattr(thermal_cfg, field_name, thermal_payload[field_name])

    # Migrate older sessions: the legacy dialog stored free-text model names
    # ("Lambertian", "Lommel-Seeliger") and unrelated knobs. Coerce any unknown
    # backend/source value back to a valid default so command building stays sane.
    if hasattr(albedo_cfg, "model") and str(getattr(albedo_cfg, "model", "")) not in ("lambert_facets", "simple"):
        albedo_cfg.model = "lambert_facets"
    if hasattr(albedo_cfg, "source") and str(getattr(albedo_cfg, "source", "")) not in (
        "constant_albedo",
        "scaled_dn_grid",
        "albedo_grid",
    ):
        albedo_cfg.source = "constant_albedo"

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

    batch_payload = payload.get("batch_propagation", {}) or {}
    if batch_payload and batch_page is not None:
        try:
            batch_page.load_data(batch_payload)
        except Exception as exc:
            warn(f"[Warning] Batch propagation page restore failed: {exc}")


def collect_visual_state(
    *,
    active_page_key: str = "",
    splitter_sizes: list[int] | None = None,
    log_collapsed: bool = False,
    telemetry_plot_type: str = "",
    telemetry_time_unit: str = "",
    artifact_filter: str = "",
    artifact_recursive: bool = False,
    batch_active_tab: int = 0,
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
        "batch_active_tab":       int(batch_active_tab),
    }


def sanitize_splitter_sizes(
    sizes: Any,
    total: int | None = None,
    *,
    min_panel: int = 1,
) -> list[int] | None:
    """Validate/clamp restored splitter sizes; return ``None`` if unusable.

    Restoring raw on-disk splitter values can produce unusable geometry — a
    panel sized to zero (disappears), a negative size (Qt undefined behavior),
    or a sum far larger than the current window (content pushed off-screen).
    This pure helper is the single guard for those cases:

    * rejects non-sequences, fewer than two entries, non-numeric, or negatives;
    * rejects an all-zero layout (would hide everything);
    * when *total* (the live splitter extent) is known and the saved layout is
      larger, scales the sizes proportionally so they fit the current window.
    """
    if not isinstance(sizes, list | tuple) or len(sizes) < 2:
        return None
    try:
        vals = [int(round(float(s))) for s in sizes]
    except (TypeError, ValueError):
        return None
    if any(v < 0 for v in vals):
        return None
    if sum(vals) <= 0:
        return None
    if total is not None and total > 0 and sum(vals) > total:
        scale = total / float(sum(vals))
        vals = [max(min_panel, int(v * scale)) for v in vals]
    return vals


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

    # Main splitter sizes — validated and clamped to the live window so a stale
    # or corrupt session can never produce unusable (zero / negative / oversized)
    # splitter geometry.
    log_collapsed = bool(visual.get("log_collapsed", False))
    if main_window is not None and hasattr(main_window, "is_log_collapsed"):
        _safe_call(None, lambda: _restore_log_collapsed(main_window, log_collapsed))

    splitter_sizes = visual.get("splitter_sizes")
    if splitter_sizes and main_window is not None and not log_collapsed:
        splitter = getattr(main_window, "main_splitter", None)
        if splitter is not None:
            total = _safe_call(0, lambda: int(splitter.height())) or None
            sanitized = sanitize_splitter_sizes(splitter_sizes, total)
            if sanitized is not None:
                _safe_call(None, lambda: splitter.setSizes(sanitized))

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

    # batch active tab
    batch_tab = int(visual.get("batch_active_tab", 0) or 0)
    if main_window is not None:
        batch_page = getattr(main_window, "page_batch", None)
        if batch_page is not None and hasattr(batch_page, "tabs"):
            _safe_call(None, lambda: batch_page.tabs.setCurrentIndex(batch_tab))


def _restore_log_collapsed(mw: Any, collapsed: bool) -> None:
    if mw.is_log_collapsed != collapsed:
        with contextlib.suppress(Exception):
            mw._toggle_log_collapsed()


def _restore_telemetry_visual(telem: Any, plot_type: str, time_unit: str) -> None:
    try:
        mp = getattr(telem, "telemetry_multiplot", None)
        if mp is None:
            mp = telem
        # Plot selection restores by canonical name (stable across the combo ->
        # segmented-control change); set_plot_by_name is a no-op for unknown names.
        if plot_type and hasattr(mp, "set_plot_by_name"):
            mp.set_plot_by_name(plot_type)
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
