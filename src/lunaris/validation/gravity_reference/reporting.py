"""Human-readable reports for gravity-reference validation."""

from __future__ import annotations

from typing import Any


def field_report_markdown(
    *,
    manifest: dict[str, Any],
    status: dict[str, Any],
    metrics: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    """Build a deterministic Markdown field-validation report."""
    lines = [
        f"# Gravity Field Validation: {manifest['benchmark_id']}",
        "",
        f"Status: **{status['status']}**",
        "",
        "## Contract",
        "",
        f"- Reference class: `{manifest['reference_class']}`",
        f"- Gravity model: `{manifest['gravity']['model_name']}`",
        f"- Degree/order: `{manifest['gravity']['degree']}/{manifest['gravity']['order']}`",
        f"- Coordinates: `{manifest['coordinates']['center']}` / `{manifest['coordinates']['frame']}`",
        f"- Threshold origin: `{status['threshold_origin']}`",
        "",
        "## Headline Metrics",
        "",
        f"- Points: {metrics['n_points']}",
        "- Max potential absolute error: "
        f"{metrics['potential_abs_error_m2_s2']['max']:.6e} m^2/s^2",
        "- Max acceleration norm error: "
        f"{metrics['acceleration_norm_error_m_s2']['max']:.6e} m/s^2",
        "- Max acceleration relative norm error: "
        f"{metrics['acceleration_rel_norm_error']['max']:.6e}",
        "- Worst point: "
        f"{metrics['worst_point']['point_id']} "
        f"({metrics['worst_point']['acceleration_norm_error_m_s2']:.6e} m/s^2)",
        "",
        "## Threshold Checks",
        "",
    ]
    if status["checks"]:
        for check in status["checks"]:
            label = "PASS" if check["passed"] else "FAIL"
            lines.append(
                f"- {label}: `{check['name']}` = {check['value']:.6e}, "
                f"limit {check['limit']:.6e}"
            )
    else:
        lines.append("- No pass/fail thresholds were declared; this run is report-only.")
    lines.extend([
        "",
        "## Provenance",
        "",
        f"- Lunaris git commit: `{provenance.get('git_commit')}`",
        f"- Dirty tree: `{provenance.get('git_dirty')}`",
        f"- Python: `{provenance.get('python_version')}`",
    ])
    return "\n".join(lines) + "\n"


def trajectory_report_markdown(
    *,
    manifest: dict[str, Any],
    status: dict[str, Any],
    metrics: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    """Build a deterministic Markdown trajectory-validation report."""
    pos = metrics["position_error_m"]
    vel = metrics["velocity_error_m_s"]
    energy = metrics.get("lunaris_energy_drift", {})
    lines = [
        f"# Gravity Trajectory Validation: {manifest['benchmark_id']}",
        "",
        f"Status: **{status['status']}**",
        "",
        "## Contract",
        "",
        f"- Reference class: `{manifest['reference_class']}`",
        f"- Gravity model: `{manifest['gravity']['model_name']}` "
        f"(degree/order {manifest['gravity']['degree']}/{manifest['gravity']['order']})",
        f"- State frame: `{manifest['frames']['state_frame']}` "
        f"(gravity fixed in `{manifest['frames']['gravity_fixed_frame']}`)",
        f"- Gravity-only: `{manifest['dynamics']['gravity_only']}`",
        f"- Threshold origin: `{status['threshold_origin']}`",
        "",
        "## Headline Metrics",
        "",
        f"- Compared epochs: {metrics['n_epochs']}",
        f"- Max position error: {pos['max']:.6e} m (final {pos['final']:.6e} m)",
        f"- RMS position error: {pos['rms']:.6e} m",
        f"- Max velocity error: {vel['max']:.6e} m/s",
    ]
    if energy:
        lines.append(
            "- Lunaris energy relative drift (max): "
            f"{energy['max_abs_relative_drift']:.6e}"
        )
    lines.extend(["", "## Threshold Checks", ""])
    if status["checks"]:
        for check in status["checks"]:
            label = "PASS" if check["passed"] else "FAIL"
            lines.append(
                f"- {label}: `{check['name']}` = {check['value']:.6e}, "
                f"limit {check['limit']:.6e}"
            )
    else:
        lines.append("- No pass/fail thresholds were declared; this run is report-only.")
    lines.extend([
        "",
        "## Provenance",
        "",
        f"- Lunaris git commit: `{provenance.get('git_commit')}`",
        f"- Dirty tree: `{provenance.get('git_dirty')}`",
        f"- Python: `{provenance.get('python_version')}`",
        f"- Reference SHA-256: `{provenance.get('reference_file_sha256')}`",
    ])
    return "\n".join(lines) + "\n"

