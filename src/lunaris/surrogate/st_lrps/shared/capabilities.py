"""Single source of truth for ST-LRPS capability status.

Historically, "this combination is not supported" decisions were scattered as
ad-hoc ``NotImplementedError`` raises across the scaling, runtime, training, and
force-direct modules. That made it impossible to tell, from one place, which
surrogate kind implements which feature, and whether a gap is a deliberate
*by-design* limitation or a *not-yet-implemented* one.

This module is that one place. The :data:`CAPABILITIES` registry maps every
meaningful ``(component, subject, feature)`` triple to a
:class:`CapabilityStatus`, and :class:`UnsupportedCapability` is the single
exception type raised whenever an unavailable capability is requested. It
subclasses :class:`NotImplementedError`, so existing ``except
NotImplementedError`` handlers and ``pytest.raises(NotImplementedError)`` sites
keep working while gaining a uniform, machine-readable message.

The companion document ``docs/ST_LRPS_CAPABILITY_MATRIX.md`` is rendered from
this registry by :func:`render_capability_matrix_markdown`; a test asserts the
two never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CapabilityStatus(str, Enum):
    """Lifecycle status of one ST-LRPS capability."""

    SUPPORTED = "supported"
    """Implemented, tested, and safe to use."""

    NOT_YET = "not_yet"
    """Planned but not implemented; requesting it raises and never silently no-ops."""

    BY_DESIGN_NA = "by_design_na"
    """Intentionally not applicable to this subject (e.g. a scalar potential for a
    direct-force model that only predicts acceleration). Not a gap to be filled."""


@dataclass(frozen=True)
class Capability:
    """One row of the capability matrix."""

    component: str
    """Coarse area: 'runtime', 'baseline', or 'training'."""

    subject: str
    """What the feature is keyed on (a runtime_model_kind, or a target_mode/baseline_kind)."""

    feature: str
    """The concrete capability name."""

    status: CapabilityStatus
    reason: str
    """Human-readable justification — shown in the matrix doc and in errors."""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.component, self.subject, self.feature)


class UnsupportedCapability(NotImplementedError):
    """Raised when an unavailable ST-LRPS capability is requested.

    Subclasses :class:`NotImplementedError` for backward compatibility with
    callers that already catch (or assert on) ``NotImplementedError``.
    """

    def __init__(self, capability: Capability) -> None:
        self.capability = capability
        self.status = capability.status
        super().__init__(
            f"[{capability.status.value}] {capability.component}/{capability.subject}: "
            f"{capability.feature} — {capability.reason}"
        )


# --- The registry -----------------------------------------------------------
# Ordering is significant only for the rendered matrix doc (stable output).

CAPABILITIES: tuple[Capability, ...] = (
    # -- Runtime surrogate kinds: what each loaded model can predict ----------
    Capability(
        "runtime", "potential_autograd", "scalar_residual_potential",
        CapabilityStatus.SUPPORTED,
        "Reference kind: the network outputs DeltaU(x); the scalar residual "
        "potential is read off directly.",
    ),
    Capability(
        "runtime", "potential_autograd", "residual_acceleration",
        CapabilityStatus.SUPPORTED,
        "Delta_a = a_sign * grad(DeltaU_scaled) * (u_scale/x_scale) via autograd.",
    ),
    Capability(
        "runtime", "potential_autograd", "total_acceleration",
        CapabilityStatus.SUPPORTED,
        "a_total = a_base + residual acceleration, with the analytical baseline.",
    ),
    Capability(
        "runtime", "force_direct", "residual_acceleration",
        CapabilityStatus.SUPPORTED,
        "The network outputs the 3-vector residual acceleration directly.",
    ),
    Capability(
        "runtime", "force_direct", "total_acceleration",
        CapabilityStatus.SUPPORTED,
        "a_total = a_base + predicted residual acceleration.",
    ),
    Capability(
        "runtime", "force_direct", "scalar_residual_potential",
        CapabilityStatus.BY_DESIGN_NA,
        "A direct-force model predicts acceleration only; there is no scalar "
        "potential DeltaU to return. Use the acceleration API instead.",
    ),
    # -- Analytical baseline subtracted to form residual targets --------------
    Capability(
        "baseline", "residual", "any_baseline_kind",
        CapabilityStatus.SUPPORTED,
        "Residual datasets already store baseline-subtracted labels, so the "
        "in-layer baseline is exactly zero.",
    ),
    Capability(
        "baseline", "full", "none",
        CapabilityStatus.SUPPORTED,
        "Full-field labels with no analytical baseline; the network sees the "
        "full field.",
    ),
    Capability(
        "baseline", "full", "point_mass",
        CapabilityStatus.SUPPORTED,
        "Monopole baseline U=mu/r, a=-mu r/|r|^3 subtracted analytically.",
    ),
    Capability(
        "baseline", "full", "spherical_harmonics",
        CapabilityStatus.SUPPORTED,
        "SH baseline through base_degree, evaluated by "
        "lunaris.physics.spherical_harmonics; requires the source gravity model "
        "coefficients (threaded in via meta.gravity_model_path).",
    ),
    # -- Training paths -------------------------------------------------------
    Capability(
        "training", "potential_autograd", "sobolev_autograd",
        CapabilityStatus.SUPPORTED,
        "Main trainer (lunaris-train-st-lrps): scalar-potential SIREN with a "
        "Sobolev U/a objective and autograd acceleration.",
    ),
    Capability(
        "training", "force_direct", "direct_residual_accel",
        CapabilityStatus.SUPPORTED,
        "Separate trainer (lunaris-train-force-direct): regress the residual "
        "acceleration directly with a 3-output head.",
    ),
)

_BY_KEY: dict[tuple[str, str, str], Capability] = {c.key: c for c in CAPABILITIES}


def get_capability(component: str, subject: str, feature: str) -> Capability:
    """Return the registered :class:`Capability`, or raise ``KeyError``."""
    return _BY_KEY[(component, subject, feature)]


def require(component: str, subject: str, feature: str) -> Capability:
    """Return the capability if SUPPORTED, else raise :class:`UnsupportedCapability`.

    Unknown keys are a programming error and raise ``KeyError`` (not the softer
    UnsupportedCapability) so typos surface immediately.
    """
    cap = _BY_KEY[(component, subject, feature)]
    if cap.status is not CapabilityStatus.SUPPORTED:
        raise UnsupportedCapability(cap)
    return cap


def require_baseline_supported(target_mode: str, baseline_kind: str) -> Capability:
    """Validate that the (target_mode, baseline_kind) baseline is available.

    Residual datasets carry the baseline implicitly, so every baseline_kind maps
    to the single 'residual/any_baseline_kind' row.
    """
    mode = str(target_mode).strip().lower()
    kind = str(baseline_kind).strip().lower()
    if mode == "residual":
        return require("baseline", "residual", "any_baseline_kind")
    return require("baseline", "full", kind)


def require_runtime_potential(runtime_model_kind: str) -> Capability:
    """Validate that the runtime kind exposes a scalar residual potential."""
    kind = str(runtime_model_kind).strip().lower()
    return require("runtime", kind, "scalar_residual_potential")


def render_capability_matrix_markdown() -> str:
    """Render :data:`CAPABILITIES` as the canonical Markdown matrix document.

    Kept deterministic (registry order, no timestamps) so a test can assert the
    checked-in doc matches the registry byte-for-byte.
    """
    badge = {
        CapabilityStatus.SUPPORTED: "[x] supported",
        CapabilityStatus.NOT_YET: "[ ] not yet",
        CapabilityStatus.BY_DESIGN_NA: "[-] N/A by design",
    }
    lines: list[str] = [
        "# ST-LRPS Capability Matrix",
        "",
        "<!-- GENERATED FROM lunaris.surrogate.st_lrps.shared.capabilities.",
        "     Do not edit by hand — run the regenerate helper and commit. A test",
        "     asserts this file equals render_capability_matrix_markdown(). -->",
        "",
        "Single source of truth for *which ST-LRPS system implements which "
        "feature, and to what extent*. Every unavailable capability raises "
        "`UnsupportedCapability` (a `NotImplementedError` subclass) routed "
        "through this registry, never an ad-hoc error.",
        "",
        "Legend: [x] supported - [ ] not yet implemented - [-] not applicable by design.",
        "",
        "| Component | Subject | Feature | Status | Notes |",
        "|---|---|---|---|---|",
    ]
    for cap in CAPABILITIES:
        lines.append(
            f"| {cap.component} | `{cap.subject}` | {cap.feature} | "
            f"{badge[cap.status]} | {cap.reason} |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "CAPABILITIES",
    "Capability",
    "CapabilityStatus",
    "UnsupportedCapability",
    "get_capability",
    "render_capability_matrix_markdown",
    "require",
    "require_baseline_supported",
    "require_runtime_potential",
]
