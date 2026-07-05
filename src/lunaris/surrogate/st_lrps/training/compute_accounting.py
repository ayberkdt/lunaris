"""Hardware-independent compute accounting for ST-LRPS training and inference.

Why this module exists
----------------------
Wall-clock time ("training took 5 hours") is **not** a portable measure of how
much computation a model performs: the same run takes 2 hours on one GPU and 10
on another. To state honestly how heavy an architecture is — and to compare two
models fairly regardless of whose machine ran them — we need a *hardware-independent*
unit.

The field-standard unit is the **floating-point operation (FLOP)**, and its
time-scaled form the **petaflop/s-day (PF-day)** introduced by OpenAI's
"AI and Compute" analysis: one PF-day is ``1e15`` FLOP/s sustained for one day,
i.e. ``8.64e19`` FLOP. Total training compute in FLOP (or PF-days) is fixed by
the architecture and the number of samples processed; it does not depend on the
device. GPU-hours, by contrast, are *not* portable (a GPU-hour on an A100 is not
a GPU-hour on a GTX 1660 Ti), so we do not use them as the primary unit.

What is measured vs estimated
-----------------------------
FLOP counts here are **measured**, not guessed, with
``torch.utils.flop_counter.FlopCounterMode`` running on the real network for a
representative batch. The ``potential_autograd`` model evaluates an acceleration
as ``a = a_sign·∇ΔU``, so its acceleration eval includes an autograd gradient
pass (and its training step a double backward). FlopCounterMode counts the
matmuls of whichever path actually runs.

Convention: FlopCounterMode counts a fused multiply-add as **2 FLOP** (a
``(m,k)·(k,n)`` matmul is ``2·m·k·n`` FLOP). Elementwise activations
(``sin``/``silu``) and norms are not counted — they are negligible next to the
linear-layer matmuls that dominate these SIREN/MLP backbones.

Hardware-dependent context (wall-clock seconds, device name, achieved FLOP/s) is
recorded *separately* and clearly labelled, so a reader can recover throughput or
compute model-FLOPs-utilisation without contaminating the portable numbers.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from lunaris.common.constants import DAY_S

logger = logging.getLogger(__name__)

COMPUTE_ACCOUNTING_SCHEMA: str = "st_lrps_compute_accounting_v1"

# One petaflop/s-day = 1e15 FLOP/s * DAY_S. The portable training-compute unit.
PFLOP_S_DAY_IN_FLOPS: float = 1.0e15 * DAY_S  # == 8.64e19

# FlopCounterMode counts a multiply-add as two FLOP (a (m,k)x(k,n) matmul is
# 2*m*k*n). Recorded into the artifact so the unit is unambiguous downstream.
FLOP_CONVENTION: str = "1 multiply-add = 2 FLOP (matmul (m,k)x(k,n) = 2*m*k*n)"

_VALID_MODEL_KINDS = ("potential_autograd",)

# Theoretical peak FP32 (non-tensor-core) throughput, FLOP/s, by a distinctive
# lower-cased substring of ``torch.cuda.get_device_name()``. Sources: NVIDIA
# product datasheets / TechPowerUp GPU database. Used only to report an OPTIONAL
# model-FLOPs-utilization (MFU) figure as machine-dependent context; an unknown
# device simply yields ``None`` (no guessing). Peaks are FP32 non-tensor-core, so
# a run using TF32/AMP has a higher real peak and its MFU-vs-FP32 reads low.
DEVICE_PEAK_FP32_FLOPS: dict[str, float] = {
    "1660 ti": 5.4e12,
    "2080 ti": 13.4e12,
    "t4": 8.1e12,
    "v100": 15.7e12,
    "a100": 19.5e12,
    "3090": 35.6e12,
    "4090": 82.6e12,
    "h100": 67.0e12,
}


def lookup_device_peak_flops(device_name: str | None) -> float | None:
    """Return the theoretical peak FP32 FLOP/s for a device name, or ``None``.

    Matches a distinctive substring (case-insensitive) of the reported device
    name against :data:`DEVICE_PEAK_FP32_FLOPS`. Unknown devices return ``None``
    rather than a guessed peak.
    """
    if not device_name:
        return None
    name = str(device_name).lower()
    # Longest keys first so e.g. "2080 ti" wins over a hypothetical "2080".
    for key in sorted(DEVICE_PEAK_FP32_FLOPS, key=len, reverse=True):
        if key in name:
            return DEVICE_PEAK_FP32_FLOPS[key]
    return None


class _eval_mode:
    """Context manager: put ``model`` in eval mode (so dropout consumes no RNG and
    flop measurement is side-effect-free) and restore the prior mode on exit."""

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._was_training = model.training

    def __enter__(self) -> nn.Module:
        self._model.eval()
        return self._model

    def __exit__(self, *exc: object) -> None:
        self._model.train(self._was_training)


def pflops_days(flops: float) -> float:
    """Convert a raw FLOP count to petaflop/s-days (the portable compute unit)."""
    return float(flops) / PFLOP_S_DAY_IN_FLOPS


def flops_to_human(flops: float) -> str:
    """Render a FLOP count with an SI-ish magnitude suffix (e.g. ``'3.4 TFLOP'``)."""
    f = float(flops)
    for scale, suffix in (
        (1e18, "EFLOP"),
        (1e15, "PFLOP"),
        (1e12, "TFLOP"),
        (1e9, "GFLOP"),
        (1e6, "MFLOP"),
        (1e3, "kFLOP"),
    ):
        if abs(f) >= scale:
            return f"{f / scale:.3g} {suffix}"
    return f"{f:.0f} FLOP"


# ---------------------------------------------------------------------------
# Classical spherical-harmonic gravity: analytic FLOP-per-eval estimate
# ---------------------------------------------------------------------------
# Unlike the surrogate (a torch graph FlopCounterMode can measure exactly), the
# classical SH gravity kernel is a numba ``@njit`` function whose compiled body is
# opaque to FlopCounterMode. To compare the surrogate's per-evaluation arithmetic
# work against "the SH degree it replaces" in the same hardware-independent unit,
# we estimate the SH cost analytically.
#
# The estimate splits a *provably exact* part from a *documented modelling* part:
#   * Exact: the number of harmonic terms a degree-N evaluation touches is the
#     count of (n, m) pairs summed in the kernel's double loop, n = 2..N and
#     m = 0..n  ->  sum_{n=2}^{N} (n + 1) = (N + 4)(N - 1) / 2 terms. This matches
#     the loops in physics/spherical_harmonics.py exactly.
#   * Modelled: the FLOP cost *per term* is a constant derived by reading those
#     kernel hot-loops (a multiply-add counted as 2 FLOP, consistent with the
#     surrogate measurement), broken down below. It is an order-accurate constant,
#     not a measured value, and is exposed as a parameter so a caller can refine it.
#
# Per-term breakdown (physics/spherical_harmonics.py):
#   - normalized ALF vertical recurrence  (A*sinφ*P - B*P2)            ~4
#   - dP/dφ derivative recurrence (two sqrt-based coefficients dominate) ~25
#   - geodesy-convention normalization scaling                         ~3
#   - acceleration assembly + 3x Kahan summation (r/φ/λ components)     ~22
#                                                                     -------
#                                                                      ~54
CLASSICAL_SH_FLOPS_PER_TERM: float = 54.0

# Degree-independent per-eval overhead: the coordinate preamble (r, 1/r, sinφ,
# unit vectors), the central -GM/r^2 term, and the final spherical->Cartesian
# acceleration conversion. Present even at degree 0/1 (point-mass only).
CLASSICAL_SH_BASE_FLOPS: float = 60.0


def classical_sh_terms(degree: int) -> int:
    """Exact number of harmonic ``(n, m)`` terms a degree-``N`` SH eval touches.

    ``sum_{n=2}^{N} (n + 1) = (N + 4)(N - 1) / 2`` (degrees 0 and 1 carry no
    perturbation terms in this kernel, which sums from ``n = 2``).
    """
    n = int(degree)
    if n < 2:
        return 0
    return (n + 4) * (n - 1) // 2


def classical_sh_flops_per_eval(
    degree: int,
    *,
    flops_per_term: float = CLASSICAL_SH_FLOPS_PER_TERM,
    base_flops: float = CLASSICAL_SH_BASE_FLOPS,
    include_base: bool = True,
) -> float:
    """Analytic estimate of FLOP to evaluate a classical SH gravity acceleration.

    This is an **estimate**, not a measurement (the numba kernel is opaque to
    FlopCounterMode): the harmonic-term count is exact, the per-term FLOP constant
    is a documented modelling assumption (see ``CLASSICAL_SH_FLOPS_PER_TERM``).
    Scales as ``O(N^2)`` in the degree. Use it to put a classical SH eval on the
    same hardware-independent axis as the surrogate's measured
    ``inference_flops_per_eval`` — an arithmetic-work comparison only (it ignores
    memory traffic, latency, and parallelism).
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}.")
    flops = classical_sh_terms(degree) * float(flops_per_term)
    if include_base:
        flops += float(base_flops)
    return float(flops)


def compare_eval_cost(
    surrogate_inference_flops_per_eval: float,
    *,
    target_sh_degree: int,
    baseline_sh_degree: int = 0,
    flops_per_term: float = CLASSICAL_SH_FLOPS_PER_TERM,
) -> dict[str, Any]:
    """Compare a surrogate's per-eval cost against the classical SH it replaces.

    A ``potential_autograd`` surrogate models a *residual* ``ΔU`` on top of a
    low-degree analytical baseline, so the honest deployed comparison is **not**
    "surrogate vs full SH". The deployed path evaluates the baseline SH **and**
    the surrogate::

        surrogate_path = SH(baseline_sh_degree) + surrogate

    and is only cheaper, arithmetically, than the full high-fidelity classical
    field when ``surrogate_path < SH(target_sh_degree)``.

    Parameters
    ----------
    surrogate_inference_flops_per_eval:
        The surrogate's **measured** per-acceleration FLOP
        (``ComputeAccounting.inference_flops_per_eval``).
    target_sh_degree:
        The high-fidelity degree the surrogate aims to reproduce (the classical
        eval it stands in for).
    baseline_sh_degree:
        The analytical SH baseline the surrogate adds a residual to (0 = point
        mass only, no SH baseline). Must not exceed ``target_sh_degree``.

    Returns a dict mixing one **measured** number (the surrogate) with
    **analytic SH estimates**; the comparison is *arithmetic work* only (it
    ignores memory traffic, latency, and GPU batch parallelism, where a surrogate
    may still win). ``speedup_vs_target > 1`` means the surrogate path does less
    arithmetic per eval than the full classical field.
    """
    if surrogate_inference_flops_per_eval <= 0.0:
        raise ValueError(
            "surrogate_inference_flops_per_eval must be > 0, got "
            f"{surrogate_inference_flops_per_eval}."
        )
    if baseline_sh_degree > target_sh_degree:
        raise ValueError(
            f"baseline_sh_degree ({baseline_sh_degree}) must not exceed "
            f"target_sh_degree ({target_sh_degree}): a residual cannot sit on a "
            "baseline finer than the field it reproduces."
        )

    surrogate = float(surrogate_inference_flops_per_eval)
    sh_target = classical_sh_flops_per_eval(target_sh_degree, flops_per_term=flops_per_term)
    sh_baseline = classical_sh_flops_per_eval(baseline_sh_degree, flops_per_term=flops_per_term)
    surrogate_path = sh_baseline + surrogate
    residual_terms = classical_sh_terms(target_sh_degree) - classical_sh_terms(baseline_sh_degree)

    return {
        "surrogate_flops_per_eval": surrogate,  # measured
        "surrogate_flops_source": "measured (FlopCounterMode)",
        "sh_target_degree": int(target_sh_degree),
        "sh_target_flops_per_eval": sh_target,  # analytic estimate
        "sh_baseline_degree": int(baseline_sh_degree),
        "sh_baseline_flops_per_eval": sh_baseline,  # analytic estimate
        "sh_flops_source": "analytic_estimate",
        "sh_residual_terms_modelled": int(residual_terms),
        "surrogate_path_flops_per_eval": surrogate_path,  # SH(baseline) + surrogate
        "ratio_vs_target": surrogate_path / sh_target,
        "speedup_vs_target": sh_target / surrogate_path,
        "comparison_basis": "arithmetic_work_only",
        "note": (
            "Surrogate cost is MEASURED; SH costs are ANALYTIC ESTIMATES. The "
            "deployed surrogate path is SH(baseline_degree) + surrogate; "
            "speedup_vs_target > 1 means it does less arithmetic per eval than the "
            "full classical SH(target_degree) field. Arithmetic-work comparison "
            "only — ignores memory, latency, and GPU batch parallelism."
        ),
    }


def build_compute_speed_section(
    *,
    sh_degrees: Mapping[str, int],
    surrogate_model_name: str | None = None,
    surrogate_inference_flops_per_eval: float | None = None,
    surrogate_target_sh_degree: int | None = None,
    surrogate_baseline_sh_degree: int = 0,
    flops_per_term: float = CLASSICAL_SH_FLOPS_PER_TERM,
) -> dict[str, Any]:
    """Assemble a hardware-independent per-eval cost section for a benchmark report.

    Puts every benchmarked gravity model on one portable axis (FLOP per
    acceleration evaluation), to sit *alongside* the benchmark's machine-dependent
    wall-clock runtime — not replace it.

    Parameters
    ----------
    sh_degrees:
        Map of classical-SH model name -> degree, e.g. ``{"sh32": 32, "sh120": 120}``.
        Their FLOP cost is the analytic estimate.
    surrogate_model_name, surrogate_inference_flops_per_eval:
        If both given, the surrogate is added with its **measured** per-eval FLOP.
    surrogate_target_sh_degree, surrogate_baseline_sh_degree:
        If the surrogate's target degree is known, a deployed-path comparison
        (:func:`compare_eval_cost`) is included.

    Always returns a self-describing dict; no measurement or I/O is performed here.
    """
    per_model: dict[str, dict[str, Any]] = {}
    for name, degree in sh_degrees.items():
        per_model[str(name)] = {
            "flops_per_eval": classical_sh_flops_per_eval(int(degree), flops_per_term=flops_per_term),
            "source": "analytic_estimate",
            "sh_degree": int(degree),
        }

    surrogate_vs_target: dict[str, Any] | None = None
    if surrogate_model_name and surrogate_inference_flops_per_eval is not None:
        per_model[str(surrogate_model_name)] = {
            "flops_per_eval": float(surrogate_inference_flops_per_eval),
            "source": "measured (FlopCounterMode)",
        }
        if surrogate_target_sh_degree is not None:
            surrogate_vs_target = compare_eval_cost(
                float(surrogate_inference_flops_per_eval),
                target_sh_degree=int(surrogate_target_sh_degree),
                baseline_sh_degree=int(surrogate_baseline_sh_degree),
                flops_per_term=flops_per_term,
            )

    return {
        "unit": "FLOP per acceleration evaluation",
        "flop_convention": FLOP_CONVENTION,
        "comparison_basis": "arithmetic_work_only",
        "note": (
            "Hardware-independent per-eval arithmetic cost, to read ALONGSIDE the "
            "machine-dependent wall-clock runtime (not as a replacement). SH costs "
            "are analytic estimates; the surrogate cost is measured. Ignores memory "
            "traffic, latency, and GPU batch parallelism."
        ),
        "per_model": per_model,
        "surrogate_vs_target": surrogate_vs_target,
    }


def _fmt_flops_per_s(value: float | None) -> str:
    return f"{flops_to_human(value)}/s" if value is not None else "n/a"


def render_compute_report(
    compute_accounting: Mapping[str, Any] | None = None,
    compute_speed: Mapping[str, Any] | None = None,
) -> str:
    """Render a human-readable Markdown "Compute & Speed" section for a report.

    Pure and best-effort: tolerates missing keys / ``None`` inputs. Leads with the
    portable (hardware-independent) numbers and keeps machine-dependent context in
    a clearly-labelled sub-line, mirroring the JSON layout.
    """
    lines: list[str] = ["## Compute & Speed", ""]

    if compute_accounting:
        ca = compute_accounting
        pf = ca.get("total_training_pflops_days")
        total = ca.get("total_training_flops")
        infer = ca.get("inference_flops_per_eval")
        lines.append("**Training compute (hardware-independent):**")
        lines.append("")
        if pf is not None and total is not None:
            lines.append(f"- Total: **{float(pf):.3e} PF-days** ({flops_to_human(float(total))})")
        if ca.get("total_samples_processed") is not None:
            lines.append(f"- Samples processed: {int(ca['total_samples_processed']):,}")
        if ca.get("n_params") is not None:
            lines.append(f"- Parameters: {int(ca['n_params']):,}")
        if infer is not None:
            lines.append(f"- Inference: **{flops_to_human(float(infer))}/eval** (per acceleration)")
        hw = ca.get("hardware") or {}
        if hw:
            dev = hw.get("device") or "unknown"
            wall = hw.get("wall_clock_seconds")
            ach = hw.get("achieved_flops_per_s")
            mfu = hw.get("model_flops_utilization")
            ctx = f"- _Machine-dependent context_: {dev}"
            if wall is not None:
                ctx += f", {float(wall):.0f}s wall-clock"
            if ach is not None:
                ctx += f", {_fmt_flops_per_s(float(ach))} achieved"
            if mfu is not None:
                ctx += f", MFU ~{100.0 * float(mfu):.1f}% (vs FP32 peak)"
            lines.append("")
            lines.append(ctx)
        lines.append("")

    if compute_speed and compute_speed.get("per_model"):
        lines.append("**Per-eval cost (arithmetic work only — read alongside wall-clock runtime):**")
        lines.append("")
        lines.append("| Model | FLOP/eval | Source |")
        lines.append("|---|---:|---|")
        for name, info in compute_speed["per_model"].items():
            fpe = info.get("flops_per_eval")
            src = info.get("source", "")
            fpe_str = flops_to_human(float(fpe)) if fpe is not None else "n/a"
            lines.append(f"| {name} | {fpe_str} | {src} |")
        cmp = compute_speed.get("surrogate_vs_target")
        if cmp:
            sp = cmp.get("speedup_vs_target")
            tgt = cmp.get("sh_target_degree")
            if sp is not None and tgt is not None:
                verdict = "fewer" if float(sp) > 1.0 else "more"
                lines.append("")
                lines.append(
                    f"- Deployed surrogate path vs full SH degree-{tgt}: "
                    f"**{float(sp):.2f}× speedup** "
                    f"(does {verdict} arithmetic per eval; surrogate measured, SH estimated)."
                )
        lines.append("")

    if len(lines) <= 2:
        lines.append("_No compute accounting available._")
    return "\n".join(lines).rstrip() + "\n"


def _measure_flops(fn: Callable[[], Any]) -> int:
    """Run ``fn`` under ``FlopCounterMode`` and return the total FLOP count."""
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return int(counter.get_total_flops())


def _as_2d(sample_input: torch.Tensor) -> torch.Tensor:
    if sample_input.dim() != 2:
        raise ValueError(
            f"sample_input must be 2-D (batch, in_dim); got shape {tuple(sample_input.shape)}."
        )
    if sample_input.shape[0] < 1:
        raise ValueError("sample_input must contain at least one sample.")
    return sample_input


def _check_model_kind(model_kind: str) -> str:
    mk = str(model_kind)
    if mk not in _VALID_MODEL_KINDS:
        raise ValueError(
            f"Unsupported model_kind={mk!r}; expected one of {_VALID_MODEL_KINDS}."
        )
    return mk


def measure_forward_flops_per_sample(model: nn.Module, sample_input: torch.Tensor) -> float:
    """Measure FLOP for a single forward (``ΔU`` or ``Δa``), per input sample."""
    x = _as_2d(sample_input)
    batch = int(x.shape[0])
    with _eval_mode(model):
        with torch.no_grad():
            total = _measure_flops(lambda: model(x.detach()))
    return float(total) / batch


def measure_inference_flops_per_eval(
    model: nn.Module, sample_input: torch.Tensor, *, model_kind: str
) -> float:
    """Measure FLOP to evaluate the surrogate **acceleration** at one query point.

    This is the honest, hardware-independent "model speed" number — directly
    comparable across surrogates (and against a classical SH model's per-eval FLOP
    cost). For ``potential_autograd`` it includes the ``∇ΔU`` autograd pass that
    runtime force evaluation actually performs.
    """
    _check_model_kind(model_kind)
    x = _as_2d(sample_input)
    batch = int(x.shape[0])
    with _eval_mode(model):
        # potential_autograd: a = ∇ΔU requires an autograd gradient pass.
        # Use .backward() rather than autograd.grad(): the latter triggers a
        # FlopCounterMode incompatibility ("leaf node ... _will_engine_execute_node")
        # when the gradient target is a leaf input. A reverse pass through the
        # whole graph has the same matmul FLOP either way.
        def _accel_eval() -> None:
            model.zero_grad(set_to_none=True)
            xr = x.detach().clone().requires_grad_(True)
            u = model(xr)
            u.sum().backward()

        total = _measure_flops(_accel_eval)
        model.zero_grad(set_to_none=True)
    return float(total) / batch


def measure_train_step_flops_per_sample(
    model: nn.Module, sample_input: torch.Tensor, *, model_kind: str
) -> float:
    """Measure FLOP for one full training step (forward + backward), per sample.

    Uses a trivial scalar reduction to drive the backward so the measurement is
    isolated from the real Sobolev loss (whose extra terms are negligible next to
    the backbone matmuls). For ``potential_autograd`` the forward builds the
    acceleration via a graph-retaining gradient, so the measured backward is the
    true double-backward through the network parameters. The model weights are
    **not** modified (no optimiser step); gradients are zeroed afterwards.
    """
    _check_model_kind(model_kind)
    x = _as_2d(sample_input)
    batch = int(x.shape[0])
    with _eval_mode(model):
        model.zero_grad(set_to_none=True)

        def _train_step() -> None:
            # potential_autograd: forward U, then ∇U (graph-retaining) via
            # .backward(create_graph=True), then an outer backward on the
            # acceleration loss — the true double-backward through the network
            # parameters. .backward() is used instead of autograd.grad() to avoid
            # the FlopCounterMode leaf-target incompatibility (see
            # measure_inference_flops_per_eval).
            xr = x.detach().clone().requires_grad_(True)
            u = model(xr)
            u.sum().backward(create_graph=True)
            a = xr.grad
            assert a is not None
            loss = a.pow(2).sum()
            loss.backward()

        with warnings.catch_warnings():
            # create_graph=True warns about a param/grad reference cycle; we break
            # it explicitly with zero_grad(set_to_none=True) right after, so the
            # leak it warns about does not occur for this one-shot measurement.
            warnings.filterwarnings("ignore", message=".*create_graph=True.*")
            total = _measure_flops(_train_step)
        model.zero_grad(set_to_none=True)
    return float(total) / batch


@dataclass(frozen=True)
class ComputeAccounting:
    """Portable compute accounting for one ST-LRPS run.

    Portable (hardware-independent) fields are FLOP-based. The ``hardware`` block
    in :meth:`to_manifest_dict` is the only machine-dependent context.
    """

    model_kind: str
    n_params: int
    measured_batch_size: int
    forward_flops_per_sample: float
    inference_flops_per_eval: float
    train_step_flops_per_sample: float
    total_samples_processed: int
    total_training_flops: float
    total_training_pflops_days: float
    wall_clock_seconds: float | None = None
    device: str | None = None
    achieved_flops_per_s: float | None = None
    device_peak_flops_per_s: float | None = None
    model_flops_utilization: float | None = None

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMPUTE_ACCOUNTING_SCHEMA,
            "flop_convention": FLOP_CONVENTION,
            "flop_counter": "torch.utils.flop_counter.FlopCounterMode",
            "model_kind": self.model_kind,
            "n_params": int(self.n_params),
            "measured_batch_size": int(self.measured_batch_size),
            # Hardware-independent model speed (per acceleration evaluation).
            "inference_flops_per_eval": float(self.inference_flops_per_eval),
            "forward_flops_per_sample": float(self.forward_flops_per_sample),
            "train_step_flops_per_sample": float(self.train_step_flops_per_sample),
            # Hardware-independent total training compute.
            "total_samples_processed": int(self.total_samples_processed),
            "total_training_flops": float(self.total_training_flops),
            "total_training_pflops_days": float(self.total_training_pflops_days),
            "total_training_flops_human": flops_to_human(self.total_training_flops),
            # Machine-dependent context — clearly separated, never the headline.
            "hardware": {
                "device": self.device,
                "wall_clock_seconds": (
                    float(self.wall_clock_seconds)
                    if self.wall_clock_seconds is not None
                    else None
                ),
                "achieved_flops_per_s": (
                    float(self.achieved_flops_per_s)
                    if self.achieved_flops_per_s is not None
                    else None
                ),
                "device_peak_fp32_flops_per_s": (
                    float(self.device_peak_flops_per_s)
                    if self.device_peak_flops_per_s is not None
                    else None
                ),
                # End-to-end MFU vs theoretical FP32 peak: a conservative lower
                # bound (matmul-only FLOP; includes data-loading/checkpoint
                # overhead; FP32 peak, so TF32/AMP runs read low). None if the
                # device peak is unknown or wall-clock is missing.
                "model_flops_utilization": (
                    float(self.model_flops_utilization)
                    if self.model_flops_utilization is not None
                    else None
                ),
            },
        }

    def summary_line(self) -> str:
        parts = [
            f"compute: {self.total_training_pflops_days:.3e} PF-days "
            f"({flops_to_human(self.total_training_flops)})",
            f"inference {flops_to_human(self.inference_flops_per_eval)}/eval",
        ]
        if self.wall_clock_seconds is not None and self.achieved_flops_per_s is not None:
            parts.append(
                f"[{self.wall_clock_seconds:.0f}s on {self.device or 'unknown'}, "
                f"{flops_to_human(self.achieved_flops_per_s)}/s]"
            )
        return " | ".join(parts)


def build_compute_accounting(
    model: nn.Module,
    sample_input: torch.Tensor,
    *,
    model_kind: str,
    total_samples_processed: int,
    wall_clock_seconds: float | None = None,
    device: str | None = None,
) -> ComputeAccounting:
    """Measure per-sample FLOP on ``model`` and assemble the run's compute account.

    Parameters
    ----------
    model:
        The built network (``PhysicsNet``); weights are not modified.
    sample_input:
        A representative ``(batch, in_dim)`` input in the network's scaled
        coordinates (the same thing the training loop feeds the model).
    model_kind:
        ``"potential_autograd"``.
    total_samples_processed:
        Total number of training samples the optimiser actually consumed across
        the whole run (sum of batch sizes over all steps). The portable training
        total is ``train_step_flops_per_sample * total_samples_processed``.
    wall_clock_seconds, device:
        Optional machine-dependent context. If both wall-clock and the total are
        available, ``achieved_flops_per_s`` is derived for reference only.
    """
    mk = _check_model_kind(model_kind)
    if total_samples_processed < 0:
        raise ValueError(f"total_samples_processed must be >= 0, got {total_samples_processed}.")

    batch = int(_as_2d(sample_input).shape[0])
    n_params = int(sum(p.numel() for p in model.parameters()))

    fwd = measure_forward_flops_per_sample(model, sample_input)
    infer = measure_inference_flops_per_eval(model, sample_input, model_kind=mk)
    step = measure_train_step_flops_per_sample(model, sample_input, model_kind=mk)

    total_flops = step * float(total_samples_processed)
    achieved: float | None = None
    if wall_clock_seconds is not None and wall_clock_seconds > 0.0:
        achieved = total_flops / float(wall_clock_seconds)

    device_peak = lookup_device_peak_flops(device)
    mfu: float | None = None
    if achieved is not None and device_peak is not None and device_peak > 0.0:
        mfu = achieved / device_peak

    return ComputeAccounting(
        model_kind=mk,
        n_params=n_params,
        measured_batch_size=batch,
        forward_flops_per_sample=fwd,
        inference_flops_per_eval=infer,
        train_step_flops_per_sample=step,
        total_samples_processed=int(total_samples_processed),
        total_training_flops=total_flops,
        total_training_pflops_days=pflops_days(total_flops),
        wall_clock_seconds=wall_clock_seconds,
        device=device,
        achieved_flops_per_s=achieved,
        device_peak_flops_per_s=device_peak,
        model_flops_utilization=mfu,
    )


__all__ = [
    "COMPUTE_ACCOUNTING_SCHEMA",
    "PFLOP_S_DAY_IN_FLOPS",
    "FLOP_CONVENTION",
    "CLASSICAL_SH_FLOPS_PER_TERM",
    "CLASSICAL_SH_BASE_FLOPS",
    "DEVICE_PEAK_FP32_FLOPS",
    "ComputeAccounting",
    "build_compute_accounting",
    "lookup_device_peak_flops",
    "classical_sh_terms",
    "classical_sh_flops_per_eval",
    "compare_eval_cost",
    "build_compute_speed_section",
    "render_compute_report",
    "measure_forward_flops_per_sample",
    "measure_inference_flops_per_eval",
    "measure_train_step_flops_per_sample",
    "pflops_days",
    "flops_to_human",
]
