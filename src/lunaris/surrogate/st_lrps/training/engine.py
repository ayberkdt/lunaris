"""
Training engine for the lunar scalar potential surrogate.

The engine is the orchestration layer: it receives a validated ``TrainConfig``,
loads lunar residual clouds, fits or restores scalers, builds the neural field,
executes the Sobolev training loop, and writes checkpoints/metrics.

Design notes
------------
* The learned quantity is scalar residual potential ``dU``.
* Residual acceleration ``da`` is computed from the autograd gradient of ``dU``.
* Validation keeps gradients enabled because acceleration metrics require that
  derivative path.
* Best-checkpoint selection can be delayed until direction loss is active, so a
  physically incomplete early epoch does not become the preferred checkpoint.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from lunaris.surrogate.st_lrps.artifacts.manager import (
    atomic_write_json,
    build_checkpoint_payload,
    build_resolved_config,
    capture_environment_snapshot,
    capture_rng_state,
    compute_file_sha256,
    compute_payload_sha256,
    ensure_run_layout,
    read_run_manifest,
    resolve_git_commit,
    resolve_resume_checkpoint,
    restore_rng_state,
    save_checkpoint,
    update_run_manifest,
    verify_critical_config_fields_match,
    write_command_txt,
    write_run_manifest,
    write_scaler_json,
)
from lunaris.surrogate.st_lrps.data.dataset_contract import DatasetContract
from lunaris.surrogate.st_lrps.data.dataset_parameters import R_MOON_SI
from lunaris.surrogate.st_lrps.data.dataset_validation import validate_dataset_file
from lunaris.surrogate.st_lrps.data.datasets import (
    DTYPE,
    BlockShuffleSampler,
    DatasetMeta,
    H5BlockDataset,
    TensorBatchSampler,
    TensorMemoryDataset,
    _discover_dataset_name,
    _resolve_loader_worker_count,
    _resolve_lunar_dataset_contract,
    build_dataset_contract,
    collate_xyz_u_a,
    identity_collate,
    infer_a_sign_from_data,
    validate_training_dataset_convention,
)
from lunaris.surrogate.st_lrps.data.splits import (
    SPLIT_INDICES_FILENAME,
    build_split_manifest,
    split_dataset_indices,
    write_split_indices,
    write_split_manifest,
)
from lunaris.surrogate.st_lrps.networks.models import (
    MODEL_BUILDER_VERSION,
    _compute_harmonic_w0_bands,
    _get_output_head_params,
    build_model_from_config,
    compute_architecture_signature,
)
from lunaris.surrogate.st_lrps.shared.contracts import TargetContract, resolve_dataset_hash
from lunaris.surrogate.st_lrps.shared.scaling import ScalerPack, fit_scaler_streaming
from lunaris.surrogate.st_lrps.training.checkpoint_manager import (
    BestCheckpointTracker,
    resolve_best_ckpt_start_epoch,
)
from lunaris.surrogate.st_lrps.training.config import (
    TrainConfig,
    apply_model_preset,
    apply_run_preset,
)
from lunaris.surrogate.st_lrps.training.config_summary import build_experiment_feature_summary
from lunaris.surrogate.st_lrps.training.losses import (
    GradNormWeights,
    LossCurriculum,
    SobolevLoss,
    _direction_loss_factor,
    collocation_laplacian_loss,
)
from lunaris.surrogate.st_lrps.training.metrics import (
    HISTORY_FIELDNAMES,
    checkpoint_selection_block,
    compute_checkpoint_score,
    flatten_epoch_metrics,
    format_batch_summary,
    format_epoch_summary,
    normalize_best_metric,
)
from lunaris.surrogate.st_lrps.training.periodic_eval import (
    completed_periodic_eval_epochs,
    resolve_periodic_eval_plan,
    run_periodic_eval,
)
from lunaris.surrogate.st_lrps.training.preflight import (
    PreflightError,
    run_preflight,
    write_preflight_report,
)

logger = logging.getLogger(__name__)


def _log_section(title: str, values: Mapping[str, Any]) -> None:
    logger.info(f"=== {title} ===")
    for key, value in values.items():
        logger.info(f"{str(key):24s}: {value}")


def set_seed(
    seed: int = 42, *, deterministic: bool = True, benchmark: bool = False
) -> dict[str, Any]:
    """
    Fixes all random number generator (RNG) seeds for reproducibility.

    ``deterministic`` / ``benchmark`` control cuDNN behavior. Defaults preserve
    the historical deterministic configuration; pass ``deterministic=False`` /
    ``benchmark=True`` for throughput at the cost of run-to-run reproducibility.

    In ``deterministic`` mode this also (a) enables
    ``torch.use_deterministic_algorithms(warn_only=True)`` so non-cuDNN CUDA ops
    take a reproducible path where one exists, (b) pins TF32 off so matmul/conv
    numerics match across Ampere+ GPUs, and (c) sets the cuBLAS workspace /
    ``PYTHONHASHSEED`` env vars the deterministic toggle and worker subprocesses
    require. Returns the applied flags so the caller can record them in the run
    provenance.
    """
    # PYTHONHASHSEED only affects interpreters started *after* it is set (e.g.
    # 'spawn' DataLoader workers), not this live process — set it for the workers
    # and for provenance, not for in-process hash randomization.
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark)

    applied: dict[str, Any] = {
        "seed": int(seed),
        "cudnn_deterministic": bool(deterministic),
        "cudnn_benchmark": bool(benchmark),
        "pythonhashseed": str(int(seed)),
    }

    # TF32 changes matmul/conv numerics on Ampere+; pin it OFF in deterministic
    # mode so a run reproduces across hardware. Best-effort (older torch).
    try:
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
        torch.backends.cudnn.allow_tf32 = not deterministic
        applied["allow_tf32"] = not deterministic
    except Exception:  # pragma: no cover - depends on torch/build
        applied["allow_tf32"] = None

    if deterministic:
        # use_deterministic_algorithms raises at the first nondeterministic CUDA
        # op unless cuBLAS has a fixed workspace; set it before the toggle.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        applied["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
        try:
            # warn_only: surface any op without a deterministic kernel instead of
            # hard-failing a long training run.
            torch.use_deterministic_algorithms(True, warn_only=True)
            applied["use_deterministic_algorithms"] = "warn_only"
        except Exception as exc:  # pragma: no cover - torch/version dependent
            logger.warning("set_seed: use_deterministic_algorithms unavailable: %s", exc)
            applied["use_deterministic_algorithms"] = False
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:  # pragma: no cover
            pass
        applied["use_deterministic_algorithms"] = False

    logger.info("set_seed applied determinism flags: %s", applied)
    return applied


def _seed_dataloader_worker(worker_id: int) -> None:
    """Seed numpy + Python RNG per DataLoader worker for reproducible draws.

    PyTorch reseeds each worker's torch RNG automatically (``base_seed +
    worker_id``) but leaves numpy/Python untouched, so any ``np.random`` /
    ``random`` use inside a dataset's ``__getitem__`` would be nondeterministic
    across runs. Derive both from torch's per-worker base seed.
    """
    base = int(torch.initial_seed()) % (2**32)
    worker_seed = (base + int(worker_id)) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_device() -> torch.device:
    """
    Selects the best available hardware accelerator.
    Priority: CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def safe_mkdir(p: str | Path) -> Path:
    path_obj = Path(p)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def _human_bytes(n: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB", "PB", "EB")
    size = float(n)
    for unit in units[:-1]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} {units[-1]}"

def _format_seconds(seconds: float) -> str:
    """Format a duration into a compact human-readable string."""
    s = float(seconds)
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{int(s):02d}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{int(s):02d}s"

def _format_cuda_memory_mib(
    allocated_mib: int,
    reserved_mib: int,
    peak_allocated_mib: int,
    peak_reserved_mib: int,
    total_vram_mib: int,
) -> str:
    return (
        f" cuda_mem={allocated_mib}/{reserved_mib}MiB"
        f" peak={peak_allocated_mib}/{peak_reserved_mib}MiB"
        f" total={total_vram_mib}MiB"
    )


def _cuda_memory_string(device: torch.device) -> str:
    """Return compact PyTorch CUDA allocator memory diagnostics.

    ``allocated`` and ``reserved`` are PyTorch allocator values, not total
    physical GPU VRAM and not the same accounting shown by ``nvidia-smi``.
    ``peak`` values are the allocator maxima since peak stats were last reset,
    which makes them more useful for batch-size decisions than the current
    post-batch allocation alone.
    """
    if device.type != "cuda":
        return ""
    try:
        mib = 1024 * 1024
        alloc_mib = int(torch.cuda.memory_allocated(device) // mib)
        reserved_mib = int(torch.cuda.memory_reserved(device) // mib)
        peak_alloc_mib = int(torch.cuda.max_memory_allocated(device) // mib)
        peak_reserved_mib = int(torch.cuda.max_memory_reserved(device) // mib)
        total_mib = int(torch.cuda.get_device_properties(device).total_memory // mib)
    except Exception:
        return ""
    return _format_cuda_memory_mib(
        alloc_mib,
        reserved_mib,
        peak_alloc_mib,
        peak_reserved_mib,
        total_mib,
    )


_CGROUP_LIMIT_FILES = (
    # (limit file, current-usage file) — cgroup v2 first, then v1.
    ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
    ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
)

# Values at/above this are the kernel's "no limit" sentinel (v1 reports
# PAGE_COUNTER_MAX, ~2^63; v2 reports the literal string "max").
_CGROUP_UNLIMITED_BYTES = float(2**60)


def _candidate_cgroup_limit_files() -> tuple[tuple[str, str], ...]:
    """Find memory-controller files for the process cgroup and its fallbacks.

    Slurm commonly places a job in a nested cgroup (for example below
    ``/sys/fs/cgroup/slurm/...``), so checking only the cgroup mount root can
    accidentally read the container/host budget instead of the job budget.
    The canonical mount-root candidates remain as a fallback for containers
    that hide ``/proc/self/cgroup``.
    """
    candidates: list[tuple[str, str]] = []
    try:
        entries = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except OSError:
        entries = []
    cgroup_root = Path("/sys/fs/cgroup")
    for entry in entries:
        try:
            _hierarchy, controllers, rel = entry.split(":", 2)
        except ValueError:
            continue
        rel_path = Path(rel.lstrip("/"))
        if not controllers:
            base = cgroup_root / rel_path
        elif "memory" in controllers.split(","):
            base = cgroup_root / "memory" / rel_path
        else:
            continue
        candidates.append((str(base / "memory.max"), str(base / "memory.current")))
        candidates.append(
            (str(base / "memory.limit_in_bytes"), str(base / "memory.usage_in_bytes"))
        )
    candidates.extend(_CGROUP_LIMIT_FILES)
    return tuple(dict.fromkeys(candidates))


def _cgroup_available_mb(
    limit_files: tuple[tuple[str, str], ...] | None = None,
) -> float | None:
    """Remaining memory (MB) inside this process's cgroup, or ``None``.

    Slurm enforces ``--mem`` through cgroups, so on a shared compute node the
    node-wide "available RAM" can be far larger than what THIS job may touch.
    Returns ``limit - current_usage`` when both are readable, the bare limit
    when only the limit is, and ``None`` when there is no (finite) cgroup
    limit — e.g. on a laptop or an unconfined container.
    """
    for limit_path, usage_path in (
        _candidate_cgroup_limit_files() if limit_files is None else limit_files
    ):
        try:
            raw = Path(limit_path).read_text(encoding="ascii").strip()
        except OSError:
            continue
        if raw == "max":
            continue
        try:
            limit_b = float(raw)
        except ValueError:
            continue
        if limit_b <= 0 or limit_b >= _CGROUP_UNLIMITED_BYTES:
            continue
        used_b = 0.0
        try:
            used_b = float(Path(usage_path).read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            used_b = 0.0
        return max(0.0, limit_b - used_b) / (1024.0 * 1024.0)
    return None


def _slurm_mem_limit_mb(environ: Mapping[str, str] | None = None) -> float | None:
    """Job memory limit (MB) implied by Slurm environment, or ``None``.

    ``SLURM_MEM_PER_NODE`` is the ``--mem`` request in MB. When only
    ``SLURM_MEM_PER_CPU`` is set, the job limit is per-CPU × allocated CPUs
    (``SLURM_CPUS_ON_NODE``, falling back to ``SLURM_CPUS_PER_TASK``).
    Values may carry a K/M/G/T suffix in some site configurations.
    """
    env = os.environ if environ is None else environ

    def _mb(raw: str | None) -> float | None:
        text = str(raw or "").strip()
        if not text:
            return None
        factor = 1.0  # plain numbers are MB per Slurm documentation
        suffix = text[-1].upper()
        if suffix in "KMGT":
            factor = {"K": 1.0 / 1024.0, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}[suffix]
            text = text[:-1]
        try:
            value = float(text)
        except ValueError:
            return None
        return value * factor if value > 0 else None

    per_node = _mb(env.get("SLURM_MEM_PER_NODE"))
    if per_node is not None:
        return per_node
    per_cpu = _mb(env.get("SLURM_MEM_PER_CPU"))
    if per_cpu is None:
        return None
    cpus = 1.0
    for key in ("SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK"):
        try:
            cpus = float(str(env.get(key, "")).strip())
            if cpus > 0:
                break
        except ValueError:
            cpus = 1.0
    return per_cpu * max(1.0, cpus)


def _available_ram_mb() -> float | None:
    """Return the RAM budget (MB) actually available to THIS process, or None.

    Takes the minimum of three independent signals so the preload safety veto
    holds on HPC nodes, not just workstations:

    * ``psutil.virtual_memory().available`` — node-wide availability (optional
      dependency; skipped when psutil is missing);
    * the cgroup limit minus current usage — what Slurm's ``--mem`` actually
      enforces on a shared node (node-wide RAM can be far larger);
    * the Slurm-declared job limit from the environment — a fallback when the
      cgroup files are not readable from inside the job.

    Returns ``None`` only when none of the signals is available; the caller
    then skips the RAM-safety veto (and says so in the preload decision log).
    """
    candidates: list[float] = []
    try:
        import psutil  # optional

        candidates.append(float(psutil.virtual_memory().available) / (1024.0 * 1024.0))
    except Exception:
        pass
    cgroup_mb = _cgroup_available_mb()
    if cgroup_mb is not None:
        candidates.append(cgroup_mb)
    slurm_mb = _slurm_mem_limit_mb()
    if slurm_mb is not None:
        candidates.append(slurm_mb)
    if not candidates:
        return None
    return min(candidates)


def _estimate_preload_ram_mb(n_rows: int) -> float:
    """Estimate peak RAM (MB) to preload ``n_rows`` of [x,y,z,U,ax,ay,az].

    The preload path reads the whole array as float64 ``(N, 7)`` and then builds
    float32 train/val copies, so the transient peak is roughly the float64 buffer
    plus the float32 copies.
    """
    f64 = float(n_rows) * 7.0 * 8.0
    f32 = float(n_rows) * 7.0 * 4.0
    return (f64 + f32) / (1024.0 * 1024.0)


def _decide_preload(
    policy: str,
    *,
    dataset_mb: float,
    auto_preload_mb: float,
    est_ram_mb: float,
    avail_ram_mb: float | None,
) -> tuple[bool, str]:
    """Resolve whether to RAM-preload the dataset and explain why.

    Returns ``(should_preload, reason)``. The 60%-of-available-RAM guard vetoes
    the ``auto`` decision; under an explicit ``always`` request it does not veto
    but emits a loud warning embedded in the reason string.
    """
    policy = str(policy).strip().lower()
    over_ram = (avail_ram_mb is not None) and (est_ram_mb > 0.60 * avail_ram_mb)

    if policy == "never":
        return False, "policy=never"
    if policy == "always":
        if avail_ram_mb is None:
            return True, (
                "policy=always (WARNING: unknown RAM budget; the RAM-safety "
                "veto cannot be evaluated — explicit preload request honoured)"
            )
        if over_ram:
            return True, (
                f"policy=always (WARNING: estimated {est_ram_mb:.0f} MB exceeds 60% of "
                f"available {avail_ram_mb:.0f} MB - OOM risk; honouring explicit request)"
            )
        return True, "policy=always"
    # auto
    if dataset_mb > auto_preload_mb:
        return False, (
            f"policy=auto: dataset {dataset_mb:.1f} MB > auto_preload_mb {auto_preload_mb:.1f} MB"
        )
    if over_ram:
        return False, (
            f"policy=auto: estimated {est_ram_mb:.0f} MB > 60% of available "
            f"{avail_ram_mb:.0f} MB (RAM safety veto)"
        )
    if avail_ram_mb is None:
        return False, (
            "policy=auto: unknown RAM budget; RAM-safety veto is unavailable "
            "(using HDF5 streaming)"
        )
    return True, (
        f"policy=auto: dataset {dataset_mb:.1f} MB <= auto_preload_mb {auto_preload_mb:.1f} MB"
    )


def _warn_batch_size_for_vram(device: torch.device, cfg: TrainConfig) -> None:
    """Advisory-only check: warn if batch_size looks large for the detected GPU.

    Sobolev training holds a second-order autograd graph (a = ∇U), so memory
    scales with batch_size, depth, and the number of multi-scale bands. This
    never changes the batch size — it only suggests using
    ``grad_accumulation_steps`` to keep the effective batch while fitting VRAM.
    """
    if device.type != "cuda":
        return
    try:
        props = torch.cuda.get_device_properties(device)
        total_gb = float(props.total_memory) / (1024.0 ** 3)
        gpu_name = props.name
    except Exception:
        return

    bs = int(cfg.batch_size)
    depth = int(getattr(cfg, "depth", 6))
    n_bands = int(getattr(cfg, "n_bands", 1))
    heavy = depth >= 6 and n_bands >= 3
    logger.info(f"CUDA device: {gpu_name} ({total_gb:.1f} GiB total VRAM)")

    suggestion = (
        "Sobolev autograd memory scales with batch_size×depth×n_bands. "
        "Prefer raising --grad-accumulation-steps over lowering the effective batch."
    )
    if total_gb <= 8.0:
        if bs > 4096:
            logger.warning(
                f"VRAM advisory: batch_size={bs} on a {total_gb:.1f} GiB GPU may OOM. "
                f"Consider --batch-size 4096 with --grad-accumulation-steps 2-4. {suggestion}"
            )
    elif total_gb <= 16.0:
        if heavy and bs >= 8192:
            logger.warning(
                f"VRAM advisory: batch_size={bs} with depth={depth}+n_bands={n_bands} on a "
                f"{total_gb:.1f} GiB GPU is borderline. If you hit OOM, use "
                f"--grad-accumulation-steps 2. {suggestion}"
            )
        elif bs > 16384:
            logger.warning(
                f"VRAM advisory: batch_size={bs} on a {total_gb:.1f} GiB GPU may be tight. {suggestion}"
            )
    else:
        if bs > 65536:
            logger.warning(
                f"VRAM advisory: batch_size={bs} is very large even for {total_gb:.1f} GiB. {suggestion}"
            )

def move_batch_to_device(
    x: torch.Tensor,
    u: torch.Tensor,
    a: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transfer a (x, u, a) batch to device with non_blocking for CUDA."""
    nb = device.type == "cuda"
    return (
        x.to(device, non_blocking=nb),
        u.to(device, non_blocking=nb),
        a.to(device, non_blocking=nb),
    )

def _laplacian_requested(cfg: TrainConfig) -> bool:
    """Return True when the config asks for any Laplacian work.

    The strong benchmark default (use_laplacian_regularization=True,
    laplacian_weight>0, laplacian_mode="diagnostic") requests Laplacian work.
    A config with all Laplacian flags/weights off/zero returns False, so normal
    training then does ZERO Laplacian computation — no in-batch penalty, no
    collocation diagnostics, no autograd overhead, no Laplacian term.

    ``laplacian_mode="off"`` is an explicit hard-disable. It wins over stale
    nonzero weights or compatibility flags so a config can turn all Laplacian
    work off without also cleaning every related field.
    """
    mode = str(getattr(cfg, "laplacian_mode", "diagnostic")).strip().lower()
    if mode == "off":
        return False
    return (
        bool(getattr(cfg, "use_laplacian_regularization", False))
        or mode == "train"
        or float(getattr(cfg, "collocation_laplacian_weight", 0.0)) > 0.0
        or float(getattr(cfg, "laplacian_weight", 0.0)) > 0.0
    )


@dataclass
class _EpochState:
    """Mutable per-epoch accumulators for :meth:`STLRPSTrainer.run_epoch`.

    One instance per epoch keeps the run_epoch decomposition honest: every
    helper reads/writes this state explicitly instead of sharing loop locals.
    """

    total_loss: float = 0.0
    total_opt_loss: float = 0.0
    total_u: float = 0.0
    total_a: float = 0.0
    total_grad_norm: float = 0.0
    total_dir: float = 0.0
    total_cossim: float = 0.0
    total_radial: float = 0.0
    total_cross: float = 0.0
    total_lap: float = 0.0
    total_mask_frac: float = 0.0
    total_a_norm_mean: float = 0.0
    total_angular_mean_deg: float = 0.0
    total_col_lap_diag: float = 0.0
    total_col_lap_train: float = 0.0
    col_lap_diag_count: int = 0
    col_lap_train_count: int = 0
    col_lap_attempt_count: int = 0
    col_lap_fail_count: int = 0
    col_lap_success_count: int = 0
    a_norm_max: float = 0.0
    n_batches: int = 0
    optimizer_steps_done: int = 0
    samples_done: int = 0
    last_stats: dict[str, float] = field(default_factory=dict)


class STLRPSTrainer:
    """
    Encapsulates the training state and execution logic.
    """
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        weights: GradNormWeights,
        device: torch.device,
        cfg: TrainConfig,
        collocation_r_min_m: float | None = None,
        collocation_r_max_m: float | None = None,
        ema_decay: float | None = None,
        ema_state_dict: Mapping[str, Any] | None = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.weights = weights
        self.device = device
        self.cfg = cfg
        self._band_diagnostic_epoch: int | None = None
        self.ema_decay = None if ema_decay is None else float(ema_decay)
        if self.ema_decay is not None and not (0.0 < self.ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1), got {self.ema_decay!r}")
        self.ema_state_dict: dict[str, torch.Tensor] | None = None
        if self.ema_decay is not None:
            self.ema_state_dict = {}
            current_state = model.state_dict()
            for name, current in current_state.items():
                previous = ema_state_dict.get(name) if ema_state_dict is not None else None
                if isinstance(previous, torch.Tensor) and previous.shape == current.shape:
                    self.ema_state_dict[name] = previous.detach().to(
                        device=current.device, dtype=current.dtype
                    ).clone()
                else:
                    self.ema_state_dict[name] = current.detach().clone()
            logger.info(
                "EMA enabled: decay=%.6f source=%s",
                self.ema_decay,
                "checkpoint" if ema_state_dict is not None else "initial_model",
            )
        self.curriculum = LossCurriculum(
            potential_only_epochs=cfg.potential_only_epochs,
            accel_ramp_epochs=cfg.accel_ramp_epochs,
            accel_min_factor=float(getattr(cfg, "accel_min_factor", 0.05)),
        )
        # Collocation Laplacian bounds
        self.collocation_r_min_m: float | None = collocation_r_min_m
        self.collocation_r_max_m: float | None = collocation_r_max_m
        # Whether any Laplacian work is requested at all. When False, the default,
        # all Laplacian paths are skipped (no autograd overhead).
        self.laplacian_requested: bool = _laplacian_requested(cfg)
        _lmode = str(getattr(cfg, "laplacian_mode", "diagnostic")).strip().lower()
        if _lmode not in ("off", "diagnostic", "train"):
            _lmode = "diagnostic"
        self.laplacian_mode: str = _lmode

        # bfloat16 instead of float16: SIREN sin(w0 · x) overflows fp16 mantissa.
        # bfloat16 has fp32 exponent range; disable AMP entirely if unavailable.
        # Laplacian regularization uses exact coordinate-basis trace HVPs in 3D
        # (and Hutchinson only for non-3D inputs), which remain AMP-compatible.
        self.use_amp = bool(cfg.amp and device.type == "cuda")
        if self.use_amp:
            if torch.cuda.is_bf16_supported():
                self._amp_dtype = torch.bfloat16
                # bfloat16 does NOT need GradScaler (same exponent as FP32)
                self.scaler_amp = None
            else:
                # GPU lacks bfloat16 → AMP is unsafe for PINN derivatives.
                logger.warning(
                    "GPU does not support bfloat16.  Disabling AMP to prevent "
                    "FP16 underflow/overflow in autograd derivatives (SIREN)."
                )
                self.use_amp = False
                self._amp_dtype = torch.float32
                self.scaler_amp = None
        else:
            self._amp_dtype = torch.float32
            self.scaler_amp = None

    def run_epoch(
        self,
        loader: DataLoader,
        is_train: bool,
        epoch: int,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        if is_train and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        self.model.train(is_train)
        if self.device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass
        accel_factor = self.curriculum.accel_factor(epoch) if is_train else 1.0

        state = _EpochState()

        # A fixed batch cache makes --overfit-batches a genuine repeated-batch
        # diagnostic. It is intentionally training-only; validation remains a
        # fixed sampler view of the configured validation split.
        overfit_n = int(getattr(self.cfg, "overfit_batches", 0) or 0)
        if is_train and overfit_n > 0:
            if not hasattr(self, "_overfit_batch_cache"):
                cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
                for cached_batch in loader:
                    cache.append(tuple(item.detach().cpu().clone() for item in cached_batch))
                    if len(cache) >= overfit_n:
                        break
                if not cache:
                    raise RuntimeError("--overfit-batches requested but the training loader is empty.")
                self._overfit_batch_cache = cache
                logger.info("Overfit sanity mode: caching %d fixed training batch(es).", len(cache))
            batch_source = iter(self._overfit_batch_cache)
            total_loader_batches = len(self._overfit_batch_cache)
        else:
            batch_source = iter(loader)
            total_loader_batches = len(loader)

        if is_train:
            lambda_dir_eff = _direction_loss_factor(epoch, self.cfg)
        else:
            lambda_dir_eff = float(max(0.0, getattr(self.cfg, "direction_loss_weight", 0.0)))
        grad_accum = max(1, int(getattr(self.cfg, "grad_accumulation_steps", 1)))

        phase = "train" if is_train else "val "
        log_every = int(max(0, self.cfg.log_every))
        total_batches_est = total_loader_batches
        if max_batches is not None:
            total_batches_est = min(total_batches_est, int(max_batches))

        # Auto logging frequency: ~10 progress updates per epoch, derived from the
        # batch count. The first and last batch are always logged below.
        if str(getattr(self.cfg, "log_every_mode", "fixed")).lower() == "auto":
            log_every = max(1, math.ceil(total_batches_est / 10))

        logger.info(f"Starting epoch {epoch + 1} {'train' if is_train else 'validation'} phase...")
        phase_t0 = time.perf_counter()

        with torch.set_grad_enabled(True):  # keep grads for val: a = ∇U
            for batch_idx, (xb, ub, ab) in enumerate(batch_source):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break

                xb, ub, ab = self._prepare_batch(xb, ub, ab)
                if batch_idx == 0:
                    self._log_band_diagnostics_once(xb, epoch)

                # Gradient accumulation bookkeeping
                is_last_batch = (
                    (batch_idx + 1 == total_loader_batches)
                    or (max_batches is not None and batch_idx + 1 >= int(max_batches))
                )
                is_accum_boundary = (batch_idx + 1) % grad_accum == 0 or is_last_batch

                if is_train and batch_idx % grad_accum == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                apply_lap = (
                    is_train
                    and self.laplacian_requested
                    and bool(self.cfg.use_laplacian_regularization)
                    and int(self.cfg.laplacian_every_n_batches) > 0
                    and (batch_idx % int(self.cfg.laplacian_every_n_batches) == 0)
                )

                # GradNorm weight update only happens on optimizer steps
                allow_weight_update = bool(accel_factor > 0.0 and is_accum_boundary)

                loss, stats = self._compute_loss(
                    xb,
                    ub,
                    ab,
                    is_train=is_train,
                    accel_factor=accel_factor,
                    allow_weight_update=allow_weight_update,
                    lambda_dir_eff=lambda_dir_eff,
                    apply_lap=apply_lap,
                )

                # Explosion guard: stop on NaN/Inf immediately to avoid corrupt checkpoints.
                if not self._validate_numerics(
                    stats, loss, phase=phase, epoch=epoch, batch=state.n_batches
                ):
                    # Return a sentinel so the caller can save a failure manifest.
                    return self._nan_sentinel(
                        state, accel_factor=accel_factor, lambda_dir_eff=lambda_dir_eff
                    )

                if is_train:
                    col_lap_loss_val, col_lap_scalar, col_lap_weight = (
                        self._collocation_laplacian_step(
                            state, epoch, is_accum_boundary=is_accum_boundary
                        )
                    )

                    # Average by the actual accumulation-group size. The final
                    # group may be shorter when max_batches is used.
                    group_start = batch_idx - (batch_idx % grad_accum)
                    group_size = min(total_batches_est, group_start + grad_accum) - group_start
                    scaled_loss = loss / float(max(1, group_size))

                    # Add collocation laplacian to loss in "train" mode.
                    # Semantics: ONE full-weight penalty per optimizer step.
                    # The penalty is computed once per accumulation group (at
                    # the boundary), so it enters UNSCALED — dividing it by
                    # group_size here would silently weaken the physics
                    # constraint by a factor of grad_accumulation_steps.
                    if (
                        col_lap_loss_val is not None
                        and self.laplacian_mode == "train"
                        and col_lap_weight > 0.0
                    ):
                        scaled_loss = scaled_loss + col_lap_weight * col_lap_loss_val
                        # NaN/Inf guard for collocation Laplacian contribution
                        _cl_check = float(scaled_loss.item())
                        if math.isnan(_cl_check) or math.isinf(_cl_check):
                            logger.error(
                                f"[train] NaN/Inf after adding collocation Laplacian at epoch={epoch+1} "
                                f"batch={state.n_batches}. Saving failure manifest and stopping."
                            )
                            self._write_failure_manifest(
                                epoch=epoch,
                                batch=state.n_batches,
                                reason="nan_loss_after_collocation_laplacian",
                                extra={"collocation_laplacian_scalar": col_lap_scalar},
                            )
                            return self._nan_sentinel(
                                state,
                                accel_factor=accel_factor,
                                lambda_dir_eff=lambda_dir_eff,
                                collocation_applied=True,
                                collocation_train_scalar=col_lap_scalar,
                            )

                    self._backward_step(
                        state,
                        scaled_loss,
                        is_accum_boundary=is_accum_boundary,
                        epoch=epoch,
                    )

                self._update_metrics(state, stats, xb, ab)

                self._log_progress(
                    state,
                    phase=phase,
                    epoch=epoch,
                    log_every=log_every,
                    total_batches_est=total_batches_est,
                    lambda_dir_eff=lambda_dir_eff,
                    phase_t0=phase_t0,
                )

        return self._write_history_summary(
            state,
            phase=phase,
            epoch=epoch,
            accel_factor=accel_factor,
            lambda_dir_eff=lambda_dir_eff,
            phase_time=time.perf_counter() - phase_t0,
        )

    # ------------------------------------------------------------------
    # run_epoch decomposition (R24): prepare_batch / compute_loss /
    # backward_step / validate_numerics / update_metrics / log_progress /
    # write_history. Each helper reads/writes the explicit _EpochState.
    # ------------------------------------------------------------------

    def _prepare_batch(
        self, xb: torch.Tensor, ub: torch.Tensor, ab: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move one loader batch to the training device."""
        return move_batch_to_device(xb, ub, ab, self.device)

    def _log_band_diagnostics_once(self, x_phys: torch.Tensor, epoch: int) -> None:
        """Log multi-band contribution diagnostics once per epoch."""

        if self._band_diagnostic_epoch == int(epoch):
            return
        if int(getattr(self.cfg, "n_bands", 1)) <= 1:
            self._band_diagnostic_epoch = int(epoch)
            return
        diagnostic_fn = getattr(self.model, "band_diagnostics", None)
        if not callable(diagnostic_fn):
            self._band_diagnostic_epoch = int(epoch)
            return
        try:
            with torch.no_grad():
                kind, values = diagnostic_fn(self.loss_fn.scale_x(x_phys))
            values_cpu = [float(value) for value in values.detach().cpu().tolist()]
            logger.info(
                "[bands] epoch=%d %s=%s w0_bands=%s",
                int(epoch) + 1,
                kind,
                [f"{value:.3e}" for value in values_cpu],
                getattr(getattr(self.model, "backbone", None), "w0_bands", None),
            )
        except Exception as exc:  # pragma: no cover - diagnostic must not stop training
            logger.warning("[bands] diagnostic failed at epoch=%d: %s", int(epoch) + 1, exc)
        self._band_diagnostic_epoch = int(epoch)

    def _compute_loss(
        self,
        xb: torch.Tensor,
        ub: torch.Tensor,
        ab: torch.Tensor,
        *,
        is_train: bool,
        accel_factor: float,
        allow_weight_update: bool,
        lambda_dir_eff: float,
        apply_lap: bool,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Evaluate the Sobolev loss under the configured autocast policy."""
        with torch.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            # Validation is a checkpoint-selection input and is deliberately
            # evaluated in FP32 even when training uses bf16 AMP.
            enabled=bool(self.use_amp and is_train),
        ):
            return self.loss_fn(
                self.model,
                xb,
                ub,
                ab,
                self.weights,
                is_train=is_train,
                accel_factor=accel_factor,
                allow_dynamic_weight_update=allow_weight_update,
                direction_lambda=lambda_dir_eff,
                direction_floor_abs=self.cfg.direction_loss_floor_abs,
                use_altitude_balanced_loss=bool(self.cfg.use_altitude_balanced_loss),
                altitude_bin_width_km=float(self.cfg.altitude_bin_width_km),
                altitude_min_km=float(self.cfg.altitude_min_km),
                altitude_max_km=float(self.cfg.altitude_max_km),
                use_radial_cross_loss=bool(self.cfg.use_radial_cross_loss),
                radial_lambda=float(self.cfg.radial_loss_weight),
                cross_lambda=float(self.cfg.cross_loss_weight),
                apply_laplacian=bool(apply_lap),
                laplacian_lambda=float(self.cfg.laplacian_weight),
                laplacian_subset_size=int(self.cfg.laplacian_subset_size),
                laplacian_n_hutchinson=int(getattr(self.cfg, "n_hutchinson_samples", 4)),
                laplacian_mode=self.laplacian_mode,
            )

    def _validate_numerics(
        self,
        stats: dict[str, float],
        loss: torch.Tensor,
        *,
        phase: str,
        epoch: int,
        batch: int,
    ) -> bool:
        """Return False (and log) when the optimizer loss went NaN/Inf."""
        loss_check = float(stats.get("loss_opt", loss.item()))
        if math.isnan(loss_check) or math.isinf(loss_check):
            logger.error(
                f"[{phase}] NaN/Inf loss detected at epoch={epoch+1} batch={batch}. "
                "Possible derivative instability. "
                "Suggestions: lower lr, ensure accel_min_factor>0, lower w0, increase accel_ramp_epochs. "
                "Stopping epoch early."
            )
            return False
        return True

    def _nan_sentinel(
        self,
        state: _EpochState,
        *,
        accel_factor: float,
        lambda_dir_eff: float,
        collocation_applied: bool = False,
        collocation_train_scalar: float = 0.0,
    ) -> dict[str, float]:
        """Sentinel epoch result after a NaN/Inf abort (caller saves manifest)."""
        return {
            "loss": float("nan"), "objective_loss": float("nan"),
            "mse_u": float("nan"), "mse_a": float("nan"),
            "loss_dir": 0.0, "cossim_mean": 0.0,
            "loss_radial": 0.0, "loss_cross": 0.0, "loss_laplacian": 0.0,
            "loss_laplacian_diag": 0.0,
            "loss_laplacian_train": float(collocation_train_scalar),
            "lambda_laplacian_eff": float(
                getattr(self.cfg, "collocation_laplacian_weight", 0.0)
            ),
            "collocation_laplacian_applied": bool(collocation_applied),
            "lambda_dir_eff": lambda_dir_eff,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "w_u": float(state.last_stats.get("w_u", self.cfg.w_u)),
            "w_a": float(state.last_stats.get("w_a", self.cfg.w_a)),
            "w_a_raw": float(state.last_stats.get("w_a_raw", self.cfg.w_a)),
            "accel_factor": float(accel_factor),
            "grad_norm": 0.0,
            "nan_detected": True,
            "val_base_loss": float("nan"),
            "val_physics_loss": float("nan"),
            "val_total_loss": float("nan"),
            "train_base_loss": float("nan"),
            "train_physics_loss": float("nan"),
            "optimizer_steps": int(state.optimizer_steps_done),
            "samples_seen": int(state.samples_done),
        }

    def _write_failure_manifest(
        self, *, epoch: int, batch: int, reason: str, extra: dict[str, Any]
    ) -> None:
        try:
            fm_path = Path(self.cfg.out) / "failure_manifest.json"
            atomic_write_json(
                fm_path,
                {"epoch": epoch, "batch": batch, "reason": reason, **extra},
            )
        except Exception:  # R29b-justified: the abort itself must still propagate
            pass

    def _collocation_laplacian_step(
        self, state: _EpochState, epoch: int, *, is_accum_boundary: bool
    ) -> tuple[torch.Tensor | None, float, float]:
        """Collocation Laplacian, computed BEFORE backward so it can be added
        to the loss in "train" mode or logged only in "diagnostic" mode.

        Returns ``(loss_tensor_or_None, scalar_value, configured_weight)``.
        """
        col_lap_weight = float(getattr(self.cfg, "collocation_laplacian_weight", 0.0))
        col_lap_every = max(1, int(getattr(self.cfg, "collocation_laplacian_every", 25)))
        col_lap_active = (
            self.laplacian_requested
            and self.laplacian_mode in ("diagnostic", "train")
            and self.collocation_r_min_m is not None
            and self.collocation_r_max_m is not None
            and bool(is_accum_boundary)
            and state.optimizer_steps_done % col_lap_every == 0
        )
        col_lap_loss_val: torch.Tensor | None = None
        col_lap_scalar = 0.0
        if not col_lap_active:
            return col_lap_loss_val, col_lap_scalar, col_lap_weight

        n_pts = max(1, int(getattr(self.cfg, "collocation_laplacian_samples",
                                   getattr(self.cfg, "laplacian_subset_size", 512))))
        n_hutch = max(1, int(getattr(self.cfg, "collocation_laplacian_hutchinson_samples",
                                     getattr(self.cfg, "n_hutchinson_samples", 4))))
        state.col_lap_attempt_count += 1
        try:
            # Always use the ScalerPack from loss_fn for consistent scaling
            cl_loss = collocation_laplacian_loss(
                self.model, self.loss_fn,
                r_min_m=float(self.collocation_r_min_m),
                r_max_m=float(self.collocation_r_max_m),
                n_points=n_pts,
                device=self.device,
                dtype=DTYPE,
                n_hutchinson=n_hutch,
                mode=self.laplacian_mode,
            )
            col_lap_scalar = float(cl_loss.detach().item())
            if math.isfinite(col_lap_scalar):
                state.col_lap_success_count += 1
                if self.laplacian_mode == "train":
                    state.total_col_lap_train += col_lap_scalar
                    state.col_lap_train_count += 1
                    col_lap_loss_val = cl_loss
                else:  # diagnostic
                    state.total_col_lap_diag += col_lap_scalar
                    state.col_lap_diag_count += 1
            else:
                state.col_lap_fail_count += 1
                if self.laplacian_mode == "train":
                    raise RuntimeError(
                        f"collocation_laplacian_loss returned non-finite "
                        f"value {col_lap_scalar} in train mode."
                    )
                logger.warning(
                    "[train] collocation_laplacian_loss non-finite "
                    f"({col_lap_scalar}); skipped this step (diagnostic mode)."
                )
        except Exception as col_e:
            state.col_lap_fail_count += 1
            # In train mode the Laplacian is part of the objective; a
            # silent skip would disable the physics constraint while the
            # logs/metrics still claim it is active. Fail loudly instead.
            if self.laplacian_mode == "train":
                raise RuntimeError(
                    "collocation_laplacian_loss failed in train mode "
                    f"(epoch={epoch+1}, batch={state.n_batches}): {col_e}. "
                    "The physics constraint cannot be silently dropped; "
                    "fix the cause or switch laplacian_mode to 'diagnostic'/'off'."
                ) from col_e
            logger.warning(f"[train] collocation_laplacian_loss failed: {col_e}")
        return col_lap_loss_val, col_lap_scalar, col_lap_weight

    def _backward_step(
        self,
        state: _EpochState,
        scaled_loss: torch.Tensor,
        *,
        is_accum_boundary: bool,
        epoch: int,
    ) -> None:
        """Backward + (on accumulation boundaries) clip, optimizer step, and
        gradient-norm bookkeeping. AMP and non-AMP share one logging contract."""

        def _clip_grad_norm() -> torch.Tensor:
            if self.cfg.max_grad_norm > 0:
                return torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.cfg.max_grad_norm
                )
            return torch.tensor(0.0, device=self.device)

        def _record_grad_norm(grad_norm: torch.Tensor) -> None:
            state.optimizer_steps_done += 1
            state.total_grad_norm += float(grad_norm)
            if float(grad_norm) > 50.0:
                logger.warning(
                    f"[train] grad_norm={float(grad_norm):.1f} > 50 at epoch={epoch+1} "
                    f"batch={state.n_batches}: possible derivative explosion. "
                    "Consider lower lr or max_grad_norm."
                )

        def _update_ema() -> None:
            if self.ema_state_dict is None or self.ema_decay is None:
                return
            one_minus = 1.0 - self.ema_decay
            for name, current in self.model.state_dict().items():
                shadow = self.ema_state_dict[name]
                if torch.is_floating_point(current):
                    shadow.mul_(self.ema_decay).add_(current.detach(), alpha=one_minus)
                else:
                    shadow.copy_(current.detach())

        if self.use_amp and self.scaler_amp is not None:
            self.scaler_amp.scale(scaled_loss).backward()
            if is_accum_boundary:
                self.scaler_amp.unscale_(self.optimizer)
                grad_norm = _clip_grad_norm()
                self.scaler_amp.step(self.optimizer)
                self.scaler_amp.update()
                _record_grad_norm(grad_norm)
                _update_ema()
        else:
            scaled_loss.backward()
            if is_accum_boundary:
                grad_norm = _clip_grad_norm()
                self.optimizer.step()
                _record_grad_norm(grad_norm)
                _update_ema()

    def _update_metrics(
        self,
        state: _EpochState,
        stats: dict[str, float],
        xb: torch.Tensor,
        ab: torch.Tensor,
    ) -> None:
        """Accumulate the per-batch statistics into the epoch state."""
        state.samples_done += int(xb.shape[0])
        state.total_loss += float(stats["loss_ref"])
        state.total_opt_loss += float(stats["loss_opt"])
        state.total_u += float(stats["mse_u"])
        state.total_a += float(stats["mse_a"])
        state.total_dir += float(stats.get("loss_dir", 0.0))
        state.total_cossim += float(stats.get("cossim_mean", 1.0))
        state.total_angular_mean_deg += float(stats.get("angular_mean_deg", 0.0))
        state.total_mask_frac += float(stats.get("mask_frac", 0.0))
        state.total_radial += float(stats.get("loss_radial", 0.0))
        state.total_cross += float(stats.get("loss_cross", 0.0))
        state.total_lap += float(stats.get("loss_laplacian", 0.0))
        a_norm_b = float(ab.detach().norm(dim=-1).mean().item())
        state.total_a_norm_mean += a_norm_b
        state.a_norm_max = max(state.a_norm_max, a_norm_b)
        state.n_batches += 1
        state.last_stats = stats

    def _log_progress(
        self,
        state: _EpochState,
        *,
        phase: str,
        epoch: int,
        log_every: int,
        total_batches_est: int,
        lambda_dir_eff: float,
        phase_t0: float,
    ) -> None:
        """Periodic in-epoch progress line (first, every Nth, and last batch)."""
        n_batches = state.n_batches
        if not (
            log_every > 0
            and (
                n_batches == 1
                or n_batches % log_every == 0
                or n_batches == total_batches_est
            )
        ):
            return
        elapsed = time.perf_counter() - phase_t0
        spb = elapsed / max(1, n_batches)
        eta = max(0.0, spb * (total_batches_est - n_batches))
        sps = state.samples_done / max(elapsed, 1e-9)
        cur_lr = float(self.optimizer.param_groups[0]["lr"])
        mem_str = _cuda_memory_string(self.device)
        # loss_opt = optimizer loss (uses accel_factor); loss_ref = full diagnostic loss.
        # Per-batch extras (radial/cross/lap/alt-balance) are only surfaced in the
        # end-of-phase summary via format_batch_summary's fixed field set.
        logger.info(
            format_batch_summary(
                phase=phase.strip(),
                epoch=epoch + 1,
                batch=n_batches,
                total_batches=total_batches_est,
                loss_opt=state.total_opt_loss / n_batches,
                loss_ref=state.total_loss / n_batches,
                loss_u=state.total_u / n_batches,
                loss_a=state.total_a / n_batches,
                loss_dir=(state.total_dir / n_batches if lambda_dir_eff > 0.0 else None),
                lr=cur_lr,
                eta_s=eta,
                samples_per_s=sps,
                memory=mem_str,
            )
        )

    def _write_history_summary(
        self,
        state: _EpochState,
        *,
        phase: str,
        epoch: int,
        accel_factor: float,
        lambda_dir_eff: float,
        phase_time: float,
    ) -> dict[str, float]:
        """End-of-phase summary log + the epoch history record (return value)."""
        n_safe = max(1, state.n_batches)
        dir_summary = (
            f" dir={state.total_dir/n_safe:.3e} cossim={state.total_cossim/n_safe:.4f}"
            f" ang={state.total_angular_mean_deg/n_safe:.2f}deg"
            f" mask_frac={state.total_mask_frac/n_safe:.2f} lam_dir={lambda_dir_eff:.3e}"
            if lambda_dir_eff > 0.0 else ""
        )
        extra_summary = ""
        if bool(self.cfg.use_radial_cross_loss):
            extra_summary += (
                f" radial={state.total_radial/n_safe:.3e} cross={state.total_cross/n_safe:.3e}"
            )
        if bool(self.cfg.use_laplacian_regularization):
            extra_summary += f" lap={state.total_lap/n_safe:.3e}"
        if bool(self.cfg.use_altitude_balanced_loss):
            extra_summary += " alt-balance=on"
        # loss_ref is always the full reference (val uses full weight; train uses accel_factor)
        logger.info(
            f"[{phase}] epoch={epoch+1} done: {state.samples_done:,} samples in "
            f"{_format_seconds(phase_time)}"
            f" ({phase_time / n_safe * 1000:.1f}ms/batch)"
            f" loss_opt={state.total_opt_loss/n_safe:.5e} loss_ref={state.total_loss/n_safe:.5e}"
            f" U={state.total_u/n_safe:.3e} a={state.total_a/n_safe:.3e}"
            f" a_norm_mean={state.total_a_norm_mean/n_safe:.3e} a_norm_max={state.a_norm_max:.3e}"
            f" accel_f={accel_factor:.3f}{dir_summary}{extra_summary}"
        )

        n_col_diag = max(1, state.col_lap_diag_count)
        n_col_train = max(1, state.col_lap_train_count)
        col_lap_diag_avg = (
            state.total_col_lap_diag / n_col_diag if state.col_lap_diag_count > 0 else 0.0
        )
        col_lap_train_avg = (
            state.total_col_lap_train / n_col_train if state.col_lap_train_count > 0 else 0.0
        )
        col_lap_applied = state.col_lap_diag_count > 0 or state.col_lap_train_count > 0
        col_lap_weight_eff = float(getattr(self.cfg, "collocation_laplacian_weight", 0.0))
        physics_loss = (
            state.total_dir + state.total_radial + state.total_cross
            + state.total_lap
        ) / n_safe
        physics_loss += col_lap_train_avg
        return {
            "loss": state.total_loss / n_safe,
            "objective_loss": state.total_opt_loss / n_safe,
            "mse_u": state.total_u / n_safe,
            "mse_a": state.total_a / n_safe,
            "loss_dir": state.total_dir / n_safe,
            "cossim_mean": state.total_cossim / n_safe,
            "angular_mean_deg": state.total_angular_mean_deg / n_safe,
            "mask_frac": state.total_mask_frac / n_safe,
            "a_norm_mean": state.total_a_norm_mean / n_safe,
            "a_norm_max": state.a_norm_max,
            "loss_radial": state.total_radial / n_safe,
            "loss_cross": state.total_cross / n_safe,
            "loss_laplacian": state.total_lap / n_safe,
            "loss_laplacian_diag": col_lap_diag_avg,
            "loss_laplacian_train": col_lap_train_avg,
            "lambda_laplacian_eff": col_lap_weight_eff,
            "collocation_laplacian_applied": col_lap_applied,
            "collocation_laplacian_attempt_count": int(state.col_lap_attempt_count),
            "collocation_laplacian_success_count": int(state.col_lap_success_count),
            "collocation_laplacian_fail_count": int(state.col_lap_fail_count),
            "lambda_dir_eff": lambda_dir_eff,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "w_u": float(state.last_stats.get("w_u", self.cfg.w_u)),
            "w_a": float(state.last_stats.get("w_a", self.cfg.w_a)),
            "w_a_raw": float(state.last_stats.get("w_a_raw", self.cfg.w_a)),
            "accel_factor": float(state.last_stats.get("accel_factor", accel_factor)),
            "grad_norm": state.total_grad_norm / max(1, state.optimizer_steps_done),
            "val_base_loss": (state.total_u + state.total_a) / n_safe,  # U + accel MSE only
            "val_physics_loss": physics_loss,
            "val_total_loss": state.total_loss / n_safe,   # alias for "loss"
            "train_base_loss": (state.total_u + state.total_a) / n_safe,
            "train_physics_loss": physics_loss,
            "optimizer_steps": int(state.optimizer_steps_done),
            "samples_seen": int(state.samples_done),
        }

def _lr_multiplier_for_epoch(
    epoch: int,
    *,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
    t_max: int | None,
) -> float:
    """
    Warm up linearly, then decay with a cosine schedule to ``min_lr_ratio``.
    """

    epoch_i = max(0, int(epoch))
    total_i = max(1, int(total_epochs))
    warmup_i = max(0, int(warmup_epochs))
    min_ratio = float(min(max(float(min_lr_ratio), 0.0), 1.0))

    if warmup_i > 0 and epoch_i < warmup_i:
        return float((epoch_i + 1) / warmup_i)

    decay_total = int(t_max) if t_max is not None else total_i
    decay_total = max(warmup_i + 1, decay_total)
    denom = max(1, decay_total - warmup_i - 1)
    progress = min(1.0, max(0.0, (epoch_i - warmup_i) / float(denom)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_ratio + (1.0 - min_ratio) * cosine)

def _apply_lr_multiplier(optimizer: torch.optim.Optimizer, multiplier: float) -> None:
    for group in optimizer.param_groups:
        base_lr = float(group.setdefault("initial_lr", group["lr"]))
        group["lr"] = base_lr * float(multiplier)

def _write_training_history_csv(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    extra_fields = sorted({str(k) for row in history for k in row.keys()} - set(HISTORY_FIELDNAMES))
    fieldnames = list(HISTORY_FIELDNAMES) + extra_fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _append_history_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _dataset_meta_snapshot(
    meta: DatasetMeta,
    *,
    dataset_name: str,
    data_path: Path | None,
    train_data_path: Path | None,
    val_data_path: Path | None,
    test_data_path: str | None,
    ood_data_path: str | None,
    target_mode: str,
    central_body: str,
    resolved_mu_si: float,
    resolved_r_ref_m: float,
) -> dict[str, Any]:
    _hash_path = data_path if data_path is not None and Path(data_path).exists() else train_data_path
    if _hash_path is not None and Path(_hash_path).exists():
        size_gb = Path(_hash_path).stat().st_size / float(1024 ** 3)
        logger.info("Hashing dataset for run provenance: %s (%.3f GiB)", _hash_path, size_gb)
    snapshot = {
        "schema_version": 1,
        "dataset_sha256": (
            compute_file_sha256(data_path)
            if data_path is not None and Path(data_path).exists()
            else (
                compute_file_sha256(train_data_path)
                if train_data_path is not None and Path(train_data_path).exists()
                else None
            )
        ),
        "dataset_name": str(dataset_name),
        "data_path": (str(data_path) if data_path is not None else None),
        "train_data_path": (str(train_data_path) if train_data_path is not None else None),
        "val_data_path": (str(val_data_path) if val_data_path is not None else None),
        "test_data_path": (str(test_data_path) if test_data_path else None),
        "ood_data_path": (str(ood_data_path) if ood_data_path else None),
        "target_mode": str(target_mode),
        "central_body": str(central_body),
        "mu_si": float(resolved_mu_si),
        "r_ref_m": float(resolved_r_ref_m),
        "unit_system": meta.unit_system,
        "requested_degree": meta.requested_degree,
        "degree_min": meta.degree_min,
        "degree_max": meta.degree_max,
        "alt_min_km": meta.alt_min_km,
        "alt_max_km": meta.alt_max_km,
        "columns": meta.columns,
        "a_sign_convention": meta.a_sign_convention,
        "derivative_convention_version": meta.derivative_convention_version,
        "DU_m": meta.DU_m,
        "TU_s": meta.TU_s,
        "VU_m_s": meta.VU_m_s,
        "include_potential": meta.include_potential,
        "gravity_model_path": meta.gravity_model_path,
        "cloud_config": meta.cloud_config,
        "raw_attrs": dict(meta.raw_attrs),
    }
    snapshot["dataset_contract"] = build_dataset_contract(
        meta,
        data_path=(data_path or train_data_path or Path(".")),
        n_samples=None,
        dataset_sha256=snapshot["dataset_sha256"],
    )
    return snapshot

def _save_training_plots(history: list[dict[str, float]], outdir: Path) -> None:
    if not history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping training-history plots.")
        return

    outdir = Path(outdir) / "plots" / "training"
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        epochs = np.asarray([int(item["epoch"]) + 1 for item in history], dtype=float)
    except Exception as exc:
        logger.warning(f"training-history plots skipped because epoch values could not be read: {exc}")
        return

    def _series(key: str, default: float = float("nan")) -> list[float]:
        values: list[float] = []
        for item in history:
            try:
                value = float(item.get(key, default))
            except Exception:
                value = float("nan")
            values.append(value if math.isfinite(value) else float("nan"))
        return values

    def _robust_ylim(series_values: list[np.ndarray], *, logy: bool) -> tuple[float, float] | None:
        valid: list[np.ndarray] = []
        for arr in series_values:
            finite = arr[np.isfinite(arr)]
            if logy:
                finite = finite[finite > 0.0]
            if finite.size:
                valid.append(finite)
        if not valid:
            return None
        merged = np.concatenate(valid)
        if merged.size == 1:
            value = float(merged[0])
            if logy:
                return max(value / 2.0, 1e-30), value * 2.0
            margin = abs(value) * 0.1 + 1e-12
            return value - margin, value + margin
        lo, hi = np.nanpercentile(merged, [1.0, 99.0])
        lo = float(lo)
        hi = float(hi)
        if not math.isfinite(lo) or not math.isfinite(hi):
            return None
        if logy:
            lo = max(lo, 1e-30)
            hi = max(hi, lo * 1.01)
            return lo / 1.25, hi * 1.25
        if hi <= lo:
            margin = abs(hi) * 0.1 + 1e-12
            return lo - margin, hi + margin
        margin = (hi - lo) * 0.08
        return lo - margin, hi + margin

    def _plot_series(
        path: Path,
        title: str,
        y_label: str,
        series: list[tuple[str, list[float]]],
        *,
        logy: bool = False,
        y_bounds: tuple[float, float] | None = None,
    ) -> None:
        try:
            fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
            plotted_arrays: list[np.ndarray] = []
            for label, values in series:
                arr = np.asarray(values, dtype=float)
                arr[~np.isfinite(arr)] = np.nan
                if logy:
                    arr[arr <= 0.0] = np.nan
                mask = np.isfinite(arr)
                if not np.any(mask):
                    continue
                plotted_arrays.append(arr)
                ax.plot(epochs[mask], arr[mask], label=label, linewidth=2.0)
            if not plotted_arrays:
                plt.close(fig)
                return
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(y_label)
            if logy:
                ax.set_yscale("log")
            ylim = y_bounds or _robust_ylim(plotted_arrays, logy=logy)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(loc="best")
            fig.savefig(path, dpi=180)
            plt.close(fig)
        except Exception as exc:
            logger.warning(f"training plot {path.name} could not be written: {exc}")

    _plot_series(
        outdir / "loss_total.png",
        "Total Loss",
        "Loss",
        [
            ("train_total", _series("train_loss_total")),
            ("val_total", _series("val_loss_total")),
        ],
        logy=True,
    )
    _plot_series(
        outdir / "loss_U.png",
        "Potential Loss",
        "MSE",
        [
            ("train_U", _series("train_loss_u")),
            ("val_U", _series("val_loss_u")),
        ],
        logy=True,
    )
    _plot_series(
        outdir / "loss_a.png",
        "Acceleration Loss",
        "MSE",
        [
            ("train_a", _series("train_loss_a")),
            ("val_a", _series("val_loss_a")),
        ],
        logy=True,
    )
    _plot_series(
        outdir / "lr_schedule.png",
        "Learning Rate Schedule",
        "Learning Rate",
        [("lr", _series("lr"))],
        logy=True,
    )
    _plot_series(
        outdir / "weights.png",
        "Sobolev Loss Weights",
        "Weight",
        [
            ("w_U", _series("w_u")),
            ("w_a_raw", _series("w_a_raw")),
            ("w_a_eff", _series("w_a_eff")),
        ],
        logy=False,
    )
    # Only plot direction loss if it was ever non-zero
    if any(
        math.isfinite(v) and v > 0.0
        for v in _series("train_loss_dir", 0.0)
    ):
        _plot_series(
            outdir / "loss_dir.png",
            "Direction Loss (1 - cos_sim)",
            "Loss",
            [
                ("train_direction", _series("train_loss_dir", 0.0)),
                ("val_direction", _series("val_loss_dir", 0.0)),
            ],
            logy=True,
        )
        _plot_series(
            outdir / "cossim.png",
            "Mean Cosine Similarity (a_pred vs a_true)",
            "cos_sim",
            [
                ("train_cos_sim", _series("train_mean_cossim", 1.0)),
                ("val_cos_sim", _series("val_mean_cossim", 1.0)),
            ],
            logy=False,
            y_bounds=(-1.05, 1.05),
        )

def _log_dataset_and_model_summary(N, _effective_target, bytes_est, cfg, data_path, dataset_body_name, dset_name, independent_val, meta, n_train, n_val, resolved_mu_si, resolved_r_ref_m, train_data_path, val_data_path):
    logger.info("=== Dataset ===")
    if independent_val:
        logger.info(f"Train file: {train_data_path.name} ({n_train:,} samples)")
        logger.info(f"Val file  : {val_data_path.name} ({n_val:,} samples)")
        if cfg.test_data:
            logger.info(f"Test file : {cfg.test_data}")
        if cfg.ood_data:
            logger.info(f"OOD file  : {cfg.ood_data}")
    _sm = getattr(cfg, "suite_manifest", None)
    if _sm:
        logger.info(f"Suite manifest: {_sm}")
    else:
        logger.info(f"File: {data_path.name}")
    logger.info(f"Target Dataset: {dset_name} | Total: [{N:,}, 7] | Size: {_human_bytes(bytes_est)}")
    logger.info(f"Train/val split: {n_train:,} / {n_val:,}")
    logger.info("=== Physics Metadata (auto-synced from HDF5) ===")
    logger.info(f"central_body : {dataset_body_name}")
    logger.info(f"unit_system  : {meta.unit_system}")
    logger.info(f"mu_si        : {resolved_mu_si}  |  r_ref_m : {resolved_r_ref_m}")
    logger.info(f"degree_max   : {meta.requested_degree}  |  degree_min : {meta.degree_min}")
    logger.info(f"target_mode  : {meta.target_mode or 'unknown (inferred: ' + _effective_target + ')'}")
    logger.info(f"columns      : {meta.columns or 'unknown'}")
    logger.info(f"a_sign_conv  : {meta.a_sign_convention or 'unknown'}")
    _dcv = getattr(meta, "derivative_convention_version", None)
    if _dcv is None:
        logger.warning(
            "derivative_convention_version: MISSING - dataset may have been generated before "
            "the dP_dphi sign fix. Latitude acceleration labels could be sign-flipped. "
            "Regenerate with the corrected spatial_cloud_generator.py."
        )
    else:
        logger.info(f"deriv_conv   : {_dcv}")
    if meta.alt_min_km is not None and meta.alt_max_km is not None:
        logger.info(f"alt range    : [{meta.alt_min_km}, {meta.alt_max_km}] km")
    logger.info(f"Conversion factors (DU/TU/VU): {meta.DU_m} / {meta.TU_s} / {meta.VU_m_s}")
    logger.info("=== Model ===")
    _n_bands_log = getattr(cfg, "n_bands", 1)
    _use_res_log = getattr(cfg, "use_residual_blocks", False)
    _grad_acc_log = getattr(cfg, "grad_accumulation_steps", 1)
    logger.info(f"{'model.activation':24s}: {cfg.activation}")
    logger.info(f"{'model.hidden':24s}: {cfg.hidden}")
    logger.info(f"{'model.depth':24s}: {cfg.depth}")
    logger.info(f"{'model.preset':24s}: {getattr(cfg, 'model_preset', 'custom')}")
    logger.info(f"{'run.preset':24s}: {getattr(cfg, 'run_preset', 'custom')}")
    logger.info(f"{'data.split_policy':24s}: {getattr(cfg, 'split_policy', 'seeded_random')}")
    logger.info(f"{'model.n_bands':24s}: {_n_bands_log}")
    logger.info(f"{'model.w0_first':24s}: {cfg.w0_first}")
    logger.info(f"{'model.w0_hidden':24s}: {cfg.w0_hidden}")
    logger.info(f"{'model.w0_bands':24s}: {getattr(cfg, 'w0_bands', None)}")
    logger.info(f"{'model.residual_blocks':24s}: {_use_res_log}")
    logger.info(f"{'grad_accum':24s}: {_grad_acc_log}")

def _log_training_curriculum(_accel_min_fac, cfg):
    logger.info("=== Training Curriculum ===")
    logger.info(
        f"potential_only_epochs={cfg.potential_only_epochs} | "
        f"accel_ramp_epochs={cfg.accel_ramp_epochs} | "
        f"accel_min_factor={_accel_min_fac}"
    )
    # Derivative training note: acceleration is ∇U, so it must be constrained from epoch 0.
    if cfg.potential_only_epochs > 0:
        logger.warning(
            "potential_only_epochs > 0 detected. "
            "SIREN can fit dU while grad(dU) drifts because acceleration is computed via autograd. "
            f"accel_min_factor={_accel_min_fac} keeps a floor to limit drift. "
            "Set accel_min_factor=0.0 only if you explicitly want pure potential-only behaviour."
        )
    if _accel_min_fac == 0.0:
        logger.info("  Derivative training note: accel_min_factor=0.0 (pure potential-only during warm-up).")
    else:
        logger.info(
            f"  Derivative training note: acceleration is always active (floor={_accel_min_fac}). "
            "This prevents grad(dU) from drifting during curriculum warm-up."
        )

    if cfg.use_altitude_balanced_loss:
        logger.info(f"  Altitude-Balanced Loss: ON (bins={cfg.altitude_bin_width_km}km)")
    if cfg.use_radial_cross_loss:
        logger.info(f"  Radial/Cross Loss: ON (radial_w={cfg.radial_loss_weight}, cross_w={cfg.cross_loss_weight})")
    if cfg.use_laplacian_regularization:
        _lap_mode_log = str(getattr(cfg, "laplacian_mode", "diagnostic")).strip().lower()
        if _lap_mode_log == "train":
            logger.info(
                f"  In-batch Laplacian Reg: ON, mode=train (gradient backpropagates) "
                f"(w={cfg.laplacian_weight}, every={cfg.laplacian_every_n_batches})"
            )
        else:
            logger.info(
                f"  In-batch Laplacian Reg: ON, mode={_lap_mode_log} (DIAGNOSTIC ONLY - logged, "
                f"NOT backpropagated). For a trainable physics constraint set --laplacian-mode train "
                f"(collocation Laplacian is the preferred trainable regulariser)."
            )
    logger.info(f"  Direction Loss: weight={cfg.direction_loss_weight}, start={cfg.direction_loss_start_epoch}, ramp={cfg.direction_loss_ramp_epochs}")
    # NOTE: best-checkpoint-metric logging moved below, after _best_metric_canonical
    # and checkpoint_selection are defined (they were referenced here before
    # assignment, which made train() raise UnboundLocalError on every run).

    # Fail fast on invalid architecture combination
    if cfg.activation.lower() == "sine" and cfg.use_fourier:
        raise ValueError(
            "activation='sine' (SIREN) and use_fourier=True are mutually exclusive. "
            "Stacking RFF on a SIREN creates a sin-of-sin composition that causes "
            "catastrophic out-of-distribution overfitting. "
            "Use one of:\n"
            "  (1) activation='silu'/'tanh' + use_fourier=True\n"
            "  (2) activation='sine' + use_fourier=False  (recommended default)"
        )

def _log_data_loading_policy(N, _avail_ram_mb, _est_ram_mb, _policy, _preload_reason, cfg, dataset_mb, should_preload):
    logger.info("=== Data Loading Policy ===")
    logger.info(f"  dataset estimated size : {dataset_mb:.1f} MB ({N:,} rows)")
    logger.info(f"  preload_policy         : {_policy}")
    logger.info(f"  auto_preload_mb        : {float(getattr(cfg, 'auto_preload_mb', 2048.0)):.1f} MB")
    logger.info(f"  estimated preload RAM  : {_est_ram_mb:.0f} MB")
    if _avail_ram_mb is not None:
        logger.info(f"  available RAM budget   : {_avail_ram_mb:.0f} MB (tightest detected limit)")
    else:
        logger.info("  available RAM budget   : unknown (preload safety veto unavailable)")
    logger.info(f"  decision               : {'RAM preload' if should_preload else 'HDF5 streaming'}")
    logger.info(f"  reason                 : {_preload_reason}")


@dataclass
class _SplitResult:
    """Outputs of the data-splitting phase, consumed by the rest of ``train()``."""

    split_seed: int
    split_policy: str
    train_indices: np.ndarray | None
    val_indices: np.ndarray | None
    n_train: int
    n_val: int
    split_manifest: dict[str, Any]
    split_manifest_path: Path
    splits: dict[str, np.ndarray] | None


def _resolve_data_splits(
    cfg: TrainConfig,
    *,
    meta: DatasetMeta,
    N: int,
    independent_val: bool,
    N_train_file: int | None,
    N_val_file: int | None,
    primary_path: Path,
    dset_name: str,
    dataset_contract_obj: DatasetContract,
    validation_report: Mapping[str, Any],
    layout: Any,
) -> _SplitResult:
    """Resolve train/val splits and write the split manifest.

    Extracted verbatim from ``train()`` phase 4; behaviour is identical.
    Independent train/val files bypass index splitting; in-file datasets use
    the configured ``split_policy`` (geometry-aware policies read positions).
    """
    split_seed = int(cfg.split_seed if cfg.split_seed is not None else cfg.seed)
    split_policy = str(getattr(cfg, "split_policy", "seeded_random") or "seeded_random")
    splits: dict[str, np.ndarray] | None = None
    if independent_val:
        train_indices = None
        val_indices = None
        n_train = N_train_file
        n_val = N_val_file
        split_manifest = {
            "schema_version": 1,
            "dataset_id": dataset_contract_obj.dataset_id,
            "split_policy": "independent_files",
            "split_seed": split_seed,
            "train_count": int(n_train),
            "val_count": int(n_val),
            "test_count": 0,
            "ood_count": 0,
            "index_hashes": {},
            "altitude_range_per_split": {},
            "created_at_utc": validation_report.get("created_at_utc"),
        }
    else:
        # Geometry-aware policies (spatial / OOD / altitude) need positions.
        # Random splits do not, so the xyz read is skipped for them.
        xyz_all: np.ndarray | None = None
        altitude_all: np.ndarray | None = None
        if split_policy not in {"seeded_random", "random"}:
            with h5py.File(primary_path, "r", swmr=True) as f:
                xyz_all = np.asarray(f[dset_name][:, 0:3], dtype=np.float64)
            if meta.unit_system == "canonical":
                # Absolute altitude thresholds are physical km.  Convert the
                # coordinate-only split input before deriving the envelope;
                # the --no-si + canonical combination is rejected earlier.
                if not meta.can_convert_to_si():
                    raise ValueError(
                        "Canonical dataset lacks DU_m/TU_s/VU_m_s required for physical split altitudes."
                    )
                assert meta.DU_m is not None
                xyz_all = xyz_all * float(meta.DU_m)
            altitude_all = (np.linalg.norm(xyz_all, axis=1) - float(meta.r_ref_m or R_MOON_SI)) / 1000.0
        _raw_split_options = {
            "spatial_lon_bins": int(getattr(cfg, "spatial_lon_bins", 12)),
            "spatial_lat_bins": int(getattr(cfg, "spatial_lat_bins", 6)),
            "spatial_val_block_fraction": getattr(cfg, "spatial_val_block_fraction", None),
            "spatial_test_block_fraction": getattr(cfg, "spatial_test_block_fraction", None),
            "spatial_altitude_bins": int(getattr(cfg, "spatial_altitude_bins", 4)),
            "ood_low_altitude_max_km": getattr(cfg, "ood_low_altitude_max_km", None),
            "ood_high_altitude_min_km": getattr(cfg, "ood_high_altitude_min_km", None),
            "ood_holdout_fraction": float(getattr(cfg, "ood_holdout_fraction", 0.2)),
        }
        split_options = {k: v for k, v in _raw_split_options.items() if v is not None}
        split_info: dict[str, Any] = {}
        try:
            splits = split_dataset_indices(
                n_rows=N,
                split_policy=split_policy,
                split_seed=split_seed,
                val_fraction=float(cfg.val_ratio),
                test_fraction=float(getattr(cfg, "test_fraction", 0.0)),
                altitude_km=altitude_all,
                xyz=xyz_all,
                options=split_options,
                split_info_out=split_info,
            )
        except (ValueError, NotImplementedError) as exc:
            raise ValueError(f"Unsupported/invalid training split_policy={split_policy!r}: {exc}") from exc
        train_indices = splits["train"]
        val_indices = splits["val"]
        n_train = int(train_indices.size)
        n_val = int(val_indices.size)
        if n_train == 0 or n_val == 0:
            raise ValueError(
                f"split_policy={split_policy!r} produced an empty train ({n_train}) or "
                f"val ({n_val}) split; adjust split fractions/thresholds."
            )
        split_manifest = build_split_manifest(
            dataset_contract=dataset_contract_obj,
            splits=splits,
            split_policy=split_policy,
            split_seed=split_seed,
            altitude_km=altitude_all,
            xyz=xyz_all,
            spatial_bins=split_info.get("spatial_bins"),
            ood_thresholds=split_info.get("ood_thresholds"),
        )
    split_manifest_path = write_split_manifest(layout.provenance_dir / "split_manifest.json", split_manifest)
    # Persist the exact split indices alongside the manifest so downstream
    # evaluation can enforce a held-out split deterministically (hash-verified)
    # rather than re-deriving it from the policy/seed and risking a rounding
    # mismatch. Independent-file splits have no in-file indices to record.
    if splits is not None:
        write_split_indices(layout.provenance_dir / SPLIT_INDICES_FILENAME, splits)
    return _SplitResult(
        split_seed=split_seed,
        split_policy=split_policy,
        train_indices=train_indices,
        val_indices=val_indices,
        n_train=n_train,
        n_val=n_val,
        split_manifest=split_manifest,
        split_manifest_path=split_manifest_path,
        splits=splits,
    )


def _maybe_load_baseline_gravity_model(
    meta: DatasetMeta, target_contract: TargetContract
) -> Any | None:
    """Load the source SH gravity model for a full-field spherical-harmonics
    baseline; return ``None`` for residual / point-mass / none contracts.

    The common (residual) path never touches the gravity file. When a full-field
    SH baseline is requested, the coefficients are loaded once from
    ``meta.gravity_model_path`` (truncated to ``base_degree``) and threaded into
    the scaler fit and the loss so the analytical SH field can be subtracted.
    """
    if not (
        target_contract.target_mode == "full"
        and target_contract.baseline_kind == "spherical_harmonics"
    ):
        return None
    path = getattr(meta, "gravity_model_path", None)
    if not path:
        raise ValueError(
            "Full-field spherical-harmonics baseline needs the source gravity "
            "model, but the dataset metadata has no gravity_model_path. "
            "Regenerate the dataset with a recorded gravity_model_path, or train "
            "on a residual dataset (which already stores baseline-subtracted "
            "labels)."
        )
    from lunaris.physics.spherical_harmonics import GravityModel

    return GravityModel.from_file(str(path), requested_degree=int(target_contract.base_degree))


def _fit_residual_scalers(
    cfg: TrainConfig,
    *,
    layout: Any,
    meta: DatasetMeta,
    primary_path: Path,
    dset_name: str,
    mu_val: float,
    a_sign: float,
    degree_min_val: int,
    effective_target: str,
    target_contract: TargetContract,
    independent_val: bool,
    train_indices: np.ndarray | None,
    split_policy: str,
    split_seed: int,
    n_train: int,
    n_val: int,
    split_manifest: dict[str, Any],
    dataset_contract_obj: DatasetContract,
) -> ScalerPack:
    """Fit (or load) the residual ΔU/Δa scalers and record their provenance.

    Extracted verbatim from ``train()`` phase 7; behaviour is identical. The
    leakage-safe contract is preserved exactly: scalers are fit on TRAIN ROWS
    ONLY (``indices=train_indices`` for single-file datasets; the dedicated
    train file for independent-file datasets), and the train-only ``fit_scope``
    provenance is written alongside.
    """
    scaler_path = layout.scaler_json
    scaler_hash_info: dict[str, Any]
    if scaler_path.exists():
        logger.info(f"Loading existing scaler from {scaler_path.name}")
        scaler = ScalerPack.load_json(scaler_path)
        # A scaler is train data, not a generic cache.  Reusing it in a fresh
        # run is safe only when its split and dataset identity match exactly.
        # Compute the content hash once if an older contract did not embed it.
        from lunaris.surrogate.st_lrps.data.dataset_contract import content_sha256_for_hdf5_dataset

        split_index_hashes = (split_manifest.get("index_hashes", {}) if isinstance(split_manifest, dict) else {}) or {}
        expected_content_sha = (
            getattr(dataset_contract_obj, "content_sha256", None)
            or (split_manifest.get("dataset_content_sha256") if isinstance(split_manifest, dict) else None)
            or content_sha256_for_hdf5_dataset(primary_path, dataset_name=dset_name)
        )
        expected_provenance = {
            "fit_scope": "train_only",
            "split_policy": ("independent_files" if independent_val else split_policy),
            "split_seed": int(split_seed),
            "train_count": int(n_train),
            "val_count": int(n_val),
            "test_count": int(split_manifest.get("test_count", 0)) if isinstance(split_manifest, dict) else 0,
            "train_index_hash": split_index_hashes.get("train"),
            "val_index_hash": split_index_hashes.get("val"),
            "test_index_hash": split_index_hashes.get("test"),
            "dataset_content_sha256": expected_content_sha,
            "dataset_contract_hash": compute_payload_sha256(dataset_contract_obj.to_dict()),
        }
        stored_provenance = dict(getattr(scaler, "provenance", {}) or {})
        mismatches = []
        for key, expected in expected_provenance.items():
            # Independent-file runs intentionally have no in-file split hashes.
            if expected is None and key.endswith("_index_hash"):
                continue
            if stored_provenance.get(key) != expected:
                mismatches.append((key, stored_provenance.get(key), expected))
        if stored_provenance.get("fit_scope") != "train_only":
            mismatches.append(("fit_scope", stored_provenance.get("fit_scope"), "train_only"))
        if mismatches:
            detail = ", ".join(f"{k}={old!r} (expected {new!r})" for k, old, new in mismatches)
            raise ValueError(
                f"Existing scaler provenance does not match the current dataset/split: {detail}. "
                "Use a fresh output directory or remove scaler.json; stale scalers are refused."
            )
        logger.info("Existing scaler provenance matches the current dataset contract and train split.")
        scaler_hash_info = {
            "scaler_hash": compute_payload_sha256(asdict(scaler)),
            "scaler_file_sha256": compute_file_sha256(scaler_path),
            "scaler_payload": asdict(scaler),
        }
    else:
        # Leakage-safe scaler provenance. Scalers (including the residual
        # ΔU/Δa target scalers) are fit on TRAIN ROWS ONLY: for single-file
        # datasets we pass train_indices so validation/test/OOD target rows
        # never influence the mean/scale; for independent files primary_path is
        # already the dedicated train file (indices=None, still train-only).
        _split_index_hashes = (
            split_manifest.get("index_hashes", {}) if isinstance(split_manifest, dict) else {}
        ) or {}
        try:
            _dataset_contract_hash = compute_payload_sha256(dataset_contract_obj.to_dict())
        except Exception:
            _dataset_contract_hash = None
        _dataset_content_sha256 = getattr(dataset_contract_obj, "content_sha256", None) or (
            split_manifest.get("dataset_content_sha256") if isinstance(split_manifest, dict) else None
        )
        scaler_split_provenance = {
            "fit_scope": "train_only",
            "split_policy": ("independent_files" if independent_val else split_policy),
            "split_seed": int(split_seed),
            "train_count": int(n_train),
            "val_count": int(n_val),
            "test_count": int(split_manifest.get("test_count", 0)) if isinstance(split_manifest, dict) else 0,
            "train_index_hash": _split_index_hashes.get("train"),
            "val_index_hash": _split_index_hashes.get("val"),
            "test_index_hash": _split_index_hashes.get("test"),
            "dataset_content_sha256": _dataset_content_sha256,
            "dataset_contract_hash": _dataset_contract_hash,
        }
        scaler = fit_scaler_streaming(
            h5_path=primary_path, dset_name=dset_name, meta=meta,
            use_si=cfg.use_si, mu_si=mu_val, a_sign=a_sign,
            n_fit=cfg.fit_rows, seed=cfg.fit_seed, chunk_rows=cfg.fit_chunk_rows,
            degree_min=degree_min_val,
            target_mode=effective_target,
            degree_max=int(meta.degree_max if meta.degree_max is not None else (meta.requested_degree or -1)),
            u_scale_mode=str(getattr(cfg, "u_scale_mode", "hybrid")),
            a_scale_mode=str(getattr(cfg, "a_scale_mode", "hybrid")),
            target_scale_multiplier=float(getattr(cfg, "target_scale_multiplier", 6.0)),
            target_contract=target_contract,
            indices=(None if independent_val else train_indices),
            split_provenance=scaler_split_provenance,
            gravity_model=_maybe_load_baseline_gravity_model(meta, target_contract),
        )
        scaler_hash_info = write_scaler_json(layout, scaler)
    logger.info(f"[artifacts] scaler_hash={scaler_hash_info['scaler_hash']}")
    update_run_manifest(
        layout,
        {
            "scaler_path": str(layout.scaler_json),
            "scaler_hash": scaler_hash_info["scaler_hash"],
            "scaler_file_sha256": scaler_hash_info["scaler_file_sha256"],
        },
    )
    return scaler


@dataclass
class _LoaderResult:
    """Outputs of the DataLoader-construction phase."""

    train_loader: DataLoader
    val_loader: DataLoader
    n_train: int
    n_val: int


def _build_dataloaders(
    cfg: TrainConfig,
    *,
    meta: DatasetMeta,
    device: torch.device,
    bytes_est: int,
    N: int,
    independent_val: bool,
    data_path: Path,
    train_data_path: Path | None,
    val_data_path: Path | None,
    dset_name: str,
    n_train: int,
    n_val: int,
    train_indices: np.ndarray | None,
    val_indices: np.ndarray | None,
) -> _LoaderResult:
    """Construct the train/val DataLoaders (RAM-preload or HDF5 streaming).

    Extracted verbatim from ``train()`` phase 8; behaviour is identical. The
    determinism guarantees are preserved exactly: ``worker_init_fn`` seeding on
    every loader and the explicit per-epoch shuffle ``torch.Generator`` (seeded
    from ``cfg.seed``) in the RAM-preload path, so ordering stays reproducible
    and resumable via the checkpointed RNG state.
    """
    dataset_mb = bytes_est / (1024.0 * 1024.0)
    # Resolve the preload policy. --preload-data is a convenience alias for "always".
    _policy = str(getattr(cfg, "preload_policy", "auto")).strip().lower()
    if bool(getattr(cfg, "preload_data", False)) and _policy != "never":
        _policy = "always"
    _est_ram_mb = _estimate_preload_ram_mb(int(N))
    _avail_ram_mb = _available_ram_mb()
    should_preload, _preload_reason = _decide_preload(
        _policy,
        dataset_mb=dataset_mb,
        auto_preload_mb=float(getattr(cfg, "auto_preload_mb", 2048.0)),
        est_ram_mb=_est_ram_mb,
        avail_ram_mb=_avail_ram_mb,
    )
    _log_data_loading_policy(N, _avail_ram_mb, _est_ram_mb, _policy, _preload_reason, cfg, dataset_mb, should_preload)
    if should_preload and "WARNING" in _preload_reason:
        logger.warning(f"Preload RAM-safety: {_preload_reason}")

    if should_preload:
        logger.info("Data mode: RAM preload")
        if independent_val:
            logger.info(f"Loading train ({n_train:,}) from {train_data_path.name}...")
            with h5py.File(train_data_path, "r", libver="latest", swmr=True) as _f:
                _arr_train = np.asarray(_f[dset_name][:], dtype=np.float64)
            logger.info(f"Loading val ({n_val:,}) from {val_data_path.name}...")
            with h5py.File(val_data_path, "r", libver="latest", swmr=True) as _f:
                _arr_val = np.asarray(_f[dset_name][:], dtype=np.float64)

            _xt, _ut, _at = _arr_train[:, 0:3], _arr_train[:, 3:4], _arr_train[:, 4:7]
            _xv, _uv, _av = _arr_val[:, 0:3], _arr_val[:, 3:4], _arr_val[:, 4:7]
            del _arr_train, _arr_val

            if cfg.use_si and meta.unit_system == "canonical":
                _xt, _ut, _at = meta.convert_xyz_U_a_to_si(_xt, _ut, _at)
                _xv, _uv, _av = meta.convert_xyz_U_a_to_si(_xv, _uv, _av)

            train_ds: Dataset = TensorMemoryDataset(
                _xt.astype(np.float32), _ut.astype(np.float32), _at.astype(np.float32)
            )
            val_ds: Dataset = TensorMemoryDataset(
                _xv.astype(np.float32), _uv.astype(np.float32), _av.astype(np.float32)
            )
            del _xt, _ut, _at, _xv, _uv, _av
        else:
            logger.info(f"Loading {N:,} rows into CPU memory (~{dataset_mb:.2f} MB)...")
            with h5py.File(data_path, "r", libver="latest", swmr=True) as _f:
                _arr = np.asarray(_f[dset_name][:], dtype=np.float64)

            _x_mem = _arr[:, 0:3]
            _u_mem = _arr[:, 3:4]
            _a_mem = _arr[:, 4:7]
            del _arr

            if cfg.use_si and meta.unit_system == "canonical":
                _x_mem, _u_mem, _a_mem = meta.convert_xyz_U_a_to_si(_x_mem, _u_mem, _a_mem)

            _x_mem = _x_mem.astype(np.float32)
            _u_mem = _u_mem.astype(np.float32)
            _a_mem = _a_mem.astype(np.float32)

            train_ds = TensorMemoryDataset(
                _x_mem[train_indices], _u_mem[train_indices], _a_mem[train_indices]
            )
            val_ds = TensorMemoryDataset(
                _x_mem[val_indices], _u_mem[val_indices], _a_mem[val_indices]
            )
            del _x_mem, _u_mem, _a_mem

        n_train = len(train_ds)
        n_val   = len(val_ds)
        pin = cfg.pin_memory and device.type == "cuda"
        mem_workers = max(0, cfg.num_workers)
        pf = cfg.prefetch_factor if (mem_workers > 0 and cfg.prefetch_factor is not None) else None
        logger.info(f"Train/val split: {n_train:,} / {n_val:,}")
        logger.info(
            f"pin_memory={pin}, non_blocking={pin}, num_workers={mem_workers} (requested={cfg.num_workers})"
            + (f", prefetch_factor={pf}" if pf is not None else "")
        )

        _dl_kw: dict[str, Any] = dict(
            batch_size=cfg.batch_size, num_workers=mem_workers, pin_memory=pin,
            persistent_workers=(mem_workers > 0), collate_fn=collate_xyz_u_a,
            worker_init_fn=_seed_dataloader_worker,
        )
        if pf is not None:
            _dl_kw["prefetch_factor"] = pf
        train_sampler = TensorBatchSampler(
            len(train_ds), cfg.batch_size, seed=cfg.seed + 100, shuffle=True, drop_last=True
        )
        val_sampler = TensorBatchSampler(
            len(val_ds), cfg.batch_size, seed=cfg.seed + 200, shuffle=False, drop_last=False
        )
        _dl_kw.pop("collate_fn", None)
        _dl_kw.pop("worker_init_fn", None)
        train_loader = DataLoader(
            train_ds, sampler=train_sampler, batch_size=None, collate_fn=identity_collate, **_dl_kw,
        )
        val_loader = DataLoader(
            val_ds, sampler=val_sampler, batch_size=None, collate_fn=identity_collate, **_dl_kw,
        )
    else:
        logger.info("Data mode: HDF5 streaming")
        if independent_val:
            train_ds = H5BlockDataset(
                train_data_path, dset_name, 0, n_train, meta, cfg.use_si, cfg.cache_rows, indices=None
            )
            val_ds = H5BlockDataset(
                val_data_path, dset_name, 0, n_val, meta, cfg.use_si, cfg.cache_rows, indices=None
            )
        else:
            train_ds = H5BlockDataset(
                data_path, dset_name, 0, N, meta, cfg.use_si, cfg.cache_rows, indices=train_indices
            )
            val_ds = H5BlockDataset(
                data_path, dset_name, 0, N, meta, cfg.use_si, cfg.cache_rows, indices=val_indices
            )

        train_sampler = BlockShuffleSampler(len(train_ds), cfg.sampler_block_size, cfg.seed + 100)
        val_sampler   = BlockShuffleSampler(len(val_ds),   cfg.sampler_block_size, cfg.seed + 200)
        _streaming_path = train_data_path if independent_val else data_path
        train_workers = _resolve_loader_worker_count(_streaming_path, cfg.num_workers)
        if train_workers == 0 and int(cfg.num_workers) > 0:
            logger.warning(
                "Windows HDF5 safety: num_workers forced to 0 for HDF5 streaming. "
                "Use --preload-data (or --auto-preload-mb) for multi-worker loading."
            )
        val_workers = max(0, train_workers // 2)
        pin = cfg.pin_memory and device.type == "cuda"
        tr_pf = cfg.prefetch_factor if (train_workers > 0 and cfg.prefetch_factor is not None) else None
        va_pf = cfg.prefetch_factor if (val_workers   > 0 and cfg.prefetch_factor is not None) else None
        logger.info(
            f"pin_memory={pin}, train_workers={train_workers} (requested={cfg.num_workers}),"
            f" val_workers={val_workers}"
            + (f", prefetch_factor={tr_pf}" if tr_pf is not None else "")
        )

        _tr_kw: dict[str, Any] = dict(
            batch_size=cfg.batch_size, sampler=train_sampler,
            num_workers=train_workers, pin_memory=pin,
            persistent_workers=(train_workers > 0), collate_fn=collate_xyz_u_a, drop_last=True,
            worker_init_fn=_seed_dataloader_worker,
        )
        _va_kw: dict[str, Any] = dict(
            batch_size=cfg.batch_size, sampler=val_sampler,
            num_workers=val_workers, pin_memory=pin,
            persistent_workers=(val_workers > 0), collate_fn=collate_xyz_u_a, drop_last=False,
            worker_init_fn=_seed_dataloader_worker,
        )
        if tr_pf is not None:
            _tr_kw["prefetch_factor"] = tr_pf
        if va_pf is not None:
            _va_kw["prefetch_factor"] = va_pf
        train_loader = DataLoader(train_ds, **_tr_kw)
        val_loader   = DataLoader(val_ds,   **_va_kw)

    return _LoaderResult(
        train_loader=train_loader,
        val_loader=val_loader,
        n_train=n_train,
        n_val=n_val,
    )


@dataclass
class _DatasetContext:
    """Outputs of the dataset discovery + metadata + contract/validation phases."""

    data_path: Path
    independent_val: bool
    train_data_path: Path | None
    val_data_path: Path | None
    primary_path: Path
    dset_name: str
    meta: DatasetMeta
    N: int
    N_train_file: int | None
    N_val_file: int | None
    bytes_est: int
    dataset_contract_obj: DatasetContract
    validation_report: dict[str, Any]


def _load_dataset_context(cfg: TrainConfig, *, layout: Any) -> _DatasetContext:
    """Resolve dataset paths, read metadata (SSOT), and load+validate the contract.

    Extracted verbatim from ``train()`` phases 2-3; behaviour is identical.
    Independent train/val files are matched on their lunar gravity contract;
    in-file datasets read the row count directly. ``N_train_file``/``N_val_file``
    are populated only for independent-file datasets (``None`` otherwise).
    """
    # 2. Dataset Discovery & Validation
    data_path = Path(cfg.data)
    independent_val = cfg.train_data is not None and cfg.val_data is not None
    train_data_path: Path | None = None
    val_data_path: Path | None = None
    N_train_file: int | None = None
    N_val_file: int | None = None

    if independent_val:
        train_data_path = Path(cfg.train_data)
        val_data_path = Path(cfg.val_data)
        if not train_data_path.exists():
            raise FileNotFoundError(f"Train dataset not found: {train_data_path}")
        if not val_data_path.exists():
            raise FileNotFoundError(f"Val dataset not found: {val_data_path}")
        primary_path = train_data_path
    else:
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        primary_path = data_path

    dset_name = cfg.dataset_name
    try:
        with h5py.File(primary_path, "r") as f:
            _ = f[dset_name]
    except (KeyError, OSError):
        dset_name = _discover_dataset_name(primary_path, preferred=cfg.dataset_name)

    # 3. Read Metadata (SSOT)
    meta = DatasetMeta.from_h5(primary_path)

    if independent_val:
        with h5py.File(train_data_path, "r", swmr=True) as f:
            N_train_file = int(f[dset_name].shape[0])
            bytes_est_train = N_train_file * 7 * (4 if str(f[dset_name].dtype) == "float32" else 8)
        with h5py.File(val_data_path, "r", swmr=True) as f:
            N_val_file = int(f[dset_name].shape[0])
            bytes_est_val = N_val_file * 7 * (4 if str(f[dset_name].dtype) == "float32" else 8)
        bytes_est = bytes_est_train + bytes_est_val
        N = N_train_file + N_val_file
        meta_val = DatasetMeta.from_h5(val_data_path)
        _resolve_lunar_dataset_contract(meta_val, data_path=val_data_path)

        def _require_meta_match(name: str, left: Any, right: Any) -> None:
            if left is None or right is None:
                return
            if isinstance(left, float) or isinstance(right, float):
                if abs(float(left) - float(right)) <= 1.0:
                    return
            elif left == right:
                return
            raise ValueError(
                f"Train/Val metadata mismatch for {name}: {left!r} vs {right!r}. "
                "Independent train/validation clouds must use the same lunar gravity contract."
            )

        _require_meta_match("central_body", meta.central_body, meta_val.central_body)
        _require_meta_match("mu_si", meta.mu_si, meta_val.mu_si)
        _require_meta_match("r_ref_m", meta.r_ref_m, meta_val.r_ref_m)
        _require_meta_match("unit_system", meta.unit_system, meta_val.unit_system)
        _require_meta_match("degree_min", meta.degree_min, meta_val.degree_min)
        _require_meta_match("requested_degree", meta.requested_degree, meta_val.requested_degree)
        _require_meta_match("target_mode", meta.target_mode, meta_val.target_mode)
        if meta_val.mu_si is not None and meta.mu_si is not None:
            if abs(meta_val.mu_si - meta.mu_si) > 1.0:
                logger.warning(f"Train/Val mu_si mismatch: {meta.mu_si} vs {meta_val.mu_si}")
        if meta_val.r_ref_m is not None and meta.r_ref_m is not None:
            if abs(meta_val.r_ref_m - meta.r_ref_m) > 1.0:
                logger.warning(f"Train/Val r_ref_m mismatch: {meta.r_ref_m} vs {meta_val.r_ref_m}")
        if meta.unit_system != meta_val.unit_system:
            logger.warning(f"Train/Val unit_system mismatch: {meta.unit_system} vs {meta_val.unit_system}")
        if meta.degree_min != meta_val.degree_min:
            logger.warning(f"Train/Val degree_min mismatch: {meta.degree_min} vs {meta_val.degree_min}")
    else:
        with h5py.File(data_path, "r", swmr=True) as f:
            N = int(f[dset_name].shape[0])
            bytes_est = N * 7 * (4 if str(f[dset_name].dtype) == "float32" else 8)

    if not bool(cfg.use_si) and meta.unit_system == "canonical":
        raise ValueError(
            "--no-si is unsafe for canonical ST-LRPS datasets: canonical coordinates "
            "are nondimensional and cannot be combined with SI lunar constants. "
            "Use --use-si or provide an SI dataset."
        )
    if cfg.use_si and meta.unit_system == "canonical" and not meta.can_convert_to_si():
        raise ValueError("Configuration demands SI units, but dataset is missing DU_m/TU_s/VU_m_s attributes.")

    # Dataset metadata is the authoritative training shell.  CLI values remain
    # available for deliberate experiments, but an ordinary run should not
    # silently retain the global 100--1000 km defaults for a different cloud.
    if (
        not bool(getattr(cfg, "_altitude_bounds_explicit", False))
        and meta.alt_min_km is not None
        and meta.alt_max_km is not None
    ):
        cfg.altitude_min_km = float(meta.alt_min_km)
        cfg.altitude_max_km = float(meta.alt_max_km)
        logger.info(
            "Altitude-balanced loss bounds resolved from dataset metadata: "
            f"[{cfg.altitude_min_km:.3f}, {cfg.altitude_max_km:.3f}] km"
        )

    dataset_contract_obj = DatasetContract.from_hdf5(
        primary_path,
        dataset_name=dset_name,
    )
    validation_report = validate_dataset_file(
        primary_path,
        out_dir=layout.provenance_dir,
        dataset_name=dset_name,
        n_check=min(1024, int(N)),
        seed=int(cfg.split_seed if cfg.split_seed is not None else cfg.seed),
    )
    if not validation_report.get("passed") and not bool(getattr(cfg, "allow_dataset_validation_fail", False)):
        raise ValueError(
            "Dataset validation failed before training: "
            + "; ".join(str(item) for item in validation_report.get("errors", []))
        )
    return _DatasetContext(
        data_path=data_path,
        independent_val=independent_val,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        primary_path=primary_path,
        dset_name=dset_name,
        meta=meta,
        N=N,
        N_train_file=N_train_file,
        N_val_file=N_val_file,
        bytes_est=bytes_est,
        dataset_contract_obj=dataset_contract_obj,
        validation_report=validation_report,
    )


@dataclass
class _ModelBundle:
    """Outputs of the model/optimizer/loss build phase."""

    model: nn.Module
    opt: AdamW
    weights: GradNormWeights
    loss_fn: nn.Module
    arch_signature: str
    degree_max_val: int
    emb_type_built: str
    in_fdim_built: int
    total_params: int


def _build_model_and_optim(
    cfg: TrainConfig,
    *,
    meta: DatasetMeta,
    degree_min_val: int,
    resolved_r_ref_m: float,
    scaler: ScalerPack,
    device: torch.device,
    a_sign: float,
    mu_val: float,
    target_contract: TargetContract,
    _resume_requested: bool,
    _resume_ckpt: dict[str, Any] | None,
    _resume_ckpt_path: Path | None,
) -> _ModelBundle:
    """Build the model, GradNorm weights, Sobolev loss, and AdamW optimizer.

    Extracted verbatim from ``train()`` phase 9; behaviour is identical. The
    reload-safety contract is preserved: the dataset degree range and the
    multi-scale band frequencies are resolved INTO ``cfg`` before the model is
    built and before ``config.json`` is written, so training and evaluation
    reconstruct the identical spectrum. Resume restores (model/GradNorm/optimizer
    state) are applied here when ``_resume_requested``.
    """
    # 9. Build model via the shared factory (build_model_from_config) — authoritative builder
    # used by evaluator and force model. This ensures SH/radial encoding flags are honoured.
    # The SIREN+Fourier mutual exclusion check is inside build_model_from_config().
    #
    # CRITICAL (reload-safety): resolve the dataset's degree range and the
    # multi-scale band frequencies INTO cfg before building, and BEFORE writing
    # config.json. Previously the model was built from cfg (which had no degree
    # fields → silent 0/50 defaults) while config.json recorded the meta-derived
    # degrees. For n_bands>1 that produced a model whose SIREN band frequencies
    # differed from what evaluation reconstructed: the state_dict matched by
    # shape but the functional model was wrong. Resolving here makes training
    # and evaluation build the identical spectrum.
    degree_max_val = int(
        meta.degree_max if meta.degree_max is not None
        else (meta.requested_degree if meta.requested_degree is not None else -1)
    )
    cfg.degree_min = int(degree_min_val)
    cfg.degree_max = int(degree_max_val)
    cfg.resolved_r_ref_m = float(resolved_r_ref_m)
    cfg.x_scale_m = float(scaler.x.scale)
    if cfg.activation.lower() == "sine" and int(getattr(cfg, "n_bands", 1)) > 1:
        if cfg.degree_max <= 0:
            raise ValueError(
                "Multi-scale SIREN (n_bands>1) requires a known degree_max from the "
                f"dataset metadata, but resolved degree_max={cfg.degree_max}. "
                "Regenerate the dataset with degree_max recorded, or use n_bands=1."
            )
        cfg.w0_bands = [
            float(w) for w in _compute_harmonic_w0_bands(
                int(cfg.n_bands), int(cfg.degree_min), int(cfg.degree_max)
            )
        ]
    else:
        cfg.w0_bands = None

    model = build_model_from_config(
        cfg,
        in_dim=3,
        device=device,
        dtype=DTYPE,
    )

    if _resume_requested and _resume_ckpt is not None:
        _msd = _resume_ckpt.get("model_state_dict") or _resume_ckpt.get("model")
        if _msd is None:
            raise RuntimeError("[resume] checkpoint has no model_state_dict.")
        _load_res = model.load_state_dict(_msd, strict=bool(getattr(cfg, "resume_strict", True)))
        _missing = list(getattr(_load_res, "missing_keys", []) or [])
        _unexpected = list(getattr(_load_res, "unexpected_keys", []) or [])
        if _missing or _unexpected:
            logger.warning(f"[resume] non-strict model load: missing={_missing} unexpected={_unexpected}")
        logger.info(f"[resume] model weights restored from {_resume_ckpt_path}")

    # Log architecture details (equivalent to old manual logging, but from the built model)
    _n_bands_built = max(1, int(getattr(cfg, "n_bands", 1)))
    _use_res_built = bool(getattr(cfg, "use_residual_blocks", False))
    if cfg.use_fourier:
        logger.info(
            f"Fourier embedding: n_features={cfg.fourier_n_features}, "
            f"sigma={cfg.fourier_sigma}, append_raw={cfg.fourier_append_raw}"
        )
    if cfg.activation.lower() == "sine":
        if _n_bands_built > 1:
            # Log the EXACT bands the model was built with (resolved into cfg above),
            # not an independently recomputed value that could silently diverge.
            logger.info(
                f"Built Multi-Scale SIREN: n_bands={_n_bands_built}, w0_bands={cfg.w0_bands}, "
                f"degree_min={cfg.degree_min}, degree_max={cfg.degree_max}, "
                f"depth={cfg.depth}, hidden={cfg.hidden}"
            )
            # Defensive cross-check: the model's resolved bands must equal cfg's.
            _model_bands = list(getattr(model, "w0_bands", []) or [])
            if _model_bands and [round(b, 4) for b in _model_bands] != [round(b, 4) for b in (cfg.w0_bands or [])]:
                raise RuntimeError(
                    f"Internal error: model w0_bands {_model_bands} != cfg.w0_bands {cfg.w0_bands}. "
                    "Refusing to train a model whose spectrum cannot be reproduced from config."
                )
        else:
            logger.info(
                f"Built SIREN backbone: depth={cfg.depth}, hidden={cfg.hidden}, "
                f"w0_first={cfg.w0_first}, w0_hidden={cfg.w0_hidden}, "
                f"residual_blocks={_use_res_built}"
            )
    else:
        logger.info(
            f"Built MLP backbone: depth={cfg.depth}, hidden={cfg.hidden}, activation={cfg.activation}"
        )
    _total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {_total_params:,}")

    # Capture model-derived architecture metadata for config.json / checkpoint
    # persistence below. (Previously this block tried to rewrite config.json
    # before it had been created, so the metadata was silently lost every run.)
    _emb_type_built = str(getattr(model, "embedding_type", "raw"))
    _in_fdim_built = int(getattr(model, "input_feature_dim", 3))
    _arch_signature = compute_architecture_signature(cfg)
    logger.info(
        f"Encoding: embedding_type={_emb_type_built}  input_feature_dim={_in_fdim_built}  "
        f"builder={MODEL_BUILDER_VERSION}  arch_signature={_arch_signature}"
    )

    weights = GradNormWeights(
        w_u=cfg.w_u,
        w_a=cfg.w_a,
        mode=cfg.gradnorm_mode,
        w_a_min=cfg.gradnorm_w_a_min,
        w_a_max=cfg.gradnorm_w_a_max,
    )
    logger.info(f"Loss weighting: mode={cfg.gradnorm_mode}  w_u={cfg.w_u:.2f}  w_a_init={cfg.w_a:.2f}")

    if _resume_requested:
        _gnw_state = (_resume_ckpt.get("training_state") or {}).get("gradnorm_weights")
        if _gnw_state:
            weights.load_state_dict(_gnw_state)
            logger.info(
                f"[resume] GradNorm state restored (w_a={weights.w_a:.4f}, ntk_done={weights._ntk_done})."
            )
        else:
            logger.warning(
                "[resume] checkpoint lacks GradNorm state; continuing with a fresh GradNorm "
                "(NTK init may recompute on the first resumed step)."
            )

    loss_fn = SobolevLoss(
        scaler=scaler,
        a_sign=a_sign,
        mu_si=mu_val,
        r_ref_m=resolved_r_ref_m,
        degree_min=degree_min_val,
        degree_max=degree_max_val,
        target_contract=target_contract,
        gravity_model=_maybe_load_baseline_gravity_model(meta, target_contract),
    ).to(device=device, dtype=DTYPE)
    logger.info(f"Residual baseline: {target_contract.baseline_description}")

    head_params = _get_output_head_params(model)
    head_param_ids = {id(param) for param in head_params}
    body_params = [param for param in model.parameters() if id(param) not in head_param_ids]
    param_groups: list[dict[str, Any]] = []
    if body_params:
        param_groups.append(
            {
                "params": body_params,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
            }
        )
    param_groups.append(
        {
            "params": head_params,
            "lr": cfg.lr * float(cfg.output_head_lr_mult),
            "weight_decay": 0.0,
        }
    )
    opt: AdamW
    if bool(getattr(cfg, "use_fused_optimizer", True)) and device.type == "cuda":
        try:
            opt = AdamW(param_groups, fused=True)
            logger.info("Optimizer: using fused AdamW on CUDA.")
        except (TypeError, RuntimeError) as exc:
            logger.warning("Fused AdamW unavailable; falling back to regular AdamW: %s", exc)
            opt = AdamW(param_groups)
    else:
        opt = AdamW(param_groups)
    logger.info(
        f"Optimizer groups: body_lr={cfg.lr:.2e}, body_wd={cfg.weight_decay:.2e}, "
        f"head_lr={cfg.lr * float(cfg.output_head_lr_mult):.2e}, head_wd=0.00e+00"
    )
    for group in opt.param_groups:
        group["initial_lr"] = float(group["lr"])

    if _resume_requested:
        _osd = _resume_ckpt.get("optimizer_state_dict")
        if _osd is not None:
            try:
                opt.load_state_dict(_osd)
                logger.info("[resume] optimizer state restored.")
            except Exception as _oexc:
                if bool(getattr(cfg, "resume_strict", True)):
                    raise RuntimeError(f"[resume] optimizer state restore failed: {_oexc}") from _oexc
                logger.warning(f"[resume] optimizer restore failed (non-strict); continuing fresh: {_oexc}")
        elif bool(getattr(cfg, "resume_strict", True)):
            raise RuntimeError(
                "[resume] checkpoint has no optimizer_state_dict (strict). "
                "Use --resume-nonstrict to continue with a fresh optimizer."
            )
        else:
            logger.warning("[resume] checkpoint has no optimizer_state_dict; continuing with a fresh optimizer.")
        # The checkpoint stores the previous run's scheduler base.  A resume
        # command is allowed to override --lr, so re-assert the current config
        # explicitly rather than silently inheriting the old base LR.
        expected_lrs = [float(cfg.lr)]
        if len(opt.param_groups) > 1:
            expected_lrs.append(float(cfg.lr) * float(cfg.output_head_lr_mult))
        for index, group in enumerate(opt.param_groups):
            base_lr = expected_lrs[min(index, len(expected_lrs) - 1)]
            group["initial_lr"] = base_lr
            group["lr"] = base_lr
        logger.info("[resume] optimizer base LR override applied from config: %s", expected_lrs)

    return _ModelBundle(
        model=model,
        opt=opt,
        weights=weights,
        loss_fn=loss_fn,
        arch_signature=_arch_signature,
        degree_max_val=degree_max_val,
        emb_type_built=_emb_type_built,
        in_fdim_built=_in_fdim_built,
        total_params=_total_params,
    )


@dataclass
class _TrainingSession:
    """Everything ``build_training_session`` (setup phases 1-10) produces that the
    training loop (phase 11, ``_run_training_loop``) consumes.

    The field set is exactly the boundary of locals that cross from setup into
    the loop; nothing else flows between the two halves of ``train()``.
    """

    cfg: TrainConfig
    model: nn.Module
    opt: AdamW
    weights: GradNormWeights
    loss_fn: nn.Module
    device: torch.device
    scaler: ScalerPack
    train_loader: DataLoader
    val_loader: DataLoader
    layout: Any
    outdir: Path
    config_path: Path
    payload: dict[str, Any]
    payload_readback: dict[str, Any]
    dataset_snapshot: dict[str, Any]
    checkpoint_selection: Mapping[str, Any]
    dset_name: str
    resolved_r_ref_m: float
    start_epoch: int
    _arch_signature: str
    _best_metric_canonical: str
    _ckpt_start: int
    _resume_requested: bool
    _resume_ckpt: dict[str, Any] | None
    _resume_prev_manifest: dict[str, Any]
    _resume_best_val: float
    _resume_best_epoch: int
    _resume_epochs_without_improve: int
    _resume_global_step: int


def build_training_session(cfg: TrainConfig) -> _TrainingSession:
    """Run setup phases 1-10 and return the assembled :class:`_TrainingSession`.

    This is a scalar residual potential surrogate, NOT a classical q,p
    Sobolev-Trained Lunar Residual Potential Surrogate.  The model learns DeltaU(x) and acceleration
    is obtained by differentiating the learned potential via autograd:
        Delta_a = a_sign * grad(DeltaU_scaled) * (u_scale / x_scale)
    """

    # 1. Initialization
    # Run-level preset first (may enforce split policy / determinism / preflight
    # posture), then the architecture preset. apply_run_preset is idempotent for
    # configs already resolved by parse_args; for programmatically built configs
    # it enforces the 'paper' posture before set_seed reads the flags below.
    apply_run_preset(cfg)
    apply_model_preset(cfg)
    if str(getattr(cfg, "runtime_model_kind", "potential_autograd")) == "force_direct":
        raise NotImplementedError(
            "runtime_model_kind='force_direct' is archived in the "
            "experimental/force-direct-archive branch and is no longer trainable on "
            "main. The Sobolev trainer supports only the conservative scalar-potential "
            "(potential_autograd) surrogate."
        )
    _determinism_flags = set_seed(
        cfg.seed,
        deterministic=bool(getattr(cfg, "deterministic", True)),
        benchmark=bool(getattr(cfg, "benchmark_cudnn", False)),
    )
    device = get_device()
    layout = ensure_run_layout(Path(cfg.out))
    outdir = layout.run_dir
    run_created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(layout.train_log)]
    )

    if device.type == "cuda":
        try:
            _n_gpu = int(torch.cuda.device_count())
            if _n_gpu > 1:
                logger.info(
                    f"{_n_gpu} CUDA devices are visible but ST-LRPS training uses a single "
                    f"device ({torch.cuda.get_device_name(device)}); the extra GPUs stay idle. "
                    "Request one GPU per job (e.g. --gres=gpu:1)."
                )
        except Exception:  # pragma: no cover - driver dependent
            pass

    # -----------------------------------------------------------------------
    # Resume resolution (epoch-level). When cfg.resume_from is set, restore
    # model/optimizer/GradNorm/RNG state (below) and continue from the last
    # completed epoch. cfg.epochs is the TOTAL target epoch count, not extra
    # epochs. Architecture/scaler/dataset-identity fields are locked to the
    # previous run so the rebuilt model + scaler match the checkpoint exactly.
    # -----------------------------------------------------------------------
    _resume_requested = bool(getattr(cfg, "resume_from", None))
    _resume_ckpt: dict[str, Any] | None = None
    _resume_ckpt_path: Path | None = None
    start_epoch = 0
    _resume_best_val = float("inf")
    _resume_best_epoch = -1
    _resume_epochs_without_improve = 0
    _resume_global_step = 0
    _resume_prev_manifest: dict[str, Any] = {}
    if _resume_requested:
        _resume_layout, _resume_ckpt_path, _resume_ckpt = resolve_resume_checkpoint(
            cfg.resume_from,
            prefer=str(getattr(cfg, "resume_checkpoint", "last")),
            device=device,
            trust_artifact=bool(getattr(cfg, "trust_artifact", False)),
        )
        if _resume_layout.run_dir != layout.run_dir:
            logger.warning(
                f"[resume] --out ({layout.run_dir}) differs from the resumed run "
                f"({_resume_layout.run_dir}); writing to the resumed run directory."
            )
            layout = ensure_run_layout(_resume_layout.run_dir)
            outdir = layout.run_dir
            cfg.out = str(outdir)
        _resume_prev_manifest = read_run_manifest(layout) or {}
        run_created_at = str(_resume_prev_manifest.get("created_at_utc") or run_created_at)

        # Load previous resolved config (prefer config.json; fall back to ckpt).
        _prev_cfg: dict[str, Any] = {}
        if layout.config_json.is_file():
            try:
                _prev_cfg = json.loads(layout.config_json.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"[resume] could not read previous config.json: {exc}")
        if not _prev_cfg:
            _prev_cfg = dict(_resume_ckpt.get("config") or {})

        # Lock architecture / encoding / scaler-math / dataset-identity fields so
        # the rebuilt model + scaler reproduce the checkpoint exactly. Optimization,
        # LR schedule, and the epoch target are intentionally taken from THIS run.
        _RESUME_LOCKED_FIELDS = (
            "activation", "hidden", "depth", "dropout", "w0_first", "w0_hidden",
            "use_residual_blocks", "n_bands", "multiscale_mode",
            "use_fourier", "fourier_append_raw", "fourier_n_features", "fourier_sigma", "fourier_seed",
            "use_sh_encoding", "sh_encoding_degree", "sh_append_raw",
            "use_radial_separation", "radial_append_raw",
            "use_radial_decay_encoding", "radial_decay_max_power", "radial_decay_append_raw",
            "use_real_sh_basis", "real_sh_degree", "real_sh_append_raw", "real_sh_include_radial",
            "u_scale_mode", "a_scale_mode", "target_scale_multiplier",
            "use_si", "dataset_name", "split_seed", "val_ratio", "seed",
        )
        _locked_changed = []
        for _f in _RESUME_LOCKED_FIELDS:
            if _f in _prev_cfg and hasattr(cfg, _f):
                if getattr(cfg, _f) != _prev_cfg.get(_f):
                    _locked_changed.append(_f)
                setattr(cfg, _f, _prev_cfg.get(_f))
        if _locked_changed:
            logger.info(f"[resume] locked architecture/scaler fields to previous run: {_locked_changed}")

        # Continuation counters from the checkpoint.
        start_epoch = int(_resume_ckpt.get("epoch", -1)) + 1
        _ck_cfg = dict(_resume_ckpt.get("config") or {})
        _bs = _ck_cfg.get("best_score", _ck_cfg.get("best_val_loss"))
        try:
            _resume_best_val = float(_bs) if (_bs is not None and math.isfinite(float(_bs))) else float("inf")
        except (TypeError, ValueError):
            _resume_best_val = float("inf")
        _be = _ck_cfg.get("best_epoch")
        try:
            _resume_best_epoch = (int(_be) - 1) if (_be is not None and int(_be) > 0) else -1
        except (TypeError, ValueError):
            _resume_best_epoch = -1
        try:
            _resume_epochs_without_improve = int(_ck_cfg.get("epochs_since_improvement", 0) or 0)
        except (TypeError, ValueError):
            _resume_epochs_without_improve = 0
        try:
            _resume_global_step = int(_resume_ckpt.get("global_step") or 0)
        except (TypeError, ValueError):
            _resume_global_step = 0

        if int(cfg.epochs) <= int(start_epoch):
            raise ValueError(
                f"--epochs ({cfg.epochs}) must be greater than the last completed epoch "
                f"({start_epoch}) to resume. --epochs is the TOTAL target epoch count; "
                f"pass e.g. --epochs {start_epoch + 100}."
            )
        logger.info(
            f"[resume] checkpoint={_resume_ckpt_path} | last_completed_epoch={start_epoch} "
            f"| resuming at epoch {start_epoch + 1} | target_epochs={cfg.epochs} "
            f"| strict={bool(getattr(cfg, 'resume_strict', True))}"
        )

    _log_section(
        "ST-LRPS Training",
        {
            "run_dir": outdir,
            "created_at_utc": run_created_at,
            "physics_design": "scalar residual potential dU; acceleration from autograd grad(dU)",
        },
    )
    _log_section(
        "Runtime / Device",
        {
            "device": device.type.upper(),
            "seed": cfg.seed,
            "torch_dtype": str(DTYPE).replace("torch.", ""),
        },
    )
    _warn_batch_size_for_vram(device, cfg)
    if device.type == "cuda":
        logger.info(
            "CUDA memory log format: cuda_mem=current_allocated/current_reserved MiB, "
            "peak=peak_allocated/peak_reserved MiB since phase start, total=physical VRAM. "
            "Values are PyTorch allocator memory, not nvidia-smi process memory."
        )

    # Effective configuration summary (so the active feature set is unambiguous in
    # the log, especially now that several features default ON).
    _grad_accum = int(getattr(cfg, "grad_accumulation_steps", 1))
    _log_section(
        "Loss Configuration",
        {
            "optim.lr": f"{cfg.lr:g}",
            "optim.weight_decay": f"{cfg.weight_decay:g}",
            "batch_size": cfg.batch_size,
            "grad_accumulation_steps": _grad_accum,
            "effective_batch": cfg.batch_size * _grad_accum,
            "accel_ramp_epochs": cfg.accel_ramp_epochs,
            "accel_min_factor": getattr(cfg, "accel_min_factor", 0.05),
            "direction.weight": cfg.direction_loss_weight,
            "direction.start_epoch": cfg.direction_loss_start_epoch,
            "direction.ramp_epochs": cfg.direction_loss_ramp_epochs,
            "direction.floor_abs": f"{cfg.direction_loss_floor_abs:g}",
            "altitude_balanced": bool(cfg.use_altitude_balanced_loss),
            "radial_cross": f"{bool(cfg.use_radial_cross_loss)} (radial={cfg.radial_loss_weight}, cross={cfg.cross_loss_weight})",
        },
    )

    command_line = write_command_txt(layout)
    _manifest_dict = {
        "schema_version": "st_lrps_run_manifest_v1",
        "run_id": outdir.name,
        "created_at_utc": run_created_at,
        "status": "running",
        "command_line": command_line,
        "command_path": str(layout.command_txt),
        "script_version": "st_lrps_engine",
        "git_commit": resolve_git_commit(),
        "data_paths": {
            "data": str(cfg.data),
            "train_data": str(cfg.train_data) if cfg.train_data else None,
            "val_data": str(cfg.val_data) if cfg.val_data else None,
            "test_data": str(cfg.test_data) if cfg.test_data else None,
            "ood_data": str(cfg.ood_data) if cfg.ood_data else None,
            "suite_manifest": str(cfg.suite_manifest) if cfg.suite_manifest else None,
        },
        "config_path": str(layout.config_json),
        "scaler_path": str(layout.scaler_json),
        "best_checkpoint_path": str(layout.ckpt_best),
        "last_checkpoint_path": str(layout.ckpt_last),
        "history_csv_path": str(layout.history_csv),
        "history_jsonl_path": str(layout.history_jsonl),
        "determinism": _determinism_flags,
        "warnings": [],
        "evaluations": [],
    }
    if _resume_requested:
        # Preserve the original run manifest (run_id, created_at, evaluations);
        # record the resume event and set status back to "running".
        _resumed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            _resume_cmd_path = layout.provenance_dir / (
                "resume_command_" + time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + ".txt"
            )
            _resume_cmd_path.write_text(command_line + "\n", encoding="utf-8")
        except Exception as _rc_exc:  # pragma: no cover - defensive
            logger.warning(f"[resume] could not write resume command provenance: {_rc_exc}")
        update_run_manifest(
            layout,
            {
                "status": "running",
                "command_line": command_line,
                "command_path": str(layout.command_txt),
                "resumed": True,
                "resumed_from_checkpoint": str(_resume_ckpt_path),
                "resume_start_epoch": int(start_epoch) + 1,
                "previous_latest_epoch": int(
                    _resume_ckpt.get("epoch_display", start_epoch) or start_epoch
                ),
                "target_epochs": int(cfg.epochs),
                "resumed_at_utc": _resumed_at,
            },
        )
    else:
        write_run_manifest(layout, _manifest_dict)
    _log_section(
        "Output Artifacts",
        {
            "config_json": layout.config_json,
            "scaler_json": layout.scaler_json,
            "train_log": layout.train_log,
            "history_csv": layout.history_csv,
            "history_jsonl": layout.history_jsonl,
            "ckpt_best": layout.ckpt_best,
            "ckpt_last": layout.ckpt_last,
        },
    )

    if cfg.quick_check:
        # Quick-check runs ONE epoch to verify the pipeline. On resume that means
        # one ADDITIONAL epoch past the last completed one (start_epoch), so the
        # quick-check still does real work instead of an empty range.
        _qc_epochs = (int(start_epoch) + 1) if _resume_requested else 1
        logger.info("=" * 62)
        logger.info("QUICK CHECK MODE: this is not a real training run.")
        logger.info(f"  epochs={_qc_epochs}  max_train_batches=5  max_val_batches=2  log_every=1")
        logger.info("=" * 62)
        cfg.epochs = _qc_epochs
        cfg.log_every = 1
        cfg.max_train_batches = cfg.max_train_batches if cfg.max_train_batches is not None else 5
        cfg.max_val_batches   = cfg.max_val_batches   if cfg.max_val_batches   is not None else 2

    # 2-3. Dataset discovery, metadata (SSOT), contract load + validation
    _dsctx = _load_dataset_context(cfg, layout=layout)
    data_path = _dsctx.data_path
    independent_val = _dsctx.independent_val
    train_data_path = _dsctx.train_data_path
    val_data_path = _dsctx.val_data_path
    primary_path = _dsctx.primary_path
    dset_name = _dsctx.dset_name
    meta = _dsctx.meta
    N = _dsctx.N
    N_train_file = _dsctx.N_train_file
    N_val_file = _dsctx.N_val_file
    bytes_est = _dsctx.bytes_est
    dataset_contract_obj = _dsctx.dataset_contract_obj
    validation_report = _dsctx.validation_report

    # 4. Data Splitting
    _split = _resolve_data_splits(
        cfg,
        meta=meta,
        N=N,
        independent_val=independent_val,
        N_train_file=N_train_file if independent_val else None,
        N_val_file=N_val_file if independent_val else None,
        primary_path=primary_path,
        dset_name=dset_name,
        dataset_contract_obj=dataset_contract_obj,
        validation_report=validation_report,
        layout=layout,
    )
    split_seed = _split.split_seed
    split_policy = _split.split_policy
    train_indices = _split.train_indices
    val_indices = _split.val_indices
    n_train = _split.n_train
    n_val = _split.n_val
    split_manifest = _split.split_manifest
    split_manifest_path = _split.split_manifest_path
    splits = _split.splits

    # 4b. Validate metadata contract
    degree_min_val = int(meta.degree_min) if meta.degree_min is not None else -1
    _effective_target = meta.target_mode or ("residual" if degree_min_val >= 0 else "full")
    dataset_body_name, resolved_mu_si, resolved_r_ref_m = _resolve_lunar_dataset_contract(
        meta, data_path=primary_path,
    )
    # Hard convention guard: reject sign-flipped legacy datasets and other
    # silently-wrong metadata before any training happens.
    validate_training_dataset_convention(
        meta,
        data_path=primary_path,
    )
    if _effective_target == "residual" and degree_min_val < 0:
        raise ValueError(
            "Residual dataset detected (target_mode='residual') but degree_min is missing or < 0. "
            "Check HDF5 metadata: degree_min must be >= 0 for residual datasets."
        )
    if meta.columns is not None:
        _cols_lower = str(meta.columns).lower()
        _has_residual_cols = ("du" in _cols_lower or "dax" in _cols_lower)
        if degree_min_val >= 0 and not _has_residual_cols and "[x,y,z,u,ax,ay,az]" in _cols_lower:
            logger.warning(
                f"Dataset columns are labeled full-field ({meta.columns!r}) but "
                f"degree_min={degree_min_val} >= 0 suggests a residual dataset. "
                "Verify dataset generation parameters."
            )

    _log_dataset_and_model_summary(N, _effective_target, bytes_est, cfg, data_path, dataset_body_name, dset_name, independent_val, meta, n_train, n_val, resolved_mu_si, resolved_r_ref_m, train_data_path, val_data_path)
    # SIREN derivative-training safety check
    if cfg.activation.lower() == "sine":
        if cfg.lr > 5e-4:
            logger.warning(
                f"SIREN+Sobolev stability: lr={cfg.lr:.2e} is high. "
                "Recommended lr <= 5e-4 (1e-4 is safer) for derivative/Sobolev training."
            )
        if float(cfg.output_head_lr_mult) > 1.0:
            logger.warning(
                f"SIREN+Sobolev: output_head_lr_mult={cfg.output_head_lr_mult} > 1.0 can "
                "destabilize the grad(U) output. Recommended value: 1.0."
            )

    _accel_min_fac = float(getattr(cfg, "accel_min_factor", 0.05))
    _log_training_curriculum(_accel_min_fac, cfg)

    # 5. Resolve mu_si
    mu_val = float(resolved_mu_si)
    logger.info(f"Hierarchical base model: mu_si = {mu_val:.6e}")

    # 6. Infer acceleration sign
    if isinstance(cfg.a_sign, str) and cfg.a_sign.lower() == "auto":
        if meta.a_sign_convention is not None:
            _sgn = str(meta.a_sign_convention).strip()
            if _sgn in ("+1", "1"):
                a_sign = 1.0
                logger.info("Acceleration sign from dataset metadata: a_sign=+1.0")
            elif _sgn == "-1":
                a_sign = -1.0
                logger.info("Acceleration sign from dataset metadata: a_sign=-1.0")
            else:
                logger.warning(f"Unrecognised a_sign_convention='{_sgn}'; falling back to auto-inference.")
                a_sign = infer_a_sign_from_data(
                    h5_path=primary_path, dset_name=dset_name, meta=meta,
                    use_si=cfg.use_si, n_probe=50_000, seed=cfg.fit_seed + 777
                )
        else:
            a_sign = infer_a_sign_from_data(
                h5_path=primary_path, dset_name=dset_name, meta=meta,
                use_si=cfg.use_si, n_probe=50_000, seed=cfg.fit_seed + 777
            )
    else:
        a_sign = float(cfg.a_sign)

    target_contract = TargetContract.from_dataset_meta(
        meta,
        resolved_mu_si=float(resolved_mu_si),
        resolved_r_ref_m=float(resolved_r_ref_m),
        a_sign=float(a_sign),
    )
    logger.info(
        "Target contract: "
        f"mode={target_contract.target_mode}, baseline={target_contract.baseline_kind}, "
        f"base_degree={target_contract.base_degree}, target_degree={target_contract.target_degree}, "
        f"frame={target_contract.frame}"
    )

    # 7. Fit isometric scalers on residuals
    scaler = _fit_residual_scalers(
        cfg,
        layout=layout,
        meta=meta,
        primary_path=primary_path,
        dset_name=dset_name,
        mu_val=mu_val,
        a_sign=a_sign,
        degree_min_val=degree_min_val,
        effective_target=_effective_target,
        target_contract=target_contract,
        independent_val=independent_val,
        train_indices=train_indices,
        split_policy=split_policy,
        split_seed=split_seed,
        n_train=n_train,
        n_val=n_val,
        split_manifest=split_manifest,
        dataset_contract_obj=dataset_contract_obj,
    )

    # 7b. Single pre-training preflight gate (go/no-go). Runs after the dataset,
    # splits, and train-only scaler exist but BEFORE model/optimizer build and the
    # training loop, so a failure aborts before any GPU training hours are spent.
    if not bool(getattr(cfg, "skip_preflight", False)):
        _cuda_free: int | None = None
        if device.type == "cuda":
            try:
                _cuda_free = int(torch.cuda.mem_get_info()[0])
            except Exception:  # pragma: no cover - driver/version dependent
                _cuda_free = None
        _preflight = run_preflight(
            out_dir=layout.run_dir,
            deterministic=bool(getattr(cfg, "deterministic", True)),
            benchmark_cudnn=bool(getattr(cfg, "benchmark_cudnn", False)),
            device_type=device.type,
            cuda_free_bytes=_cuda_free,
            validation_report=validation_report,
            splits=(splits if not independent_val else None),
            split_policy=split_policy,
            split_manifest=split_manifest if isinstance(split_manifest, dict) else None,
            scaler_provenance=getattr(scaler, "provenance", None),
            determinism_flags=_determinism_flags,
        )
        _preflight_path = write_preflight_report(layout.provenance_dir, _preflight)
        update_run_manifest(
            layout,
            {
                "preflight_passed": bool(_preflight.go),
                "preflight_report_path": str(_preflight_path),
            },
        )
        logger.info(
            "[preflight] go=%s — %s", _preflight.go, _preflight.summary()
        )
        if not _preflight.go and not bool(getattr(cfg, "allow_preflight_fail", False)):
            raise PreflightError(
                "pre-training preflight gate failed: " + _preflight.summary()
                + " (pass --allow-preflight-fail to record and continue, or "
                "--skip-preflight to bypass)."
            )

    # 8. Construct DataLoaders
    _loaders = _build_dataloaders(
        cfg,
        meta=meta,
        device=device,
        bytes_est=bytes_est,
        N=N,
        independent_val=independent_val,
        data_path=data_path,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        dset_name=dset_name,
        n_train=n_train,
        n_val=n_val,
        train_indices=train_indices,
        val_indices=val_indices,
    )
    train_loader = _loaders.train_loader
    val_loader = _loaders.val_loader
    n_train = _loaders.n_train
    n_val = _loaders.n_val

    # 9. Build model, GradNorm weights, Sobolev loss, and optimizer
    _mb = _build_model_and_optim(
        cfg,
        meta=meta,
        degree_min_val=degree_min_val,
        resolved_r_ref_m=resolved_r_ref_m,
        scaler=scaler,
        device=device,
        a_sign=a_sign,
        mu_val=mu_val,
        target_contract=target_contract,
        _resume_requested=_resume_requested,
        _resume_ckpt=_resume_ckpt,
        _resume_ckpt_path=_resume_ckpt_path,
    )
    model = _mb.model
    opt = _mb.opt
    weights = _mb.weights
    loss_fn = _mb.loss_fn
    _arch_signature = _mb.arch_signature
    degree_max_val = _mb.degree_max_val
    _emb_type_built = _mb.emb_type_built
    _in_fdim_built = _mb.in_fdim_built
    _total_params = _mb.total_params

    # 10. Save canonical config + provenance snapshot
    config_path = layout.config_json
    _is_residual = (_effective_target == "residual")
    dataset_snapshot = _dataset_meta_snapshot(
        meta,
        dataset_name=dset_name,
        data_path=(None if independent_val else data_path),
        train_data_path=(train_data_path if independent_val else None),
        val_data_path=(val_data_path if independent_val else None),
        test_data_path=cfg.test_data,
        ood_data_path=cfg.ood_data,
        target_mode=_effective_target,
        central_body=dataset_body_name,
        resolved_mu_si=resolved_mu_si,
        resolved_r_ref_m=resolved_r_ref_m,
    )
    dataset_snapshot["target_contract"] = target_contract.to_dict()
    dataset_snapshot["dataset_contract"] = dataset_contract_obj.to_dict()
    dataset_snapshot["dataset_validation_report_path"] = str(layout.provenance_dir / "dataset_validation_report.json")
    dataset_snapshot["split_manifest_path"] = str(split_manifest_path)
    dataset_snapshot["split_manifest"] = split_manifest
    dataset_snapshot["dataset_safety"] = {
        "strict_dataset_contract": True,
        "allow_dataset_validation_fail": bool(getattr(cfg, "allow_dataset_validation_fail", False)),
    }
    atomic_write_json(layout.provenance_dir / "dataset_meta.json", dataset_snapshot)
    feature_summary = build_experiment_feature_summary(cfg, target_contract, model)
    atomic_write_json(layout.provenance_dir / "feature_summary.json", feature_summary)

    suite_manifest_path = str(getattr(cfg, "suite_manifest", "") or "").strip()
    suite_manifest: dict[str, Any] = {}
    if suite_manifest_path:
        try:
            manifest_path = Path(suite_manifest_path)
            if manifest_path.exists():
                suite_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                atomic_write_json(layout.provenance_dir / "suite_manifest_snapshot.json", suite_manifest)
            else:
                logger.warning(f"Suite manifest not found while writing config: {suite_manifest_path}")
        except Exception as exc:
            logger.warning(f"Could not read suite manifest while writing config: {exc}")

    capture_environment_snapshot(layout, extra={"device": str(device), "run_id": outdir.name})

    _ckpt_start_res = resolve_best_ckpt_start_epoch(cfg)
    _ckpt_start = _ckpt_start_res.start_epoch
    _ckpt_start_reason = _ckpt_start_res.reason
    if _ckpt_start_res.warning:
        logger.warning(_ckpt_start_res.warning)
    _best_metric_canonical = normalize_best_metric(getattr(cfg, "best_metric", "hybrid"))
    checkpoint_selection = checkpoint_selection_block(
        cfg,
        start_epoch=int(_ckpt_start),
        reason=_ckpt_start_reason,
    )
    _log_section(
        "Checkpoint Selection",
        {
            "best_metric": checkpoint_selection["best_metric"],
            "formula": checkpoint_selection["formula"],
            "start_epoch": checkpoint_selection["start_epoch_display"],
            "reason": checkpoint_selection.get("reason"),
            "lower_is_better": checkpoint_selection["lower_is_better"],
        },
    )
    _bm = str(_best_metric_canonical)
    logger.info(f"  Best-checkpoint metric: {_bm} | formula={checkpoint_selection['formula']}")
    if _bm == "direction_loss":
        logger.warning("best_metric='direction_loss' is experimental. "
                       "Early epochs may select underdeveloped checkpoints. "
                       "Consider 'hybrid' instead.")
    if _bm == "hybrid" and float(getattr(cfg, "direction_loss_weight", 0.0)) == 0.0:
        logger.warning("best_metric='hybrid' selected but direction_loss_weight=0. "
                       "Hybrid score will effectively use val_base_loss. "
                       "Set --direction-loss-weight > 0 to enable hybrid selection.")

    resolved_cfg_source = asdict(cfg)
    resolved_cfg_source.update(
        {
            "resolved_a_sign": float(a_sign),
            "resolved_mu_si": float(mu_val),
            "resolved_r_ref_m": float(resolved_r_ref_m),
            "mu_si": float(mu_val),
            "r_ref_m": float(resolved_r_ref_m),
            "degree_min": int(degree_min_val),
            "degree_max": int(degree_max_val),
            "target_mode": _effective_target,
            "target_contract": target_contract.to_dict(),
            "baseline_kind": target_contract.baseline_kind,
            "base_degree": int(target_contract.base_degree),
            "target_degree": int(target_contract.target_degree),
            "runtime_model_kind": str(getattr(cfg, "runtime_model_kind", "potential_autograd")),
            "residual_mode": _is_residual,
            "unit_system": meta.unit_system,
            "central_body": dataset_body_name,
            "train_data_path": (str(train_data_path) if independent_val else None),
            "val_data_path": (str(val_data_path) if independent_val else None),
            "test_data_path": (str(cfg.test_data) if cfg.test_data else None),
            "ood_data_path": (str(cfg.ood_data) if cfg.ood_data else None),
            "suite_manifest": suite_manifest_path or None,
            "dataset_name": str(dset_name),
            "x_mean_policy": "origin_fixed",
            "x_scale_policy": "max_norm_from_origin",
            "lr_schedule": {
                "kind": "warmup_cosine",
                "warmup_epochs": int(cfg.warmup_epochs),
                "min_lr_ratio": float(cfg.min_lr_ratio),
                "t_max": (int(cfg.t_max) if cfg.t_max is not None else None),
            },
            "loss_features": {
                "altitude_balanced": bool(cfg.use_altitude_balanced_loss),
                "radial_cross": bool(cfg.use_radial_cross_loss),
                "laplacian_regularization": bool(cfg.use_laplacian_regularization),
                "direction_loss_weight": float(cfg.direction_loss_weight),
                "direction_loss_start_epoch": int(cfg.direction_loss_start_epoch),
                "direction_loss_ramp_epochs": int(cfg.direction_loss_ramp_epochs),
                "checkpoint_settle_epochs": int(getattr(cfg, "checkpoint_settle_epochs", 5)),
            },
            "best_val_loss": None,
            "best_epoch": None,
            "best_score": None,
            "best_score_name": str(_best_metric_canonical),
            "best_metric": str(_best_metric_canonical),
            "best_ckpt_start_epoch_resolved": int(_ckpt_start),
            "best_ckpt_start_epoch_resolved_display": int(_ckpt_start + 1),
            "checkpoint_selection": checkpoint_selection,
            "model_builder_version": MODEL_BUILDER_VERSION,
            "embedding_type": _emb_type_built,
            "input_feature_dim": int(_in_fdim_built),
            "x_scale_m": float(scaler.x.scale),
            "total_params": int(_total_params),
            "w0_bands": list(cfg.w0_bands) if cfg.w0_bands is not None else None,
            "dataset_meta": dataset_snapshot,
            "feature_summary": feature_summary,
        }
    )
    if suite_manifest:
        resolved_cfg_source.update(
            {
                "suite_id": suite_manifest.get("suite_id"),
                "suite_name": suite_manifest.get("suite_name"),
                "suite_files": suite_manifest.get("output_files"),
                "suite_component_counts": suite_manifest.get("train_components"),
                "suite_train_total_n": suite_manifest.get("train_total_n"),
                "suite_seeds": {
                    "train_uniform": (suite_manifest.get("train_components") or {}).get("stratified_uniform", {}).get("seed"),
                    "train_inverse_r2": (suite_manifest.get("train_components") or {}).get("inverse_r2", {}).get("seed"),
                    "train_residual_mag": (suite_manifest.get("train_components") or {}).get("residual_mag", {}).get("seed"),
                    "train_boundary": (suite_manifest.get("train_components") or {}).get("boundary", {}).get("seed"),
                    "val": suite_manifest.get("val_seed"),
                    "test": suite_manifest.get("test_seed"),
                    "ood_low": suite_manifest.get("ood_low_seed"),
                    "ood_high": suite_manifest.get("ood_high_seed"),
                },
            }
        )

    payload = build_resolved_config(
        resolved_cfg_source,
        dataset_snapshot,
        model,
        scaler,
        _arch_signature,
        determinism=_determinism_flags,
    )
    atomic_write_json(config_path, payload)
    payload_readback = json.loads(config_path.read_text(encoding="utf-8"))
    verify_critical_config_fields_match(payload_readback, payload)
    if _resume_requested:
        # Compatibility gate: the rebuilt config must agree with the resumed
        # checkpoint on architecture/dataset/scaler-critical fields.
        try:
            verify_critical_config_fields_match(payload_readback, dict(_resume_ckpt.get("config") or {}))
            logger.info("[resume] checkpoint config is compatible with the rebuilt run config.")
        except RuntimeError as _vexc:
            if bool(getattr(cfg, "resume_strict", True)):
                raise RuntimeError(
                    f"[resume] architecture/dataset-critical config mismatch vs checkpoint: {_vexc}"
                ) from _vexc
            logger.warning(f"[resume] non-strict: ignoring config mismatch vs checkpoint: {_vexc}")
    update_run_manifest(
        layout,
        {
            "config_path": str(layout.config_json),
            "resolved_config_summary": {key: payload.get(key) for key in (
                "activation",
                "hidden",
                "depth",
                "n_bands",
                "w0_bands",
                "embedding_type",
                "input_feature_dim",
                "model_preset",
                "run_preset",
                "split_policy",
                "runtime_model_kind",
                "architecture_signature",
                "degree_min",
                "degree_max",
                "target_mode",
                "target_contract",
                "artifact_contract",
                "training_config_hash",
                "run_provenance",
            )},
            "run_provenance": payload.get("run_provenance"),
            "feature_summary_path": str(layout.provenance_dir / "feature_summary.json"),
            "dataset_validation_report_path": str(layout.provenance_dir / "dataset_validation_report.json"),
            "split_manifest_path": str(split_manifest_path),
            "architecture_signature": _arch_signature,
            "artifact_contract": payload.get("artifact_contract"),
            "training_config_hash": payload.get("training_config_hash"),
            "dataset_contract": payload.get("dataset_contract"),
            "split_manifest": split_manifest,
            "dataset_validation_passed": bool(validation_report.get("passed")),
            "dataset_safety_overrides": dataset_snapshot.get("dataset_safety_overrides"),
            "dataset_hash": resolve_dataset_hash(payload.get("dataset_contract")),
            "w0_bands": payload.get("w0_bands"),
            "status": "running",
        },
    )

    return _TrainingSession(
        cfg=cfg,
        model=model,
        opt=opt,
        weights=weights,
        loss_fn=loss_fn,
        device=device,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        layout=layout,
        outdir=outdir,
        config_path=config_path,
        payload=payload,
        payload_readback=payload_readback,
        dataset_snapshot=dataset_snapshot,
        checkpoint_selection=checkpoint_selection,
        dset_name=dset_name,
        resolved_r_ref_m=resolved_r_ref_m,
        start_epoch=start_epoch,
        _arch_signature=_arch_signature,
        _best_metric_canonical=_best_metric_canonical,
        _ckpt_start=_ckpt_start,
        _resume_requested=_resume_requested,
        _resume_ckpt=_resume_ckpt,
        _resume_prev_manifest=_resume_prev_manifest,
        _resume_best_val=_resume_best_val,
        _resume_best_epoch=_resume_best_epoch,
        _resume_epochs_without_improve=_resume_epochs_without_improve,
        _resume_global_step=_resume_global_step,
    )


def _stop_signal_list() -> list[int]:
    """Signals that request a graceful, epoch-level training stop.

    SIGINT (Ctrl+C) and SIGTERM (Slurm walltime/preemption, plain ``kill``)
    everywhere; SIGUSR1 additionally where it exists (POSIX; the target of
    ``sbatch --signal=B:USR1@<sec>`` early-warning configurations).
    """
    import signal as _signal

    sigs: list[int] = [int(_signal.SIGINT), int(_signal.SIGTERM)]
    usr1 = getattr(_signal, "SIGUSR1", None)
    if usr1 is not None:
        sigs.append(int(usr1))
    return sigs


def _install_stop_signal_handlers(handler: Any) -> dict[int, Any]:
    """Register ``handler`` for every stop signal; return the replaced handlers.

    Best-effort: registration requires the main thread (``ValueError``
    otherwise) and a signal may be unsupported on a platform (``OSError``);
    such signals are skipped and absent from the returned mapping, so training
    still runs (interruption then falls back to plain ``KeyboardInterrupt``).
    """
    import signal as _signal

    orig: dict[int, Any] = {}
    for sig in _stop_signal_list():
        try:
            orig[sig] = _signal.signal(sig, handler)
        except (ValueError, OSError):  # pragma: no cover - platform/thread dependent
            continue
    return orig


def _restore_stop_signal_handlers(orig: Mapping[int, Any]) -> None:
    """Best-effort restore of handlers captured by ``_install_stop_signal_handlers``."""
    import signal as _signal

    for sig, old in orig.items():
        try:
            _signal.signal(sig, old)
        except Exception:  # pragma: no cover - platform/thread dependent
            pass


def _run_training_loop(session: _TrainingSession) -> None:
    """Execute the Sobolev training loop (phase 11) for a built session.

    Unpacks the session into the same local names the original ``train()`` body
    used, so the loop logic below is byte-for-byte identical to before the split.
    """
    cfg = session.cfg
    model = session.model
    opt = session.opt
    weights = session.weights
    loss_fn = session.loss_fn
    device = session.device
    scaler = session.scaler
    train_loader = session.train_loader
    val_loader = session.val_loader
    layout = session.layout
    outdir = session.outdir
    config_path = session.config_path
    payload = session.payload
    payload_readback = session.payload_readback
    dataset_snapshot = session.dataset_snapshot
    checkpoint_selection = session.checkpoint_selection
    dset_name = session.dset_name
    resolved_r_ref_m = session.resolved_r_ref_m
    start_epoch = session.start_epoch
    _arch_signature = session._arch_signature
    _best_metric_canonical = session._best_metric_canonical
    _ckpt_start = session._ckpt_start
    _resume_requested = session._resume_requested
    _resume_ckpt = session._resume_ckpt
    _resume_prev_manifest = session._resume_prev_manifest
    _resume_best_val = session._resume_best_val
    _resume_best_epoch = session._resume_best_epoch
    _resume_epochs_without_improve = session._resume_epochs_without_improve
    _resume_global_step = session._resume_global_step

    # 11. Train
    # Resolve collocation altitude bounds — only when a Laplacian is requested.
    # By default no Laplacian is requested, so these stay None and the collocation
    # path is fully skipped (no overhead).
    _col_r_min_m: float | None = None
    _col_r_max_m: float | None = None
    _lap_requested = _laplacian_requested(cfg)
    _col_lmode = str(getattr(cfg, "laplacian_mode", "diagnostic")).strip().lower()
    if _col_lmode not in ("off", "diagnostic", "train"):
        _col_lmode = "diagnostic"
    if not _lap_requested:
        logger.info("Laplacian: not requested (default) — no Laplacian diagnostics or regularization.")
    if _lap_requested and _col_lmode in ("diagnostic", "train"):
        _r_ref_col = float(resolved_r_ref_m)
        _col_alt_min = getattr(cfg, "collocation_alt_min_km", None)
        _col_alt_max = getattr(cfg, "collocation_alt_max_km", None)
        _col_alt_min_resolved = float(_col_alt_min) if _col_alt_min is not None else float(cfg.altitude_min_km)
        _col_alt_max_resolved = float(_col_alt_max) if _col_alt_max is not None else float(cfg.altitude_max_km)
        if _col_lmode == "train":
            if _col_alt_min_resolved is None or _col_alt_max_resolved is None:
                raise ValueError(
                    "laplacian_mode='train' requires collocation altitude bounds to be resolvable. "
                    "Set --collocation-alt-min-km / --collocation-alt-max-km or "
                    "--altitude-min-km / --altitude-max-km."
                )
        _col_r_min_m = _r_ref_col + _col_alt_min_resolved * 1000.0
        _col_r_max_m = _r_ref_col + _col_alt_max_resolved * 1000.0
        logger.info(
            f"Collocation Laplacian: mode={_col_lmode}, r_min={_col_r_min_m:.3e} m, "
            f"r_max={_col_r_max_m:.3e} m "
            f"(alt [{_col_alt_min_resolved:.1f}, {_col_alt_max_resolved:.1f}] km)"
        )

    trainer = STLRPSTrainer(
        model, loss_fn, opt, weights, device, cfg,
        collocation_r_min_m=_col_r_min_m,
        collocation_r_max_m=_col_r_max_m,
        ema_decay=getattr(cfg, "ema_decay", None),
        ema_state_dict=(
            _resume_ckpt.get("ema_state_dict")
            if _resume_requested and isinstance(_resume_ckpt, Mapping)
            else None
        ),
    )

    # Best-checkpoint selection + early-stopping policy (CheckpointManager).
    # The tracker owns the decision state; the local best_* mirrors below are kept
    # in sync after every tracker mutation so the existing payload / manifest /
    # summary read-sites continue to work unchanged.
    tracker = BestCheckpointTracker(start_epoch=int(_ckpt_start), patience=int(cfg.patience))
    best_val = tracker.best_val
    best_epoch = tracker.best_epoch
    epochs_without_improve = tracker.epochs_without_improve
    _prev_val_cossim = 1.0   # for direction drift detection
    _prev_val_mse_a = float("inf")
    best_path = layout.ckpt_best
    last_path = layout.ckpt_last
    _previous_checkpoint_hashes = (
        _resume_prev_manifest.get("checkpoint_hashes", {})
        if isinstance(_resume_prev_manifest, Mapping)
        else {}
    )
    best_ckpt_hash: str | None = (
        str(_previous_checkpoint_hashes.get("best"))
        if isinstance(_previous_checkpoint_hashes, Mapping)
        and _previous_checkpoint_hashes.get("best")
        else None
    )
    last_ckpt_hash: str | None = (
        str(_previous_checkpoint_hashes.get("last"))
        if isinstance(_previous_checkpoint_hashes, Mapping)
        and _previous_checkpoint_hashes.get("last")
        else None
    )
    log_path = layout.history_jsonl
    history: list[dict[str, float]] = []
    _prev_mse_a: float | None = None  # for epoch-level explosion detection
    global_step = 0
    run_status = "completed"
    _loop_t0 = time.perf_counter()
    _total_train_samples = 0  # exact count of samples pushed through fwd/bwd

    if _resume_requested:
        tracker.restore(
            best_val=_resume_best_val,
            best_epoch=_resume_best_epoch,
            epochs_without_improve=_resume_epochs_without_improve,
        )
        best_val = tracker.best_val
        best_epoch = tracker.best_epoch
        epochs_without_improve = tracker.epochs_without_improve
        global_step = _resume_global_step
        logger.info(
            "[resume] restored counters: "
            f"best_epoch={best_epoch + 1 if best_epoch >= 0 else 'none'}, "
            f"best_val={best_val if math.isfinite(best_val) else 'inf'}, "
            f"epochs_without_improve={epochs_without_improve}, global_step={global_step}"
        )

    if _ckpt_start > 0:
        logger.info(
            f"[checkpoint] best tracking starts at epoch {_ckpt_start + 1} "
            "(auto waits until direction-loss training is active, ramped, and settled; "
            "epochs before this are excluded from best-ckpt selection)."
        )
    else:
        logger.info("[checkpoint] best tracking starts at epoch 1.")
    payload["best_ckpt_start_epoch_resolved"] = int(_ckpt_start)
    payload["best_ckpt_start_epoch_resolved_display"] = int(_ckpt_start + 1)
    payload["checkpoint_settle_epochs"] = int(getattr(cfg, "checkpoint_settle_epochs", 5))
    payload["checkpoint_selection"] = dict(checkpoint_selection)
    atomic_write_json(config_path, payload)
    ckpt_config_base = dict(payload)
    update_run_manifest(
        layout,
        {
            "best_ckpt_start_epoch": int(_ckpt_start),
            "best_ckpt_start_epoch_display": int(_ckpt_start + 1),
            "checkpoint_settle_epochs": int(getattr(cfg, "checkpoint_settle_epochs", 5)),
            "checkpoint_selection": dict(checkpoint_selection),
        },
    )
    logger.info("[artifacts] schema=st_lrps_checkpoint_v2")
    logger.info(f"[artifacts] architecture_signature={_arch_signature}")
    logger.info("Beginning training loop...")
    # Restore RNG state and decide history append-vs-overwrite for resume.
    _hist_mode = "w"
    if _resume_requested:
        _rng_state = (_resume_ckpt.get("training_state") or {}).get("rng_state")
        if _rng_state:
            restore_rng_state(_rng_state)
            logger.info("[resume] RNG state restored (epoch-level; DataLoader worker order is not bitwise-guaranteed).")
        else:
            logger.warning("[resume] checkpoint lacks RNG state; continuing with the seeded RNG.")
        if bool(getattr(cfg, "resume_append_history", True)) and log_path.exists():
            try:
                with open(log_path, encoding="utf-8") as _hf:
                    for _line in _hf:
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _prev_row = json.loads(_line)
                        except Exception:
                            continue
                        # Project to the flat row schema (strip nested train/val blocks).
                        history.append({
                            k: v for k, v in _prev_row.items()
                            if k not in ("train", "val", "checkpoint_report")
                        })
                logger.info(f"[resume] loaded {len(history)} previous history rows (append mode).")
                _hist_mode = "a"
            except Exception as _hexc:
                logger.warning(f"[resume] could not read previous history ({_hexc}); writing fresh rows.")
                _hist_mode = "w"
        elif log_path.exists():
            log_path.unlink()
    else:
        if log_path.exists():
            log_path.unlink()

    # Periodic Evaluation During Training (monitoring only; OFF by default).
    # Schedule is resolved from the first epoch that will actually run, so a
    # resume drops already-passed scheduled epochs. Evaluation runs in a separate
    # process after ckpt_last is saved and cannot affect training state.
    periodic_plan = resolve_periodic_eval_plan(cfg, start_epoch=int(start_epoch) + 1)
    _periodic_completed: set = set()
    if periodic_plan.enabled:
        _periodic_completed = completed_periodic_eval_epochs(outdir)
        logger.info(
            f"[periodic-eval] scheduled epochs: {','.join(str(e) for e in periodic_plan.epochs)} "
            f"| dataset={periodic_plan.dataset} checkpoint={periodic_plan.prefer_checkpoint} "
            f"max_samples={periodic_plan.max_samples} device={periodic_plan.device} "
            f"continue_on_fail={periodic_plan.continue_on_fail}"
        )
        if _periodic_completed:
            _already = sorted(e for e in _periodic_completed if e in periodic_plan.epochs_set)
            if _already:
                logger.info(f"[periodic-eval] already recorded (resume) — will skip: {_already}")

    # Graceful, epoch-level interruption. The first stop signal sets a flag;
    # the loop finishes the current epoch, saves ckpt_last, marks the manifest
    # "interrupted", and exits. A second signal restores the original handlers
    # for a hard stop. Beyond Ctrl+C (SIGINT), Slurm delivers SIGTERM at
    # walltime/preemption and sites commonly configure `sbatch
    # --signal=B:USR1@<sec>` as an early warning, so all three take the same
    # graceful path (see hpc/*.sbatch). Installing a handler requires the main
    # thread; if unavailable (e.g. tests in a worker thread) we fall back to
    # plain KeyboardInterrupt.
    import signal as _signal
    _interrupt = {"flag": False}
    _orig_handlers: dict[int, Any] = {}

    def _on_stop_signal(_signum, _frame):  # pragma: no cover - signal timing dependent
        if _interrupt["flag"]:
            _restore_stop_signal_handlers(_orig_handlers)
            raise KeyboardInterrupt
        _interrupt["flag"] = True
        try:
            _name = _signal.Signals(_signum).name
        except ValueError:
            _name = str(_signum)
        logger.warning(
            f"[interrupt] Stop requested ({_name}) — will finish the current epoch, "
            "save, and exit. Send the signal again to force-quit."
        )

    _orig_handlers = _install_stop_signal_handlers(_on_stop_signal)

    with open(log_path, _hist_mode, encoding="utf-8") as logf:
        for epoch in range(start_epoch, cfg.epochs):
            epoch_t0 = time.perf_counter()
            lr_scale = _lr_multiplier_for_epoch(
                epoch,
                total_epochs=cfg.epochs,
                warmup_epochs=cfg.warmup_epochs,
                min_lr_ratio=cfg.min_lr_ratio,
                t_max=cfg.t_max,
            )
            _apply_lr_multiplier(opt, lr_scale)
            _ldir_log = _direction_loss_factor(epoch, cfg)
            if _ldir_log > 0.0 or epoch == cfg.direction_loss_start_epoch:
                logger.info(f"[epoch {epoch+1}] effective lambda_dir={_ldir_log:.4e}")
            tr = trainer.run_epoch(train_loader, is_train=True,  epoch=epoch, max_batches=cfg.max_train_batches)
            _total_train_samples += int(tr.get("samples_seen", 0))

            # Epoch-level explosion detection: save failure manifest and stop on NaN.
            if tr.get("nan_detected"):
                run_status = "failed"
                logger.error(
                    f"Training stopped at epoch {epoch+1} due to NaN/Inf loss. "
                    f"Saving failure manifest to {outdir / 'failure_manifest.json'}."
                )
                atomic_write_json(
                    outdir / "failure_manifest.json",
                    {"epoch": epoch, "reason": "nan_loss", "config": asdict(cfg)},
                )
                update_run_manifest(
                    layout,
                    {
                        "status": "failed",
                        "latest_epoch": int(epoch + 1),
                        "notes": [f"Training stopped due to NaN/Inf loss at epoch {epoch + 1}."],
                    },
                )
                break

            # Warn if acceleration loss jumped 100x vs previous epoch (early explosion signal).
            _cur_mse_a = float(tr.get("mse_a", 0.0))
            if _prev_mse_a is not None and _prev_mse_a > 1e-12 and _cur_mse_a > 100.0 * _prev_mse_a:
                logger.warning(
                    f"Epoch {epoch+1}: acceleration loss jumped {_cur_mse_a/_prev_mse_a:.0f}x "
                    f"({_prev_mse_a:.3e} -> {_cur_mse_a:.3e}). "
                    "Possible derivative instability. Consider: lower lr, ensure accel_min_factor>0, "
                    "lower w0, increase accel_ramp_epochs."
                )
            _prev_mse_a = _cur_mse_a

            va = trainer.run_epoch(val_loader,   is_train=False, epoch=epoch, max_batches=cfg.max_val_batches)
            epoch_time_s = time.perf_counter() - epoch_t0
            global_step += int(tr.get("optimizer_steps", 0))

            # Direction drift warning: magnitude improving but direction metric worsening.
            _val_cossim_now = float(va.get("cossim_mean", 1.0))
            _val_mse_a_now = float(va.get("mse_a", 0.0))
            if (
                epoch > 0
                and float(getattr(cfg, "direction_loss_weight", 0.0)) > 0.0
                and _val_mse_a_now < _prev_val_mse_a * 0.98
                and _val_cossim_now < _prev_val_cossim - 0.005
            ):
                logger.warning(
                    f"Epoch {epoch+1}: val mse_a improved ({_prev_val_mse_a:.3e} → {_val_mse_a_now:.3e}) "
                    f"but direction metric is drifting "
                    f"(cossim: {_prev_val_cossim:.4f} → {_val_cossim_now:.4f}). "
                    "Consider increasing direction_loss_weight or lowering direction_loss_floor_abs."
                )
            _prev_val_cossim = _val_cossim_now
            _prev_val_mse_a = _val_mse_a_now

            _best_metric_mode = str(_best_metric_canonical)
            va["eligible_for_best"] = bool(epoch >= _ckpt_start)
            _ckpt_score, checkpoint_report = compute_checkpoint_score(va, cfg)
            va["val_checkpoint_score"] = float(_ckpt_score)
            va["checkpoint_formula"] = str(checkpoint_report["formula"])
            va["best_metric"] = str(checkpoint_report["best_metric"])

            ckpt_config = dict(ckpt_config_base)
            ckpt_config["best_val_loss"] = float(best_val) if math.isfinite(best_val) else None
            ckpt_config["best_epoch"] = int(best_epoch + 1) if best_epoch >= 0 else None
            ckpt_config["best_score"] = float(best_val) if math.isfinite(best_val) else None
            ckpt_config["best_score_name"] = str(_best_metric_mode)
            ckpt_config["checkpoint_selection"] = dict(checkpoint_selection)
            ckpt_config["checkpoint_report"] = dict(checkpoint_report)
            ckpt_config["best_ckpt_start_epoch_resolved"] = int(_ckpt_start)
            ckpt_config["current_epoch"] = int(epoch + 1)
            ckpt_config["current_val_ref_loss"] = float(va["loss"])
            ckpt_config["epochs_since_improvement"] = int(epochs_without_improve)

            scheduler_state = {
                "kind": "warmup_cosine",
                "epoch": int(epoch),
                "warmup_epochs": int(cfg.warmup_epochs),
                "min_lr_ratio": float(cfg.min_lr_ratio),
                "t_max": (int(cfg.t_max) if cfg.t_max is not None else None),
            }
            checkpoint_train_stats = dict(tr)
            checkpoint_train_stats["gradnorm_status"] = str(getattr(cfg, "gradnorm_mode", "fixed"))
            # Resume state: GradNorm internal state + RNG snapshot so a resumed
            # run continues loss-weighting and randomness from this epoch.
            checkpoint_train_stats["gradnorm_weights"] = weights.state_dict()
            checkpoint_train_stats["rng_state"] = capture_rng_state()
            checkpoint_payload = build_checkpoint_payload(
                kind="last",
                epoch=epoch,
                model=model,
                optimizer=opt,
                scheduler=scheduler_state,
                cfg=ckpt_config,
                scaler=scaler,
                train_stats=checkpoint_train_stats,
                val_stats=va,
                dataset_meta=dataset_snapshot,
                architecture_signature=_arch_signature,
                global_step=global_step,
                ema_state_dict=trainer.ema_state_dict,
            )
            verify_critical_config_fields_match(payload_readback, checkpoint_payload["config"])

            checkpoint_report["is_best_update"] = False
            checkpoint_report["best_epoch"] = int(best_epoch + 1) if best_epoch >= 0 else None
            checkpoint_report["best_score"] = float(best_val) if math.isfinite(best_val) else None
            # Advance the best-checkpoint / patience policy (CheckpointManager),
            # then mirror its state into the local names the rest of the loop reads.
            _best_update = tracker.update(epoch=epoch, score=_ckpt_score)
            best_val = tracker.best_val
            best_epoch = tracker.best_epoch
            epochs_without_improve = tracker.epochs_without_improve
            if not _best_update.eligible:
                # Burn-in phase: save last checkpoint but do not update best or count patience.
                logger.info(f"[checkpoint] waiting: epoch {epoch+1} < start epoch {_ckpt_start + 1}")
                if epoch == _ckpt_start - 1:
                    logger.info(
                        f"[checkpoint] waiting complete: epoch {epoch+1}. "
                        f"Best-checkpoint tracking and patience counter start from next epoch."
                    )
            else:
                if _best_update.is_best:
                    checkpoint_payload["scoring"]["score"] = float(_ckpt_score)
                    checkpoint_payload["config"]["best_val_loss"] = float(best_val)
                    checkpoint_payload["config"]["best_epoch"] = int(best_epoch + 1)
                    checkpoint_payload["config"]["best_score"] = float(_ckpt_score)
                    checkpoint_payload["config"]["best_score_name"] = str(_best_metric_mode)
                    checkpoint_payload["config"]["checkpoint_selection"] = dict(checkpoint_selection)
                    checkpoint_payload["config"]["best_val_base_loss"] = float(va.get("val_base_loss", va.get("mse_u", 0.0) + va.get("mse_a", 0.0)))
                    checkpoint_payload["config"]["best_val_total_loss"] = float(va.get("val_total_loss", va["loss"]))
                    checkpoint_payload["config"]["best_val_physics_loss"] = float(va.get("val_physics_loss", 0.0))
                    checkpoint_payload["config"]["epochs_since_improvement"] = 0
                    checkpoint_report["is_best_update"] = True
                    checkpoint_report["best_epoch"] = int(best_epoch + 1)
                    checkpoint_report["best_score"] = float(best_val)
                    checkpoint_payload["config"]["checkpoint_report"] = dict(checkpoint_report)
                    checkpoint_payload["scoring"].update(dict(checkpoint_report))
                    best_save_info = save_checkpoint(
                        layout,
                        kind="best",
                        payload=checkpoint_payload,
                        epoch=epoch,
                        return_metadata=True,
                    )
                    best_ckpt_hash = str(best_save_info["sha256"])
                    logger.info(f"[artifacts] checkpoint saved: kind=best epoch={epoch + 1}")
                    logger.info(f"[checkpoint] best updated: val_ref={va['loss']:.6e} score={_ckpt_score:.6e} epoch={best_epoch + 1}")
                    update_run_manifest(
                        layout,
                        {
                            "best_checkpoint_path": str(best_path),
                            "last_checkpoint_path": str(last_path),
                            "best_epoch": int(best_epoch + 1),
                            "best_score": float(_ckpt_score),
                            "checkpoint_selection": dict(checkpoint_selection),
                            "checkpoint_report": dict(checkpoint_report),
                            "latest_epoch": int(epoch + 1),
                            "checkpoint_hashes": {
                                "best": best_ckpt_hash,
                                "last": last_ckpt_hash,
                            },
                        },
                    )
                # Non-improving eligible epochs: the tracker already advanced
                # epochs_without_improve (mirrored into the local above).

            checkpoint_payload["config"]["best_val_loss"] = float(best_val) if math.isfinite(best_val) else None
            checkpoint_payload["config"]["best_epoch"] = int(best_epoch + 1) if best_epoch >= 0 else None
            checkpoint_payload["config"]["best_score"] = float(best_val) if math.isfinite(best_val) else None
            checkpoint_payload["config"]["best_score_name"] = str(_best_metric_mode)
            checkpoint_payload["config"]["checkpoint_selection"] = dict(checkpoint_selection)
            checkpoint_report["best_epoch"] = int(best_epoch + 1) if best_epoch >= 0 else None
            checkpoint_report["best_score"] = float(best_val) if math.isfinite(best_val) else None
            checkpoint_payload["config"]["checkpoint_report"] = dict(checkpoint_report)
            checkpoint_payload["scoring"].update(dict(checkpoint_report))
            checkpoint_payload["config"]["epochs_since_improvement"] = int(epochs_without_improve)
            last_save_info = save_checkpoint(
                layout,
                kind="last",
                payload=checkpoint_payload,
                epoch=epoch,
                return_metadata=True,
                write_epoch_snapshot=bool(
                    getattr(cfg, "save_epoch_snapshots", False)
                    and ((epoch + 1) % max(1, int(getattr(cfg, "epoch_snapshot_every", 1))) == 0)
                ),
            )
            last_ckpt_hash = str(last_save_info["sha256"])
            logger.info(f"[artifacts] checkpoint saved: kind=last epoch={epoch + 1}")
            logger.info(f"[checkpoint] last saved: epoch={epoch + 1}")
            update_run_manifest(
                layout,
                {
                    "status": "running",
                    "latest_epoch": int(epoch + 1),
                    "checkpoint_report": dict(checkpoint_report),
                    "checkpoint_selection": dict(checkpoint_selection),
                    "last_checkpoint_path": str(last_path),
                    "checkpoint_hashes": {
                        "best": best_ckpt_hash,
                        "last": last_ckpt_hash,
                    },
                },
            )

            tr_with_time = dict(tr)
            tr_with_time["epoch_time_s"] = float(epoch_time_s)
            row = flatten_epoch_metrics(epoch, tr_with_time, va, checkpoint_report, cfg)
            # Extra compatibility / diagnostics not in the stable core schema.
            row["val_mae_a_vec"] = float(va.get("mae_a_vec", 0.0)) if va.get("mae_a_vec") is not None else None
            row["val_rmse_a_vec"] = float(va.get("rmse_a_vec", 0.0)) if va.get("rmse_a_vec") is not None else None
            history.append(row)
            jsonl_row = dict(row)
            jsonl_row["train"] = dict(tr)
            jsonl_row["val"] = dict(va)
            jsonl_row["checkpoint_report"] = dict(checkpoint_report)
            logf.write(json.dumps(jsonl_row, sort_keys=True, default=str) + "\n")
            logf.flush()

            # Periodic evaluation (monitoring only). Runs AFTER ckpt_last is saved
            # and the history row is written. It is a fully isolated subprocess and
            # cannot change optimizer/scheduler/GradNorm/RNG/model/checkpoint state.
            if periodic_plan.enabled:
                _epoch_1based = int(epoch) + 1
                if (
                    _epoch_1based in periodic_plan.epochs_set
                    and _epoch_1based not in _periodic_completed
                ):
                    _peval_ok = True
                    try:
                        _peval_ok = run_periodic_eval(
                            cfg, outdir, _epoch_1based, periodic_plan,
                            log=logger, dataset_name=dset_name,
                        )
                    except Exception as _peval_exc:  # pragma: no cover - defensive
                        _peval_ok = False
                        logger.warning(f"[periodic-eval] epoch={_epoch_1based} unexpected error: {_peval_exc}")
                    _periodic_completed.add(_epoch_1based)
                    if not _peval_ok and not periodic_plan.continue_on_fail:
                        run_status = "failed"
                        logger.error(
                            f"[periodic-eval] aborting training at epoch {_epoch_1based} "
                            "because continue_on_fail is disabled."
                        )
                        update_run_manifest(
                            layout,
                            {
                                "status": "failed",
                                "latest_epoch": int(epoch + 1),
                                "notes": [
                                    f"Training stopped: periodic evaluation failed at epoch {_epoch_1based} "
                                    "with --periodic-eval-fail-fast."
                                ],
                            },
                        )
                        break

            logger.info(format_epoch_summary(row, total_epochs=int(cfg.epochs)))

            if tracker.should_early_stop():
                logger.info(
                    f"Early stopping triggered after {epochs_without_improve} epochs without validation improvement. "
                    f"Best epoch: {best_epoch + 1} | best_val_loss={best_val:.6e}"
                )
                break

            if _interrupt["flag"]:
                run_status = "interrupted"
                _interrupt_done = int(epoch) + 1
                _resume_hint = f"python -m lunaris.surrogate.st_lrps.training.cli --resume-from {outdir} --epochs {cfg.epochs}"
                logger.warning(
                    f"[interrupt] Stopping after completed epoch {_interrupt_done}. "
                    f"ckpt_last.pt is saved. Resume with:\n  {_resume_hint}"
                )
                update_run_manifest(
                    layout,
                    {
                        "status": "interrupted",
                        "latest_epoch": _interrupt_done,
                        "interrupted_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "resume_hint": _resume_hint,
                    },
                )
                break

    # Restore the original stop-signal handlers (best-effort).
    _restore_stop_signal_handlers(_orig_handlers)

    _write_training_history_csv(history, layout.history_csv)
    _save_training_plots(history, outdir)

    payload["best_val_loss"] = float(best_val) if math.isfinite(best_val) else None
    payload["best_epoch"] = int(best_epoch + 1) if best_epoch >= 0 else None
    payload["best_score"] = float(best_val) if math.isfinite(best_val) else None
    payload["best_score_name"] = str(_best_metric_canonical)
    payload["best_metric"] = str(_best_metric_canonical)
    payload["checkpoint_selection"] = dict(checkpoint_selection)
    atomic_write_json(config_path, payload)

    # Hardware-independent compute accounting (FLOP / petaflop/s-days), measured on
    # the trained network. Recorded for the paper/report so "how much compute this
    # architecture does" is portable across machines, unlike wall-clock or GPU-hours.
    # Measurement must never fail a real training run.
    compute_accounting_dict = None
    try:
        from lunaris.surrogate.st_lrps.training.compute_accounting import (
            build_compute_accounting,
        )

        _probe_dtype = next(model.parameters()).dtype
        _probe_batch = min(int(getattr(cfg, "batch_size", 256) or 256), 256)
        _probe = torch.randn(_probe_batch, 3, device=device, dtype=_probe_dtype)
        _dev_type = getattr(device, "type", None) or str(device).split(":")[0]
        _device_name = (
            torch.cuda.get_device_name(device) if _dev_type == "cuda" else str(device)
        )
        _compute_acc = build_compute_accounting(
            model,
            _probe,
            model_kind=str(getattr(model, "runtime_model_kind", "potential_autograd")),
            total_samples_processed=int(_total_train_samples),
            wall_clock_seconds=float(time.perf_counter() - _loop_t0),
            device=_device_name,
        )
        compute_accounting_dict = _compute_acc.to_manifest_dict()
        logger.info("[compute] %s", _compute_acc.summary_line())
    except Exception as _cexc:  # pragma: no cover - never fail a run on measurement
        logger.warning("[compute] compute accounting skipped: %s", _cexc)

    update_run_manifest(
        layout,
        {
            "status": run_status,
            "best_epoch": int(best_epoch + 1) if best_epoch >= 0 else None,
            "best_score": float(best_val) if math.isfinite(best_val) else None,
            "best_metric": str(_best_metric_canonical),
            "checkpoint_selection": dict(checkpoint_selection),
            "latest_epoch": (int(history[-1]["epoch"]) + 1) if history else 0,
            "checkpoint_hashes": {
                "best": best_ckpt_hash,
                "last": last_ckpt_hash,
            },
            "compute_accounting": compute_accounting_dict,
        },
    )

    eval_suggestion = (
        f"python -m lunaris.surrogate.st_lrps.evaluation.cli --model-dir {outdir} "
        f"--data {cfg.test_data or cfg.val_data or cfg.data} --out {outdir / 'evals' / 'publication_eval'}"
    )
    _log_section(
        "Training Complete",
        {
            "status": run_status,
            "best_epoch": int(best_epoch + 1) if best_epoch >= 0 else "none",
            "best_score": f"{best_val:.6e}" if math.isfinite(best_val) else "none",
            "best_metric": str(_best_metric_canonical),
            "best_formula": checkpoint_selection["formula"],
            "best_checkpoint": best_path if best_path.exists() else "not selected",
            "last_checkpoint": last_path,
            "history_csv": layout.history_csv,
            "history_jsonl": layout.history_jsonl,
            "eval_suggestion": eval_suggestion,
        },
    )


def train(cfg: TrainConfig) -> None:
    """Main execution pipeline: build the training session (setup phases 1-10),
    then run the Sobolev training loop (phase 11)."""
    session = build_training_session(cfg)
    _run_training_loop(session)


# ---------------------------------------------------------------------------
# CLI & Auto-Configuration Helpers
# ---------------------------------------------------------------------------


__all__ = ['STLRPSTrainer', 'train', 'set_seed', 'get_device']
