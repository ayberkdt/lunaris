"""Best-checkpoint selection / early-stopping policy for ST-LRPS training.

This is the ``CheckpointManager`` concern factored out of the monolithic
``engine.train`` loop: the *decision* of when an epoch is a new best, how the
patience counter advances, and when to early-stop. It owns only the small
selection state — ``best_val`` / ``best_epoch`` / ``epochs_without_improve`` —
and is deliberately free of any file IO, logging, or checkpoint-payload mutation
so it can be unit-tested in isolation (no torch, no run directory).

The per-epoch *score* itself is computed elsewhere
(``losses.compute_checkpoint_score``); this tracker only compares scores and
manages the burn-in / patience policy:

* **burn-in** — epochs before ``start_epoch`` are not eligible for best-selection
  and do not advance the patience counter (``best_ckpt_start_epoch`` resolution
  lives in the engine).
* **new best** — among eligible epochs, a strictly lower score updates the best
  and resets patience.
* **early stop** — triggered once ``epochs_without_improve >= patience``.

``best_epoch`` is stored 0-based internally (``-1`` means "no best yet"), matching
the engine's convention; :pyattr:`best_epoch_display` gives the 1-based value used
in logs and manifests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BestUpdate",
    "BestCheckpointTracker",
    "BestCkptStartResolution",
    "resolve_best_ckpt_start_epoch",
]


@dataclass(frozen=True)
class BestCkptStartResolution:
    """Resolved burn-in start for best-checkpoint tracking.

    ``warning`` is non-None only for a manual start earlier than the
    direction-ready epoch; the engine emits it (this resolver stays IO-free so it
    is unit-testable).
    """

    start_epoch: int
    reason: str
    warning: str | None = None


def resolve_best_ckpt_start_epoch(cfg: Any) -> BestCkptStartResolution:
    """Resolve the first epoch eligible for best-checkpoint selection.

    Mirrors the engine policy: ``best_ckpt_start_epoch = -1`` (auto) waits until
    the direction loss has ramped in and settled
    (``direction_loss_start_epoch + direction_loss_ramp_epochs +
    checkpoint_settle_epochs``) when direction loss is active, or starts at 0 when
    it is disabled. A non-negative value is honored verbatim, with a warning when
    it precedes the direction-ready epoch while direction loss is active.
    """
    raw = int(getattr(cfg, "best_ckpt_start_epoch", -1))
    direction_ready = (
        int(getattr(cfg, "direction_loss_start_epoch", 0))
        + int(getattr(cfg, "direction_loss_ramp_epochs", 0))
        + int(getattr(cfg, "checkpoint_settle_epochs", 5))
    )
    direction_weight = float(getattr(cfg, "direction_loss_weight", 0.0))

    if raw < 0:
        if direction_weight > 0.0:
            return BestCkptStartResolution(
                start_epoch=direction_ready,
                reason="waits until direction loss ramp is active and settled",
            )
        return BestCkptStartResolution(
            start_epoch=0,
            reason="direction loss disabled; checkpoint tracking starts immediately",
        )

    warning: str | None = None
    if direction_weight > 0.0 and raw < direction_ready:
        warning = (
            "Manual best_ckpt_start_epoch is earlier than the direction-ready epoch "
            f"({raw} < {direction_ready}). The run is allowed, but auto "
            "mode is safer for production checkpoints."
        )
    return BestCkptStartResolution(
        start_epoch=raw,
        reason="manual best_ckpt_start_epoch",
        warning=warning,
    )


@dataclass(frozen=True)
class BestUpdate:
    """Outcome of feeding one epoch's score to the tracker."""

    eligible: bool                  # epoch was past burn-in (best-selection active)
    is_best: bool                   # this epoch set a new best
    best_val: float                 # best score so far (inf if none)
    best_epoch: int                 # 0-based best epoch (-1 if none)
    epochs_without_improve: int     # consecutive eligible epochs without improvement


class BestCheckpointTracker:
    """Stateful best-checkpoint selection + early-stopping policy.

    Parameters
    ----------
    start_epoch:
        First 0-based epoch eligible for best-selection (the resolved
        ``best_ckpt_start_epoch``). Earlier epochs are burn-in.
    patience:
        Number of consecutive eligible epochs without improvement that triggers
        early stopping.
    """

    def __init__(self, *, start_epoch: int, patience: int) -> None:
        self.start_epoch = int(start_epoch)
        self.patience = int(patience)
        self.best_val: float = float("inf")
        self.best_epoch: int = -1
        self.epochs_without_improve: int = 0

    # --- resume ----------------------------------------------------------------

    def restore(
        self,
        *,
        best_val: float,
        best_epoch: int,
        epochs_without_improve: int,
    ) -> None:
        """Restore counters from a resumed checkpoint."""
        self.best_val = float(best_val)
        self.best_epoch = int(best_epoch)
        self.epochs_without_improve = int(epochs_without_improve)

    # --- queries ---------------------------------------------------------------

    @property
    def has_best(self) -> bool:
        return self.best_epoch >= 0

    @property
    def best_epoch_display(self) -> int | None:
        """1-based best epoch for logs/manifests, or ``None`` if no best yet."""
        return self.best_epoch + 1 if self.best_epoch >= 0 else None

    @property
    def best_val_or_none(self) -> float | None:
        """Best score, or ``None`` when it is still non-finite (no best yet)."""
        return self.best_val if math.isfinite(self.best_val) else None

    def is_burn_in(self, epoch: int) -> bool:
        """True when ``epoch`` (0-based) precedes best-selection eligibility."""
        return int(epoch) < self.start_epoch

    # --- update ----------------------------------------------------------------

    def update(self, *, epoch: int, score: float) -> BestUpdate:
        """Feed one epoch's selection score and advance the policy in place.

        During burn-in (``epoch < start_epoch``) nothing changes: not eligible,
        never a best, patience untouched. Otherwise a strictly lower score sets a
        new best and resets patience; a non-improving score increments patience.
        """
        if self.is_burn_in(epoch):
            return BestUpdate(
                eligible=False,
                is_best=False,
                best_val=self.best_val,
                best_epoch=self.best_epoch,
                epochs_without_improve=self.epochs_without_improve,
            )

        is_best = False
        if float(score) < self.best_val:
            self.best_val = float(score)
            self.best_epoch = int(epoch)
            self.epochs_without_improve = 0
            is_best = True
        else:
            self.epochs_without_improve += 1

        return BestUpdate(
            eligible=True,
            is_best=is_best,
            best_val=self.best_val,
            best_epoch=self.best_epoch,
            epochs_without_improve=self.epochs_without_improve,
        )

    def should_early_stop(self) -> bool:
        """True once patience is exhausted."""
        return self.epochs_without_improve >= self.patience
