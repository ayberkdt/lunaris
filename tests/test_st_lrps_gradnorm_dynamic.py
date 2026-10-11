"""GradNormWeights dynamic-mode update cadence.

``mode="dynamic"`` recomputes the EMA gradient-norm ratio on step 1 and then
every ``update_interval`` steps. The cadence test used
``counter % update_interval != 1``: right for intervals >= 2, but with
``update_interval=1`` ``counter % 1`` is always 0, so every step after the
first was skipped instead of updating every step. ``update_interval=0``
raised ``ZeroDivisionError`` mid-training instead of a config error.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.training.losses import GradNormWeights  # noqa: E402


def _count_updates(monkeypatch, interval: int, steps: int) -> list[int]:
    updated: list[int] = []

    def _fake_ratio(self, loss_u, loss_a, shared_params):
        updated.append(self._step_counter)
        self.last_gradnorm_status = "ok"
        return 1.0

    monkeypatch.setattr(GradNormWeights, "_compute_grad_norm_ratio", _fake_ratio)
    gn = GradNormWeights(mode="dynamic", update_interval=interval)
    dummy = torch.zeros(1)
    for _ in range(steps):
        gn.compute_gradnorm_weights(dummy, dummy, [])
    return updated


def test_update_interval_one_updates_every_step(monkeypatch):
    assert _count_updates(monkeypatch, 1, 5) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("interval", [2, 3, 10])
def test_update_interval_n_updates_at_one_plus_multiples(monkeypatch, interval):
    # Unchanged from the previous behaviour for intervals >= 2.
    steps = 3 * interval + 1
    expected = [s for s in range(1, steps + 1) if s % interval == 1]
    assert _count_updates(monkeypatch, interval, steps) == expected


def test_update_interval_zero_is_rejected():
    with pytest.raises(ValueError, match="update_interval"):
        GradNormWeights(mode="dynamic", update_interval=0)
