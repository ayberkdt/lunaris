"""``allow_sample_failures`` must be settable on the real batch config.

``CPUBatchPropagator`` reads ``getattr(cfg, "allow_sample_failures", False)``
to NaN-fill a failed sample instead of aborting, but ``BatchPropagationConfig``
(frozen, slots) had no such field, so the branch was unreachable outside tests
that used a ``SimpleNamespace``.
"""

from __future__ import annotations

from lunaris.batch.backend_policy import fallback_forbidden
from lunaris.common.batch_defs import BatchPropagationConfig


def test_allow_sample_failures_is_a_real_field_defaulting_to_false() -> None:
    assert BatchPropagationConfig(n_samples=4).allow_sample_failures is False
    assert BatchPropagationConfig(n_samples=4, allow_sample_failures=True).allow_sample_failures is True


def test_batch_fallback_posture_is_sh_fallback_policy_error() -> None:
    # The only shipped knob for the paper-safe fallback posture of a batch run.
    assert fallback_forbidden(BatchPropagationConfig(n_samples=4, sh_fallback_policy="error"))
    assert not fallback_forbidden(BatchPropagationConfig(n_samples=4))
