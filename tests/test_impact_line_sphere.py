"""1A/1B impact geometry regressions for fixed-step GPU/Torch propagation.

``line_sphere_alpha`` solves ||p0 + alpha*(p1-p0)||^2 = R^2 and returns the
entering root, so the interpolated crossing position lies exactly on the impact
sphere regardless of how oblique or near-tangent the segment is. 1B additionally
requires the hit mask to catch outside-to-outside segments that cross the sphere.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.requires_torch

from lunaris.core.torch_frame import (  # noqa: E402
    line_sphere_alpha,
    line_sphere_intersection,
)

R = 1_737_400.0


def _segments() -> tuple:
    prev = torch.tensor(
        [
            [R + 50_000.0, 0.0, 0.0],          # radial
            [R + 40_000.0, 80_000.0, 0.0],     # oblique (tangential component)
            [0.0, R + 30_000.0, 90_000.0],     # oblique 3D
            [R + 20_000.0, 120_000.0, 0.0],    # shallow / near-tangent crossing
        ],
        dtype=torch.float64,
    )
    curr = torch.tensor(
        [
            [R - 20_000.0, 0.0, 0.0],
            [R - 60_000.0, 90_000.0, 0.0],
            [0.0, R - 50_000.0, 100_000.0],
            [R - 10_000.0, 130_000.0, 0.0],
        ],
        dtype=torch.float64,
    )
    return prev, curr


def test_line_sphere_alpha_places_crossing_exactly_on_sphere() -> None:
    prev, curr = _segments()
    # Precondition: every segment starts outside and ends inside the sphere.
    assert torch.all(prev.norm(dim=1) > R)
    assert torch.all(curr.norm(dim=1) <= R)

    alpha = line_sphere_alpha(prev, curr, R)
    assert torch.all(alpha >= 0.0) and torch.all(alpha <= 1.0)

    cross = prev + alpha.unsqueeze(1) * (curr - prev)
    # Exactly on the impact sphere for every geometry (oblique / near-tangent).
    err = (cross.norm(dim=1) - R).abs()
    assert float(err.max()) < 1.0, f"max on-sphere error = {float(err.max()):.3e} m"


def test_line_sphere_alpha_scalar_radius_tensor_equivalent() -> None:
    prev, curr = _segments()
    a_float = line_sphere_alpha(prev, curr, R)
    a_tensor = line_sphere_alpha(prev, curr, torch.tensor(R, dtype=torch.float64))
    torch.testing.assert_close(a_float, a_tensor)


def test_line_sphere_alpha_degenerate_step_does_not_crash() -> None:
    # Zero-displacement (frozen) rows must not divide by zero; alpha falls back to 1.
    p = torch.tensor([[R + 1000.0, 0.0, 0.0]], dtype=torch.float64)
    alpha = line_sphere_alpha(p, p.clone(), R)
    assert torch.isfinite(alpha).all()
    assert float(alpha[0]) == pytest.approx(1.0)


def test_line_sphere_intersection_catches_outside_to_outside_tunneling() -> None:
    prev = torch.tensor([[R + 50_000.0, 0.25 * R, 0.0]], dtype=torch.float64)
    curr = torch.tensor([[-R - 50_000.0, 0.25 * R, 0.0]], dtype=torch.float64)
    assert float(prev.norm()) > R
    assert float(curr.norm()) > R

    hit, alpha = line_sphere_intersection(prev, curr, R)

    assert bool(hit[0])
    assert 0.0 < float(alpha[0]) < 1.0
    cross = prev + alpha.unsqueeze(1) * (curr - prev)
    assert abs(float(cross.norm()) - R) < 1e-6


def test_line_sphere_intersection_rejects_outside_miss() -> None:
    prev = torch.tensor([[R + 50_000.0, 1.25 * R, 0.0]], dtype=torch.float64)
    curr = torch.tensor([[-R - 50_000.0, 1.25 * R, 0.0]], dtype=torch.float64)

    hit, alpha = line_sphere_intersection(prev, curr, R)

    assert not bool(hit[0])
    assert float(alpha[0]) == pytest.approx(1.0)
