"""Frame-contract validation helpers."""

from __future__ import annotations

import numpy as np

SUPPORTED_SIMPLE_FRAMES = {
    "MOON_FIXED",
    "MOON_PA",
    "MOON_ME",
    "MOON_J2000",
    "J2000",
    "ICRF",
}


def require_supported_frame(value: str) -> str:
    frame = str(value).strip().upper()
    if frame not in SUPPORTED_SIMPLE_FRAMES:
        raise ValueError(f"Unsupported frame: {value!r}")
    return frame


def verify_rotation_matrix(matrix: np.ndarray, *, atol: float = 1e-10) -> None:
    rot = np.asarray(matrix, dtype=np.float64)
    if rot.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3,3).")
    ident = rot.T @ rot
    if not np.allclose(ident, np.eye(3), atol=atol, rtol=0.0):
        raise ValueError("Rotation matrix is not orthogonal.")
    if not np.isclose(np.linalg.det(rot), 1.0, atol=atol, rtol=0.0):
        raise ValueError("Rotation matrix determinant is not +1.")

