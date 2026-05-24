# -*- coding: utf-8 -*-
"""
Gravity model adapter shared by the application and validation tools.

The spherical-harmonics loader and the propagation core evolved with slightly
different attribute names.  This module keeps that boundary explicit: callers
load whichever gravity object is convenient, then normalize it once before
handing it to :class:`core.dynamics.DynamicsEngine`.
"""

from __future__ import annotations

from typing import Any


def _first_attr(obj: Any, *names: str) -> Any:
    """Return the first existing attribute from *names* or raise clearly."""

    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    joined = ", ".join(names)
    raise AttributeError(f"gravity model is missing all of: {joined}")


class GravityModelAdapter:
    """
    Normalize SH gravity-model attributes to the core dynamics contract.

    TODO(ST_LRPS): Remove this adapter once models emit strictly identical structs.

    ``models.spherical_harmonics.GravityModel`` exposes names such as
    ``max_degree`` and ``c_coeffs``.  ``core.dynamics`` expects names such as
    ``degree_max`` and ``Cnm``.  The adapter deliberately does not copy large
    arrays; it only forwards attribute access.
    """

    __slots__ = ("_g",)

    def __init__(self, gravity_model: Any) -> None:
        self._g = gravity_model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._g, name)

    @property
    def degree_max(self) -> int:
        return int(_first_attr(self._g, "degree_max", "max_degree"))

    @property
    def R_ref_m(self) -> float:
        return float(_first_attr(self._g, "R_ref_m", "r_ref", "r_ref_m"))

    @property
    def GM_m3s2(self) -> float:
        return float(_first_attr(self._g, "GM_m3s2", "mu", "gm_m3s2"))

    @property
    def ws(self) -> Any:
        if hasattr(self._g, "ws"):
            return getattr(self._g, "ws")
        if hasattr(self._g, "workspace"):
            return getattr(self._g, "workspace")
        if hasattr(self._g, "make_workspace"):
            return self._g.make_workspace()
        raise AttributeError("gravity model must expose ws, workspace, or make_workspace().")

    def make_workspace(self) -> Any:
        if hasattr(self._g, "make_workspace"):
            return self._g.make_workspace()
        return self.ws

    @property
    def Cnm(self) -> Any:
        return _first_attr(self._g, "Cnm", "c_coeffs")

    @property
    def Snm(self) -> Any:
        return _first_attr(self._g, "Snm", "s_coeffs")

    @property
    def diag(self) -> Any:
        return _first_attr(self._g, "diag", "diag_coeffs")

    @property
    def subdiag(self) -> Any:
        return _first_attr(self._g, "subdiag", "subdiag_coeffs")

    @property
    def A(self) -> Any:
        return _first_attr(self._g, "A", "a_coeffs")

    @property
    def B(self) -> Any:
        return _first_attr(self._g, "B", "b_coeffs")

    @property
    def scale_m(self) -> Any:
        return _first_attr(self._g, "scale_m", "scale_m_table")


def adapt_gravity_model(gravity_model: Any) -> Any:
    """
    Return a gravity model that satisfies ``core.dynamics`` expectations.

    Objects that already expose the full strict contract, including ``ws``, are
    returned unchanged.  Loader objects with equivalent legacy names are wrapped
    in :class:`GravityModelAdapter`.
    """

    required = ("degree_max", "R_ref_m", "GM_m3s2", "Cnm", "Snm", "diag", "subdiag", "A", "B", "scale_m", "ws")
    if all(hasattr(gravity_model, name) for name in required):
        return gravity_model
    return GravityModelAdapter(gravity_model)


__all__ = ["GravityModelAdapter", "adapt_gravity_model"]
