"""Geometry-facing conveniences built on the generic lattice-QP solver."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from scipy import sparse

from .._api import LatticeQPResult, solve_lattice_qp
from ..policies import ConeRoundingPolicy


def _policy_from_endpoints(
    integer_count: int,
    edge_endpoints: np.ndarray,
    base_cones: np.ndarray,
    minimum_cones: np.ndarray,
    edge_orientations: np.ndarray | None,
    *,
    double_cover: bool,
    selection: str,
    projection: str,
    threshold: float | None,
) -> ConeRoundingPolicy:
    endpoints = np.asarray(edge_endpoints)
    if endpoints.shape != (integer_count, 2):
        raise ValueError("edge_endpoints shape must be (len(integer_indices), 2)")
    if endpoints.dtype.kind not in "iub":
        if endpoints.dtype.kind != "f" or not np.isfinite(endpoints).all() or not np.equal(
            endpoints, np.floor(endpoints)
        ).all():
            raise ValueError("edge_endpoints must contain finite cone indices")
    endpoints = np.ascontiguousarray(endpoints, dtype=np.int64)
    base = np.asarray(base_cones)
    minimum = np.asarray(minimum_cones)
    if base.ndim != 1 or minimum.ndim != 1 or len(base) != len(minimum):
        raise ValueError("base_cones and minimum_cones must be matching vectors")
    if np.any(endpoints < 0) or np.any(endpoints >= len(base)):
        raise ValueError("edge_endpoints contains an out-of-range cone index")
    if np.any(endpoints[:, 0] == endpoints[:, 1]):
        raise ValueError("edge_endpoints must contain distinct cone pairs")

    if edge_orientations is None:
        orientations = np.ones(integer_count, dtype=np.int64)
    else:
        orientations = np.asarray(edge_orientations)
        if orientations.shape != (integer_count,) or not np.isin(
            orientations, (-1, 1)
        ).all():
            raise ValueError("edge_orientations must be a +/-1 vector")
        orientations = np.ascontiguousarray(orientations, dtype=np.int64)
    scale = 2 if double_cover else 1
    rows = endpoints.reshape(-1)
    positions = np.repeat(np.arange(integer_count, dtype=np.int64), 2)
    coefficients = np.column_stack((-orientations, orientations)).reshape(-1) * scale
    order = np.lexsort((positions, rows))
    rows = rows[order]
    positions = positions[order]
    coefficients = coefficients[order]
    indptr = np.empty(len(base) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(np.bincount(rows, minlength=len(base)), out=indptr[1:])
    return ConeRoundingPolicy(
        incidence_indptr=indptr,
        incidence_indices=positions,
        incidence_coefficients=coefficients,
        base_cones=base,
        minimum_cones=minimum,
        selection=selection,  # type: ignore[arg-type]
        projection=projection,  # type: ignore[arg-type]
        threshold=threshold,
    )


def solve_cone_miq(
    H: sparse.spmatrix,
    g: np.ndarray,
    *,
    integer_indices: np.ndarray,
    lattice_steps: np.ndarray | float = 1.0,
    cone_policy: ConeRoundingPolicy | None = None,
    edge_endpoints: np.ndarray | None = None,
    cone_endpoints: np.ndarray | None = None,
    edge_orientations: np.ndarray | None = None,
    base_cones: np.ndarray | None = None,
    minimum_cones: np.ndarray | None = None,
    double_cover: bool = False,
    selection: str = "multiple",
    projection: str = "nearest",
    threshold: float | None = None,
    **solver_options: Any,
) -> LatticeQPResult:
    """Solve a cone-aware MIQP using topology-free signed incidence data.

    Pass a :class:`ConeRoundingPolicy` directly, or provide one oriented edge
    per integer position through ``edge_endpoints=(tail_cone, head_cone)``.
    The endpoint form contributes ``-q`` at the tail and ``+q`` at the head;
    ``edge_orientations`` may reverse individual rows.  ``double_cover=True``
    scales these signed coefficients (or a supplied policy's coefficients) by
    two.  All optimization behavior remains in :func:`solve_lattice_qp`.
    This is a single linear lattice-QP solve, not a nonlinear Penner-coordinate
    outer iteration with angle Jacobians, line search, or intrinsic flips.
    """

    if not isinstance(double_cover, (bool, np.bool_)):
        raise TypeError("double_cover must be a boolean")
    if cone_policy is not None and not isinstance(cone_policy, ConeRoundingPolicy):
        raise TypeError("cone_policy must be a ConeRoundingPolicy")
    if cone_policy is not None:
        if any(
            value is not None
            for value in (
                edge_endpoints,
                cone_endpoints,
                edge_orientations,
                base_cones,
                minimum_cones,
            )
        ):
            raise ValueError("cone_policy cannot be combined with endpoint cone data")
        policy = cone_policy
        if double_cover:
            coefficients = np.asarray(policy.incidence_coefficients)
            limit = np.iinfo(np.int64).max // 2
            if coefficients.size and np.any(np.abs(coefficients.astype(object)) > limit):
                raise ValueError(
                    "double-cover incidence coefficients exceed int64 range"
                )
            policy = replace(
                policy,
                incidence_coefficients=(
                    2 * np.asarray(coefficients, dtype=np.int64)
                ),
            )
    else:
        if edge_endpoints is not None and cone_endpoints is not None:
            raise ValueError("provide only one of edge_endpoints or cone_endpoints")
        endpoints = edge_endpoints if edge_endpoints is not None else cone_endpoints
        if endpoints is None or base_cones is None or minimum_cones is None:
            raise ValueError(
                "provide cone_policy or endpoint, base_cones, and minimum_cones data"
            )
        policy = _policy_from_endpoints(
            len(np.asarray(integer_indices).reshape(-1)),
            endpoints,
            base_cones,
            minimum_cones,
            edge_orientations,
            double_cover=bool(double_cover),
            selection=selection,
            projection=projection,
            threshold=threshold,
        )
    if "rounding" in solver_options:
        raise ValueError("solve_cone_miq determines rounding from cone_policy")
    return solve_lattice_qp(
        H,
        g,
        integer_indices=integer_indices,
        lattice_steps=lattice_steps,
        rounding=policy,
        **solver_options,
    )


__all__ = ["solve_cone_miq"]
