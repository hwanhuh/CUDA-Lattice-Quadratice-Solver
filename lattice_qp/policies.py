"""Serializable rounding policies shared by the CPU and CUDA backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


SelectionPolicy = Literal["sequential", "multiple", "all"]
ProjectionPolicy = Literal["nearest", "floor", "ceil", "away_from_zero"]


@dataclass(frozen=True)
class RoundingPolicy:
    """Choose rounding batches independently from their lattice projection.

    ``sequential`` commits one closest variable per projected solve,
    ``multiple`` commits the closest prefix within ``threshold``, and ``all``
    commits every integer variable in one batch.  A ``None`` threshold uses
    :func:`solve_lattice_qp`'s ``multiple_rounding_threshold`` argument.
    """

    selection: SelectionPolicy = "multiple"
    projection: ProjectionPolicy = "nearest"
    threshold: float | None = None

    def __post_init__(self) -> None:
        if self.selection not in {"sequential", "multiple", "all"}:
            raise ValueError("selection must be 'sequential', 'multiple', or 'all'")
        if self.projection not in {"nearest", "floor", "ceil", "away_from_zero"}:
            raise ValueError(
                "projection must be 'nearest', 'floor', 'ceil', or 'away_from_zero'"
            )
        if self.threshold is not None and (
            not np.isfinite(self.threshold) or self.threshold <= 0.0
        ):
            raise ValueError("threshold must be finite and positive")


@dataclass(frozen=True)
class ConeRoundingPolicy:
    """Stateful lattice projection that enforces lower cone bounds.

    Cone incidence is CSR over *integer positions*: an entry ``(c, j, a)``
    contributes ``a * q[j]`` to cone ``c``, where
    ``q[j] = x[integer_indices[j]] / lattice_steps[j]`` is integral.  Thus the
    representation is independent of mesh topology and directly supports the
    ``+/-2`` coefficients used by double covers.
    """

    incidence_indptr: np.ndarray
    incidence_indices: np.ndarray
    incidence_coefficients: np.ndarray
    base_cones: np.ndarray
    minimum_cones: np.ndarray
    selection: SelectionPolicy = "multiple"
    projection: ProjectionPolicy = "nearest"
    threshold: float | None = None

    def __post_init__(self) -> None:
        if self.selection not in {"sequential", "multiple", "all"}:
            raise ValueError("selection must be 'sequential', 'multiple', or 'all'")
        if self.projection not in {"nearest", "floor", "ceil", "away_from_zero"}:
            raise ValueError(
                "projection must be 'nearest', 'floor', 'ceil', or 'away_from_zero'"
            )
        if self.threshold is not None and (
            not np.isfinite(self.threshold) or self.threshold <= 0.0
        ):
            raise ValueError("threshold must be finite and positive")
        # Keep malformed topology data from crossing the public policy
        # boundary.  Range checks that depend on the problem's integer count
        # remain in the solver's normalized-rounding validation.
        for name, value in (
            ("incidence_indptr", self.incidence_indptr),
            ("incidence_indices", self.incidence_indices),
            ("incidence_coefficients", self.incidence_coefficients),
            ("base_cones", self.base_cones),
            ("minimum_cones", self.minimum_cones),
        ):
            raw = np.asarray(value)
            if raw.ndim != 1:
                raise ValueError(f"{name} shape must be one-dimensional")
            if raw.dtype.kind not in "iub":
                if raw.dtype.kind != "f" or not np.isfinite(raw).all() or not np.equal(
                    raw, np.floor(raw)
                ).all():
                    raise ValueError(f"{name} must contain finite integers")
        indptr = np.asarray(self.incidence_indptr)
        if len(indptr) == 0 or int(indptr[0]) != 0 or np.any(indptr[1:] < indptr[:-1]):
            raise ValueError("incidence_indptr must be monotone and start at zero")
        if len(indptr) != len(np.asarray(self.base_cones)) + 1:
            raise ValueError("incidence_indptr must have len(base_cones) + 1 entries")
        indices = np.asarray(self.incidence_indices)
        coefficients = np.asarray(self.incidence_coefficients)
        if int(indptr[-1]) != len(indices) or len(coefficients) != len(indices):
            raise ValueError("cone incidence CSR array sizes are inconsistent")
        if len(np.asarray(self.base_cones)) != len(np.asarray(self.minimum_cones)):
            raise ValueError("base_cones and minimum_cones must have matching lengths")
        if np.any(indices < 0):
            raise ValueError("incidence_indices must be nonnegative")
        if np.any(coefficients == 0):
            raise ValueError("incidence_coefficients must be nonzero")
        for cone_index in range(len(np.asarray(self.base_cones))):
            begin, end = int(indptr[cone_index]), int(indptr[cone_index + 1])
            if len(np.unique(indices[begin:end])) != end - begin:
                raise ValueError("each cone row must contain unique integer positions")

    @property
    def min_cones(self) -> np.ndarray:
        """Compatibility spelling used by some geometry pipelines."""

        return self.minimum_cones


__all__ = [
    "ConeRoundingPolicy",
    "ProjectionPolicy",
    "RoundingPolicy",
    "SelectionPolicy",
]
