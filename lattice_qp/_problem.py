"""Normalized lattice-QP input and validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


def integral_vector(value: object, name: str, *, int32: bool = False) -> np.ndarray:
    """Convert an integer-like vector to a contiguous, bounded integer array."""

    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} shape must be one-dimensional")
    if raw.dtype.kind not in "iub":
        if raw.dtype.kind != "f" or not np.isfinite(raw).all() or not np.equal(
            raw, np.floor(raw)
        ).all():
            raise ValueError(f"{name} must contain finite integers")
    dtype = np.int32 if int32 else np.int64
    lower = np.iinfo(dtype).min
    upper = np.iinfo(dtype).max
    if raw.size and (np.any(raw < lower) or np.any(raw > upper)):
        raise ValueError(f"{name} is outside the {np.dtype(dtype).name} range")
    return np.ascontiguousarray(raw, dtype=dtype)


@dataclass(frozen=True)
class NormalizedProblem:
    """Canonical sparse problem representation consumed by each backend."""

    H: sparse.csr_matrix
    g: np.ndarray
    integer_indices: np.ndarray
    lattice_steps: np.ndarray
    x0: np.ndarray
    block_pairs: np.ndarray

    def objective(self, x: np.ndarray) -> float:
        return float(0.5 * x.dot(self.H @ x) - self.g.dot(x))

    def fixed_mask(self) -> np.ndarray:
        fixed = np.zeros(self.H.shape[0], dtype=bool)
        fixed[self.integer_indices] = True
        return fixed

    def relaxation_residual(self, x: np.ndarray) -> float:
        return float(
            np.linalg.norm(self.H @ x - self.g)
            / max(np.linalg.norm(self.g), np.finfo(np.float64).tiny)
        )

    def projected_residual(self, x: np.ndarray) -> float:
        fixed = self.fixed_mask()
        projected_rhs = self.g.copy()
        if len(self.integer_indices):
            projected_rhs -= np.asarray(
                self.H[:, self.integer_indices] @ x[self.integer_indices]
            ).reshape(-1)
        projected_rhs[fixed] = 0.0
        gradient = np.asarray(self.H @ x - self.g)
        gradient[fixed] = 0.0
        return float(
            np.linalg.norm(gradient)
            / max(np.linalg.norm(projected_rhs), np.finfo(np.float64).tiny)
        )

    def integrality_residual(self, x: np.ndarray) -> float:
        if not len(self.integer_indices):
            return 0.0
        coordinates = x[self.integer_indices] / self.lattice_steps
        return float(np.max(np.abs(coordinates - np.rint(coordinates))))


def normalize_problem(
    H: sparse.spmatrix,
    g: np.ndarray,
    integer_indices: np.ndarray,
    lattice_steps: np.ndarray | float,
    x0: np.ndarray | None,
    block_pairs: np.ndarray | None,
    *,
    check_symmetry: bool,
) -> NormalizedProblem:
    matrix = sparse.csr_matrix(H, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("H must be a square sparse matrix")
    matrix.sum_duplicates()
    matrix.sort_indices()
    if matrix.shape[0] == 0:
        raise ValueError("H must not be empty")
    if not np.isfinite(matrix.data).all():
        raise ValueError("H contains non-finite values")
    if check_symmetry:
        difference = matrix - matrix.T
        scale = max(1.0, float(np.max(np.abs(matrix.data), initial=0.0)))
        error = float(np.max(np.abs(difference.data), initial=0.0))
        if error > 1.0e-11 * scale:
            raise ValueError(f"H must be symmetric (maximum error {error:.3e})")

    linear = np.ascontiguousarray(np.asarray(g, dtype=np.float64).reshape(-1))
    if linear.shape != (matrix.shape[0],) or not np.isfinite(linear).all():
        raise ValueError("g must be a finite vector matching H")

    discrete = np.ascontiguousarray(
        np.asarray(integer_indices, dtype=np.int64).reshape(-1)
    )
    if len(discrete) == 0:
        periods = np.empty(0, dtype=np.float64)
    else:
        if np.any(discrete < 0) or np.any(discrete >= matrix.shape[0]):
            raise ValueError("integer_indices contains an out-of-range index")
        if len(np.unique(discrete)) != len(discrete):
            raise ValueError("integer_indices must be unique")
        raw_periods = np.asarray(lattice_steps, dtype=np.float64)
        if raw_periods.ndim == 0:
            periods = np.full(len(discrete), float(raw_periods), dtype=np.float64)
        else:
            periods = np.ascontiguousarray(raw_periods.reshape(-1))
        if periods.shape != discrete.shape:
            raise ValueError("lattice_steps must be scalar or match integer_indices")
        if not np.isfinite(periods).all() or np.any(periods <= 0.0):
            raise ValueError("lattice_steps must be finite and positive")

    initial = np.zeros(matrix.shape[0], dtype=np.float64) if x0 is None else np.asarray(
        x0, dtype=np.float64
    ).reshape(-1)
    initial = np.ascontiguousarray(initial)
    if initial.shape != linear.shape or not np.isfinite(initial).all():
        raise ValueError("x0 must be a finite vector matching H")

    if block_pairs is None:
        pairs = np.empty((0, 2), dtype=np.int64)
    else:
        pairs = np.ascontiguousarray(np.asarray(block_pairs, dtype=np.int64))
        if pairs.size == 0:
            pairs = np.empty((0, 2), dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("block_pairs must have shape (n, 2)")
        if np.any(pairs < 0) or np.any(pairs >= matrix.shape[0]):
            raise ValueError("block_pairs contains an out-of-range index")
        flattened = pairs.reshape(-1)
        if len(np.unique(flattened)) != len(flattened):
            raise ValueError("block_pairs must be disjoint and contain no self-pairs")

    if matrix.shape[0] > np.iinfo(np.int32).max or matrix.nnz > np.iinfo(np.int32).max:
        raise ValueError("CUDA backend requires int32-sized CSR matrices")
    return NormalizedProblem(matrix, linear, discrete, np.ascontiguousarray(periods), initial, pairs)


__all__ = ["NormalizedProblem", "normalize_problem", "integral_vector"]
