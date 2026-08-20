"""Result models shared by the lattice-QP backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatticeQPResult:
    """A feasible lattice-QP candidate and its numerical certificate."""

    x: np.ndarray
    objective: float
    relaxation_objective: float
    relaxation_relative_residual: float
    relative_residual: float
    integrality_residual: float
    converged: bool
    backend: str
    rounding: str
    final_solve: str
    rounding_batches: int
    linear_solves: int
    linear_iterations: int
    pcg_host_synchronizations: int
    pcg_check_interval: int
    continuation_solves: int
    solve_seconds: float
    cone_correction_count: int = 0
    cone_violation_count: int = 0
    cone_max_violation: int = 0
    cone_feasible: bool = True
    cone_values: np.ndarray | None = None
    cone_violations: np.ndarray | None = None


__all__ = ["LatticeQPResult"]
