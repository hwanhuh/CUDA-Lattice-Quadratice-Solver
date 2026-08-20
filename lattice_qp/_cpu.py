"""Sparse CPU backend for relax-and-fix lattice-QP solves."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from ._rounding import ProjectionState, ResolvedRounding


def cpu_pcg(
    H: sparse.csr_matrix,
    rhs: np.ndarray,
    guess: np.ndarray,
    fixed: np.ndarray,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, int, float, bool]:
    """Solve the free-variable projected system with diagonally preconditioned CG."""

    free = ~fixed
    result = np.zeros_like(rhs)
    if not np.any(free):
        return result, 0, 0.0, True
    reduced = H[free][:, free].tocsr()
    reduced_rhs = rhs[free]
    diagonal = reduced.diagonal()
    inverse = np.ones_like(diagonal)
    safe = np.abs(diagonal) > np.finfo(np.float64).tiny
    inverse[safe] = 1.0 / diagonal[safe]
    preconditioner = sparse_linalg.LinearOperator(
        reduced.shape, matvec=lambda value: inverse * value, dtype=np.float64
    )
    iterations = 0

    def count_iteration(_value: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    solved, info = sparse_linalg.cg(
        reduced,
        reduced_rhs,
        x0=guess[free],
        rtol=tolerance,
        atol=0.0,
        maxiter=maximum_iterations,
        M=preconditioner,
        callback=count_iteration,
    )
    result[free] = solved
    actual = reduced_rhs - reduced @ solved
    denominator = max(float(np.linalg.norm(reduced_rhs)), np.finfo(np.float64).tiny)
    residual = float(np.linalg.norm(actual) / denominator)
    return result, iterations, residual, info == 0 and residual <= tolerance


class CpuRelaxAndFixSolver:
    """Stateful CPU relax-and-fix backend with explicit rounding transitions."""

    def __init__(
        self,
        H: sparse.csr_matrix,
        g: np.ndarray,
        integer_indices: np.ndarray,
        lattice_steps: np.ndarray,
        x0: np.ndarray,
        *,
        rounding: ResolvedRounding,
        final_solve: str,
        tolerance: float,
        maximum_iterations: int,
        intermediate_tolerance: float,
        intermediate_maximum_iterations: int,
    ) -> None:
        self.H = H
        self.g = g
        self.integer_indices = integer_indices
        self.lattice_steps = lattice_steps
        self.rounding = rounding
        self.final_solve = final_solve
        self.tolerance = tolerance
        self.maximum_iterations = maximum_iterations
        self.intermediate_tolerance = intermediate_tolerance
        self.intermediate_maximum_iterations = intermediate_maximum_iterations

        self.fixed = np.zeros(H.shape[0], dtype=bool)
        self.fixed_values = np.zeros(H.shape[0], dtype=np.float64)
        self.rhs = g.copy()
        self.solution = x0.copy()
        self.relaxation = np.zeros_like(g)
        self.total_iterations = 0
        self.linear_solves = 0
        self.rounding_batches = 0
        self.continuation_solves = 0
        count = len(integer_indices)
        self.next_active = np.full(count, -1, dtype=np.int64)
        self.previous_active = np.full(count, -1, dtype=np.int64)
        if count:
            self.next_active[:-1] = np.arange(1, count, dtype=np.int64)
            self.previous_active[1:] = np.arange(count - 1, dtype=np.int64)
        self.active_head = 0 if count else -1
        self.active_count = count
        self.projection_state = ProjectionState(rounding, count)

    def _solve_linear_system(
        self, tolerance: float, maximum_iterations: int
    ) -> tuple[float, bool]:
        self.solution, iterations, residual, converged = cpu_pcg(
            self.H,
            self.rhs,
            self.solution,
            self.fixed,
            tolerance,
            maximum_iterations,
        )
        self.total_iterations += iterations
        self.linear_solves += 1
        return residual, converged

    def _commit(self, positions: np.ndarray) -> None:
        variables: list[int] = []
        values: list[float] = []
        for raw_position in positions:
            position = int(raw_position)
            variable = int(self.integer_indices[position])
            if self.fixed[variable]:
                continue
            coordinate = float(self.solution[variable] / self.lattice_steps[position])
            lattice_value = self.projection_state.commit(position, coordinate)
            variables.append(variable)
            values.append(float(lattice_value * self.lattice_steps[position]))
        variable_array = np.asarray(variables, dtype=np.int64)
        value_array = np.asarray(values, dtype=np.float64)
        self.fixed[variable_array] = True
        self.fixed_values[variable_array] = value_array
        if len(variables):
            self.rhs -= np.asarray(self.H[:, variable_array] @ value_array).reshape(-1)
            self.rhs[self.fixed] = 0.0

    def _deactivate(self, positions: np.ndarray) -> None:
        """Remove newly fixed integer positions without rebuilding a mask."""

        for raw_position in positions:
            position = int(raw_position)
            previous = int(self.previous_active[position])
            following = int(self.next_active[position])
            if previous >= 0:
                self.next_active[previous] = following
            else:
                self.active_head = following
            if following >= 0:
                self.previous_active[following] = previous
            self.previous_active[position] = -2
            self.next_active[position] = -2
        self.active_count -= len(positions)

    def _active_positions(self) -> np.ndarray:
        active = np.empty(self.active_count, dtype=np.int64)
        cursor = self.active_head
        for offset in range(self.active_count):
            active[offset] = cursor
            cursor = int(self.next_active[cursor])
        return active

    def _select_batch(self, remaining: np.ndarray) -> np.ndarray:
        values = self.solution[self.integer_indices[remaining]]
        periods = self.lattice_steps[remaining]
        coordinates = values / periods
        proposed = np.asarray(
            [
                self.projection_state.propose(int(position), float(coordinate))
                for position, coordinate in zip(remaining, coordinates)
            ],
            dtype=np.float64,
        )
        residues = np.abs(coordinates - proposed)
        order = np.lexsort((self.integer_indices[remaining], residues))
        selected: list[int] = []
        residue_sum = 0.0
        for ordered in order:
            position = int(remaining[ordered])
            residue = float(residues[ordered])
            if self.rounding.selection == "sequential" and selected:
                break
            if selected and residue_sum + residue > self.rounding.threshold:
                break
            selected.append(position)
            residue_sum += residue
        return np.asarray(selected, dtype=np.int64)

    def _projected_solve(self) -> None:
        residual, converged = self._solve_linear_system(
            self.intermediate_tolerance, self.intermediate_maximum_iterations
        )
        if not converged and residual > 2.0 * self.intermediate_tolerance:
            self.continuation_solves += 1
            residual, converged = self._solve_linear_system(
                self.tolerance, self.maximum_iterations
            )
            if not converged:
                raise RuntimeError(
                    f"CPU projected continuation did not converge ({residual:.3e})"
                )

    def _round_integer_variables(self) -> None:
        if self.rounding.selection == "all":
            if len(self.integer_indices):
                self._commit(np.arange(len(self.integer_indices), dtype=np.int64))
                self.rounding_batches = 1
            return
        while self.active_count:
            selected = self._select_batch(self._active_positions())
            self._commit(selected)
            self._deactivate(selected)
            self.rounding_batches += 1
            if self.active_count:
                self._projected_solve()

    def _finalize(self) -> tuple[float, bool]:
        self.solution[self.fixed] = self.fixed_values[self.fixed]
        if self.final_solve == "pcg" and np.any(~self.fixed):
            residual, converged = self._solve_linear_system(
                self.tolerance, self.maximum_iterations
            )
            self.solution[self.fixed] = self.fixed_values[self.fixed]
            if not converged:
                raise RuntimeError(f"CPU final PCG did not converge ({residual:.3e})")
            return residual, converged
        gradient = self.H @ self.solution - self.g
        gradient[self.fixed] = 0.0
        residual = (
            float(
                np.linalg.norm(gradient)
                / max(np.linalg.norm(self.g[~self.fixed]), np.finfo(np.float64).tiny)
            )
            if np.any(~self.fixed)
            else 0.0
        )
        return residual, self.final_solve == "pcg" or residual <= self.tolerance

    def solve(self) -> dict[str, object]:
        residual, converged = self._solve_linear_system(
            self.tolerance, self.maximum_iterations
        )
        if not converged:
            raise RuntimeError(
                f"CPU continuous relaxation did not converge ({residual:.3e})"
            )
        self.relaxation = self.solution.copy()
        self._round_integer_variables()
        residual, converged = self._finalize()
        cone_values, cone_violations, violation_count, max_violation, cone_feasible = (
            self.projection_state.statistics()
        )
        return {
            "x": self.solution,
            "relaxation_x": self.relaxation,
            "relative_residual": residual,
            "converged": converged,
            "rounding_batches": self.rounding_batches,
            "linear_solves": self.linear_solves,
            "linear_iterations": self.total_iterations,
            "pcg_host_synchronizations": 0,
            "continuation_solves": self.continuation_solves,
            "correction_count": self.projection_state.correction_count,
            "cone_values": cone_values,
            "cone_violations": cone_violations,
            "cone_violation_count": violation_count,
            "cone_max_violation": max_violation,
            "cone_feasible": cone_feasible,
        }


def solve_cpu(
    H: sparse.csr_matrix,
    g: np.ndarray,
    integer_indices: np.ndarray,
    lattice_steps: np.ndarray,
    x0: np.ndarray,
    *,
    rounding: ResolvedRounding,
    final_solve: str,
    tolerance: float,
    maximum_iterations: int,
    intermediate_tolerance: float,
    intermediate_maximum_iterations: int,
) -> dict[str, object]:
    """Run one CPU relax-and-fix solve."""

    return CpuRelaxAndFixSolver(
        H,
        g,
        integer_indices,
        lattice_steps,
        x0,
        rounding=rounding,
        final_solve=final_solve,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
        intermediate_tolerance=intermediate_tolerance,
        intermediate_maximum_iterations=intermediate_maximum_iterations,
    ).solve()


__all__ = ["CpuRelaxAndFixSolver", "cpu_pcg", "solve_cpu"]
