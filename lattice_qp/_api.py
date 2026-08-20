"""Public validation, CPU fallback, and result contract."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

try:
    from . import _core
except ImportError:  # pragma: no cover - exercised by source-only installs
    _core = None


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


def _as_problem(
    H: sparse.spmatrix,
    g: np.ndarray,
    integer_indices: np.ndarray,
    lattice_steps: np.ndarray | float,
    x0: np.ndarray | None,
    block_pairs: np.ndarray | None,
    *,
    check_symmetry: bool,
) -> tuple[
    sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
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
    return matrix, linear, discrete, np.ascontiguousarray(periods), initial, pairs


def _objective(H: sparse.csr_matrix, g: np.ndarray, x: np.ndarray) -> float:
    return float(0.5 * x.dot(H @ x) - g.dot(x))


def _cpu_pcg(
    H: sparse.csr_matrix,
    rhs: np.ndarray,
    guess: np.ndarray,
    fixed: np.ndarray,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, int, float, bool]:
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


def _solve_cpu(
    H: sparse.csr_matrix,
    g: np.ndarray,
    integer_indices: np.ndarray,
    lattice_steps: np.ndarray,
    x0: np.ndarray,
    *,
    rounding: str,
    final_solve: str,
    tolerance: float,
    maximum_iterations: int,
    intermediate_tolerance: float,
    intermediate_maximum_iterations: int,
    multiple_rounding_threshold: float,
) -> dict[str, object]:
    fixed = np.zeros(H.shape[0], dtype=bool)
    fixed_values = np.zeros(H.shape[0], dtype=np.float64)
    rhs = g.copy()
    solution, iterations, residual, converged = _cpu_pcg(
        H, rhs, x0, fixed, tolerance, maximum_iterations
    )
    if not converged:
        raise RuntimeError(f"CPU continuous relaxation did not converge ({residual:.3e})")
    relaxation = solution.copy()
    total_iterations = iterations
    linear_solves = 1
    batches = 0
    continuations = 0

    # Keep the integer positions in an intrusive list.  The old implementation
    # rebuilt ``remaining`` with ``flatnonzero`` after every batch, which made
    # removing a batch of k variables an O(n_integer) operation (and also
    # required a second full mask scan to decide whether to continue).  The
    # linked-list order is the original input order; selection still applies
    # the same residue/index lexicographic ordering below.
    integer_count = len(integer_indices)
    next_active = np.full(integer_count, -1, dtype=np.int64)
    previous_active = np.full(integer_count, -1, dtype=np.int64)
    if integer_count:
        next_active[:-1] = np.arange(1, integer_count, dtype=np.int64)
        previous_active[1:] = np.arange(integer_count - 1, dtype=np.int64)
    active_head = 0 if integer_count else -1
    active_count = integer_count

    def snap(position: int) -> float:
        variable = int(integer_indices[position])
        period = lattice_steps[position]
        return float(period * np.floor(solution[variable] / period + 0.5))

    def commit(positions: np.ndarray) -> None:
        nonlocal rhs
        variables = integer_indices[positions]
        values = np.asarray([snap(int(position)) for position in positions])
        newly_fixed = ~fixed[variables]
        variables = variables[newly_fixed]
        values = values[newly_fixed]
        fixed[variables] = True
        fixed_values[variables] = values
        if len(variables):
            rhs -= np.asarray(H[:, variables] @ values).reshape(-1)
            rhs[fixed] = 0.0

    def deactivate(positions: np.ndarray) -> None:
        """Remove newly fixed integer positions without rebuilding a mask."""
        nonlocal active_head, active_count
        for raw_position in positions:
            position = int(raw_position)
            previous = int(previous_active[position])
            following = int(next_active[position])
            if previous >= 0:
                next_active[previous] = following
            else:
                active_head = following
            if following >= 0:
                previous_active[following] = previous
            # Mark removed nodes to make accidental double removal obvious
            # during development; active traversal never visits these nodes.
            previous_active[position] = -2
            next_active[position] = -2
        active_count -= len(positions)

    if rounding == "greedy":
        if len(integer_indices):
            commit(np.arange(len(integer_indices), dtype=np.int64))
            batches = 1
    else:
        active_positions = np.empty(integer_count, dtype=np.int64)
        while active_count:
            # Selection must inspect every active variable, but it no longer
            # scans the full fixed mask or allocates/removes via flatnonzero.
            # The traversal preserves the original integer_indices order.
            cursor = active_head
            for offset in range(active_count):
                active_positions[offset] = cursor
                cursor = int(next_active[cursor])
            remaining = active_positions[:active_count]
            values = solution[integer_indices[remaining]]
            periods = lattice_steps[remaining]
            snapped = periods * np.floor(values / periods + 0.5)
            residues = np.abs(values - snapped) / periods
            order = np.lexsort((integer_indices[remaining], residues))
            selected: list[int] = []
            residue_sum = 0.0
            for ordered in order:
                position = int(remaining[ordered])
                residue = float(residues[ordered])
                if selected and residue_sum + residue > multiple_rounding_threshold:
                    break
                selected.append(position)
                residue_sum += residue
            selected_array = np.asarray(selected, dtype=np.int64)
            commit(selected_array)
            deactivate(selected_array)
            batches += 1
            if active_count:
                solution, count, residual, converged = _cpu_pcg(
                    H,
                    rhs,
                    solution,
                    fixed,
                    intermediate_tolerance,
                    intermediate_maximum_iterations,
                )
                total_iterations += count
                linear_solves += 1
                if not converged and residual > 2.0 * intermediate_tolerance:
                    continuations += 1
                    solution, count, residual, converged = _cpu_pcg(
                        H, rhs, solution, fixed, tolerance, maximum_iterations
                    )
                    total_iterations += count
                    linear_solves += 1
                    if not converged:
                        raise RuntimeError(
                            f"CPU projected continuation did not converge ({residual:.3e})"
                        )

    solution[fixed] = fixed_values[fixed]
    if final_solve == "pcg" and np.any(~fixed):
        solved, count, residual, converged = _cpu_pcg(
            H, rhs, solution, fixed, tolerance, maximum_iterations
        )
        solution[~fixed] = solved[~fixed]
        solution[fixed] = fixed_values[fixed]
        total_iterations += count
        linear_solves += 1
        if not converged:
            raise RuntimeError(f"CPU final PCG did not converge ({residual:.3e})")
    else:
        gradient = H @ solution - g
        gradient[fixed] = 0.0
        residual = float(
            np.linalg.norm(gradient)
            / max(np.linalg.norm(g[~fixed]), np.finfo(np.float64).tiny)
        ) if np.any(~fixed) else 0.0
        converged = final_solve == "pcg" or residual <= tolerance
    return {
        "x": solution,
        "relaxation_x": relaxation,
        "relative_residual": residual,
        "converged": converged,
        "rounding_batches": batches,
        "linear_solves": linear_solves,
        "linear_iterations": total_iterations,
        "pcg_host_synchronizations": 0,
        "continuation_solves": continuations,
    }


def solve_lattice_qp(
    H: sparse.spmatrix,
    g: np.ndarray,
    *,
    integer_indices: np.ndarray,
    lattice_steps: np.ndarray | float = 1.0,
    x0: np.ndarray | None = None,
    rounding: Literal["multiple", "greedy"] = "multiple",
    final_solve: Literal["pcg", "none"] = "pcg",
    block_pairs: np.ndarray | None = None,
    backend: Literal["auto", "cuda", "cpu"] = "auto",
    tolerance: float = 2.0e-6,
    maximum_iterations: int = 20_000,
    intermediate_tolerance: float = 1.0e-3,
    intermediate_maximum_iterations: int = 50,
    multiple_rounding_threshold: float = 0.5,
    pcg_check_interval: int = 4,
    check_symmetry: bool = True,
) -> LatticeQPResult:
    """Solve a sparse convex lattice QP with deterministic relax-and-fix.

    The problem contract is ``min 0.5*x.T@H@x - g.T@x`` with selected
    coordinates constrained to ``x[i] in lattice_steps[i] * Z``. ``H`` must
    be symmetric positive semidefinite and the continuous/projected systems
    must be bounded. Equality constraints must already have been eliminated.
    This is a feasible multiple-rounding heuristic, not an exact MIQP solver.
    """

    if rounding not in {"multiple", "greedy"}:
        raise ValueError("rounding must be 'multiple' or 'greedy'")
    if final_solve not in {"pcg", "none"}:
        raise ValueError("final_solve must be 'pcg' or 'none'")
    if backend not in {"auto", "cuda", "cpu"}:
        raise ValueError("backend must be 'auto', 'cuda', or 'cpu'")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if maximum_iterations < 1 or intermediate_maximum_iterations < 1:
        raise ValueError("iteration limits must be positive")
    if not 0.0 < multiple_rounding_threshold:
        raise ValueError("multiple_rounding_threshold must be positive")
    if isinstance(pcg_check_interval, bool) or not isinstance(
        pcg_check_interval, (int, np.integer)
    ):
        raise ValueError("pcg_check_interval must be an integer in [1, 64]")
    if not 1 <= int(pcg_check_interval) <= 64:
        raise ValueError("pcg_check_interval must be in [1, 64]")
    pcg_check_interval = int(pcg_check_interval)

    matrix, linear, discrete, periods, initial, pairs = _as_problem(
        H,
        g,
        integer_indices,
        lattice_steps,
        x0,
        block_pairs,
        check_symmetry=check_symmetry,
    )
    started = perf_counter()
    selected_backend = backend
    native_error: Exception | None = None
    if backend in {"auto", "cuda"} and _core is not None:
        try:
            native = _core.solve_cuda(
                np.asarray(matrix.indptr, dtype=np.int32),
                np.asarray(matrix.indices, dtype=np.int32),
                np.asarray(matrix.data, dtype=np.float64),
                linear,
                discrete,
                periods,
                initial,
                pairs,
                rounding,
                final_solve,
                tolerance,
                maximum_iterations,
                intermediate_tolerance,
                intermediate_maximum_iterations,
                multiple_rounding_threshold,
                pcg_check_interval,
            )
            selected_backend = "cuda"
        except ValueError:
            raise
        except RuntimeError as error:  # pragma: no cover - hardware dependent
            native_error = error
            if backend == "cuda":
                raise
    elif backend == "cuda":
        raise RuntimeError("lattice_qp CUDA extension is not installed")

    if selected_backend == "cuda":
        native_solution = np.asarray(native["x"], dtype=np.float64)
        native_relaxation = np.asarray(native["relaxation_x"], dtype=np.float64)
        relaxation_residual = float(
            np.linalg.norm(matrix @ native_relaxation - linear)
            / max(np.linalg.norm(linear), np.finfo(np.float64).tiny)
        )
        fixed_mask = np.zeros(matrix.shape[0], dtype=bool)
        fixed_mask[discrete] = True
        projected_rhs = linear.copy()
        if len(discrete):
            projected_rhs -= np.asarray(
                matrix[:, discrete] @ native_solution[discrete]
            ).reshape(-1)
        projected_rhs[fixed_mask] = 0.0
        final_gradient = np.asarray(matrix @ native_solution - linear)
        final_gradient[fixed_mask] = 0.0
        final_residual = float(
            np.linalg.norm(final_gradient)
            / max(np.linalg.norm(projected_rhs), np.finfo(np.float64).tiny)
        )
        host_verified = (
            np.isfinite(relaxation_residual)
            and relaxation_residual <= 1.05 * tolerance
            and np.isfinite(final_residual)
            and (final_solve == "none" or final_residual <= 1.05 * tolerance)
        )
        if not host_verified:
            native_error = RuntimeError(
                "CUDA result failed host true-residual verification: "
                f"relaxation={relaxation_residual:.3e}, final={final_residual:.3e}"
            )
            if backend == "cuda":
                raise native_error
            selected_backend = "auto"

    if selected_backend != "cuda":
        native = _solve_cpu(
            matrix,
            linear,
            discrete,
            periods,
            initial,
            rounding=rounding,
            final_solve=final_solve,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            intermediate_tolerance=intermediate_tolerance,
            intermediate_maximum_iterations=intermediate_maximum_iterations,
            multiple_rounding_threshold=multiple_rounding_threshold,
        )
        selected_backend = "cpu" if native_error is None else "cpu_fallback"

    solution = np.ascontiguousarray(np.asarray(native["x"], dtype=np.float64))
    relaxation = np.ascontiguousarray(
        np.asarray(native["relaxation_x"], dtype=np.float64)
    )
    if len(discrete):
        integrality = float(
            np.max(np.abs(solution[discrete] / periods - np.rint(solution[discrete] / periods)))
        )
    else:
        integrality = 0.0
    relaxation_residual = float(
        np.linalg.norm(matrix @ relaxation - linear)
        / max(np.linalg.norm(linear), np.finfo(np.float64).tiny)
    )
    fixed_mask = np.zeros(matrix.shape[0], dtype=bool)
    fixed_mask[discrete] = True
    projected_rhs = linear.copy()
    if len(discrete):
        projected_rhs -= np.asarray(matrix[:, discrete] @ solution[discrete]).reshape(-1)
    projected_rhs[fixed_mask] = 0.0
    final_gradient = np.asarray(matrix @ solution - linear)
    final_gradient[fixed_mask] = 0.0
    final_residual = float(
        np.linalg.norm(final_gradient)
        / max(np.linalg.norm(projected_rhs), np.finfo(np.float64).tiny)
    )
    return LatticeQPResult(
        x=solution,
        objective=_objective(matrix, linear, solution),
        relaxation_objective=_objective(matrix, linear, relaxation),
        relaxation_relative_residual=relaxation_residual,
        relative_residual=final_residual,
        integrality_residual=integrality,
        converged=bool(native["converged"]),
        backend=selected_backend,
        rounding=rounding,
        final_solve=final_solve,
        rounding_batches=int(native["rounding_batches"]),
        linear_solves=int(native["linear_solves"]),
        linear_iterations=int(native["linear_iterations"]),
        pcg_host_synchronizations=int(native["pcg_host_synchronizations"]),
        pcg_check_interval=pcg_check_interval,
        continuation_solves=int(native["continuation_solves"]),
        solve_seconds=perf_counter() - started,
    )
