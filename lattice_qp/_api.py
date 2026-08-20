"""Public lattice-QP orchestration and backend selection."""

from __future__ import annotations

from time import perf_counter
from typing import Literal

import numpy as np
from scipy import sparse

from ._cpu import solve_cpu
from ._problem import normalize_problem
from ._result import LatticeQPResult
from ._rounding import resolve_rounding
from .policies import ConeRoundingPolicy, RoundingPolicy

try:
    from . import _core
except ImportError:  # pragma: no cover - exercised by source-only installs
    _core = None


def _bounded_integer(value: object, name: str, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [1, {upper}]")
    converted = int(value)
    if not 1 <= converted <= upper:
        raise ValueError(f"{name} must be in [1, {upper}]")
    return converted


def solve_lattice_qp(
    H: sparse.spmatrix,
    g: np.ndarray,
    *,
    integer_indices: np.ndarray,
    lattice_steps: np.ndarray | float = 1.0,
    x0: np.ndarray | None = None,
    rounding: str | RoundingPolicy | ConeRoundingPolicy = "multiple",
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

    if final_solve not in {"pcg", "none"}:
        raise ValueError("final_solve must be 'pcg' or 'none'")
    if backend not in {"auto", "cuda", "cpu"}:
        raise ValueError("backend must be 'auto', 'cuda', or 'cpu'")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not np.isfinite(intermediate_tolerance) or intermediate_tolerance <= 0.0:
        raise ValueError("intermediate_tolerance must be finite and positive")
    int32_maximum = int(np.iinfo(np.int32).max)
    maximum_iterations = _bounded_integer(
        maximum_iterations, "maximum_iterations", int32_maximum
    )
    intermediate_maximum_iterations = _bounded_integer(
        intermediate_maximum_iterations,
        "intermediate_maximum_iterations",
        int32_maximum,
    )
    if not 0.0 < multiple_rounding_threshold:
        raise ValueError("multiple_rounding_threshold must be positive")
    pcg_check_interval = _bounded_integer(
        pcg_check_interval, "pcg_check_interval", 64
    )

    problem = normalize_problem(
        H,
        g,
        integer_indices,
        lattice_steps,
        x0,
        block_pairs,
        check_symmetry=check_symmetry,
    )
    resolved_rounding = resolve_rounding(
        rounding, multiple_rounding_threshold, len(problem.integer_indices)
    )
    started = perf_counter()
    selected_backend = backend
    native_error: Exception | None = None
    if backend in {"auto", "cuda"} and _core is not None:
        try:
            native = _core.solve_cuda(
                np.asarray(problem.H.indptr, dtype=np.int32),
                np.asarray(problem.H.indices, dtype=np.int32),
                np.asarray(problem.H.data, dtype=np.float64),
                problem.g,
                problem.integer_indices,
                problem.lattice_steps,
                problem.x0,
                problem.block_pairs,
                resolved_rounding.selection,
                resolved_rounding.projection,
                final_solve,
                tolerance,
                maximum_iterations,
                intermediate_tolerance,
                intermediate_maximum_iterations,
                resolved_rounding.threshold,
                pcg_check_interval,
                resolved_rounding.incidence_indptr,
                resolved_rounding.incidence_indices,
                resolved_rounding.incidence_coefficients,
                resolved_rounding.base_cones,
                resolved_rounding.minimum_cones,
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
        relaxation_residual = problem.relaxation_residual(native_relaxation)
        final_residual = problem.projected_residual(native_solution)
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
        native = solve_cpu(
            problem.H,
            problem.g,
            problem.integer_indices,
            problem.lattice_steps,
            problem.x0,
            rounding=resolved_rounding,
            final_solve=final_solve,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations,
            intermediate_tolerance=intermediate_tolerance,
            intermediate_maximum_iterations=intermediate_maximum_iterations,
        )
        selected_backend = "cpu" if native_error is None else "cpu_fallback"

    solution = np.ascontiguousarray(np.asarray(native["x"], dtype=np.float64))
    relaxation = np.ascontiguousarray(np.asarray(native["relaxation_x"], dtype=np.float64))
    integrality = problem.integrality_residual(solution)
    relaxation_residual = problem.relaxation_residual(relaxation)
    final_residual = problem.projected_residual(solution)
    return LatticeQPResult(
        x=solution,
        objective=problem.objective(solution),
        relaxation_objective=problem.objective(relaxation),
        relaxation_relative_residual=relaxation_residual,
        relative_residual=final_residual,
        integrality_residual=integrality,
        converged=bool(native["converged"]),
        backend=selected_backend,
        rounding=resolved_rounding.label,
        final_solve=final_solve,
        rounding_batches=int(native["rounding_batches"]),
        linear_solves=int(native["linear_solves"]),
        linear_iterations=int(native["linear_iterations"]),
        pcg_host_synchronizations=int(native["pcg_host_synchronizations"]),
        pcg_check_interval=pcg_check_interval,
        continuation_solves=int(native["continuation_solves"]),
        solve_seconds=perf_counter() - started,
        cone_correction_count=int(native.get("correction_count", 0)),
        cone_violation_count=int(native.get("cone_violation_count", 0)),
        cone_max_violation=int(native.get("cone_max_violation", 0)),
        cone_feasible=bool(native.get("cone_feasible", True)),
        cone_values=np.ascontiguousarray(
            np.asarray(native.get("cone_values", []), dtype=np.int64)
        ),
        cone_violations=np.ascontiguousarray(
            np.asarray(native.get("cone_violations", []), dtype=np.int64)
        ),
    )


__all__ = ["LatticeQPResult", "solve_lattice_qp"]
