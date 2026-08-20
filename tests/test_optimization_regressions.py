"""Regression coverage for the projected-state and rounding optimizations.

These tests intentionally exercise the public solver contract.  In particular,
the CPU reference below spells out the original full-scan multiple-rounding
policy instead of importing any implementation details from ``_api``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from lattice_qp import solve_lattice_qp


def _reference_rounding(
    H: sparse.csr_matrix,
    g: np.ndarray,
    integer_indices: np.ndarray,
    periods: np.ndarray,
    *,
    threshold: float,
    final_solve: str,
) -> tuple[np.ndarray, int, float]:
    """Reference implementation of the pre-optimization full-scan policy."""

    n = H.shape[0]
    fixed = np.zeros(n, dtype=bool)
    fixed_values = np.zeros(n, dtype=np.float64)
    rhs = np.asarray(g, dtype=np.float64).copy()
    solution = np.asarray(sparse_linalg.spsolve(H, rhs), dtype=np.float64)
    remaining = np.ones(len(integer_indices), dtype=bool)
    batches = 0

    while np.any(remaining):
        positions = np.flatnonzero(remaining)
        values = solution[integer_indices[positions]]
        snapped = periods[positions] * np.floor(
            values / periods[positions] + 0.5
        )
        residues = np.abs(values - snapped) / periods[positions]
        order = np.lexsort((integer_indices[positions], residues))
        selected: list[int] = []
        residue_sum = 0.0
        for ordered in order:
            position = int(positions[ordered])
            residue = float(residues[ordered])
            if selected and residue_sum + residue > threshold:
                break
            selected.append(position)
            residue_sum += residue

        selected_array = np.asarray(selected, dtype=np.int64)
        variables = integer_indices[selected_array]
        values = periods[selected_array] * np.floor(
            solution[variables] / periods[selected_array] + 0.5
        )
        fixed[variables] = True
        fixed_values[variables] = values
        rhs -= np.asarray(H[:, variables] @ values).reshape(-1)
        rhs[fixed] = 0.0
        remaining[selected_array] = False
        batches += 1

        if np.any(remaining):
            free = ~fixed
            solution = np.zeros(n, dtype=np.float64)
            solution[free] = sparse_linalg.spsolve(
                H[free][:, free], rhs[free]
            )
            solution[fixed] = fixed_values[fixed]

    solution[fixed] = fixed_values[fixed]
    if final_solve == "pcg" and np.any(~fixed):
        free = ~fixed
        solution[free] = sparse_linalg.spsolve(H[free][:, free], rhs[free])
        solution[fixed] = fixed_values[fixed]

    gradient = np.asarray(H @ solution - g)
    gradient[fixed] = 0.0
    projected_rhs = np.asarray(g, dtype=np.float64).copy()
    projected_rhs[fixed] = 0.0
    residual = float(
        np.linalg.norm(gradient)
        / max(np.linalg.norm(projected_rhs), np.finfo(np.float64).tiny)
    )
    return solution, batches, residual


def _cpu_fixture() -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    diagonal = np.array([1.0, 2.0, 1.5, 3.0, 0.75, 2.5, 1.25, 4.0, 2.0])
    H = sparse.diags(diagonal, format="csr")
    target = np.array([0.10, -0.75, 1.20, 0.25, -0.30, 0.5, 2.40, -0.125, 0.2])
    integer_indices = np.array([0, 2, 4, 6], dtype=np.int64)
    periods = np.array([1.0, 1.0, 0.5, 2.0], dtype=np.float64)
    return H, np.asarray(H @ target), integer_indices, periods


def test_cpu_rounding_matches_full_scan_reference_and_is_deterministic() -> None:
    H, g, integer_indices, periods = _cpu_fixture()
    reference_x, reference_batches, reference_residual = _reference_rounding(
        H,
        g,
        integer_indices,
        periods,
        threshold=0.5,
        final_solve="pcg",
    )

    for rounding in ("multiple", "greedy"):
        first = solve_lattice_qp(
            H,
            g,
            integer_indices=integer_indices,
            lattice_steps=periods,
            rounding=rounding,
            final_solve="pcg",
            multiple_rounding_threshold=0.5,
            backend="cpu",
        )
        second = solve_lattice_qp(
            H,
            g,
            integer_indices=integer_indices,
            lattice_steps=periods,
            rounding=rounding,
            final_solve="pcg",
            multiple_rounding_threshold=0.5,
            backend="cpu",
        )

        np.testing.assert_array_equal(first.x, second.x)
        assert first.rounding_batches == second.rounding_batches
        assert first.relative_residual == second.relative_residual
        assert first.integrality_residual == 0.0

        if rounding == "multiple":
            np.testing.assert_allclose(first.x, reference_x, rtol=2e-10, atol=2e-10)
            assert first.rounding_batches == reference_batches
            assert first.relative_residual == pytest.approx(
                reference_residual, rel=1e-8, abs=1e-12
            )
            assert first.rounding_batches > 1


def _cuda_available() -> bool:
    try:
        from lattice_qp import _api

        return _api._core is not None and bool(_api._core.cuda_available())
    except (ImportError, RuntimeError, OSError):
        return False


def _cuda_fixture() -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    n = 12
    diagonal = np.full(n, 3.0, dtype=np.float64)
    diagonal[::3] = 4.0
    H = sparse.diags(diagonal, format="lil")
    for index in range(n - 1):
        H[index, index + 1] = -0.35
        H[index + 1, index] = -0.35
    H = H.tocsr()
    target = np.array(
        [0.2, -0.4, 1.3, 0.6, -1.2, 0.1, 2.2, -0.7, 0.4, 1.1, -0.2, 0.5],
        dtype=np.float64,
    )
    integer_indices = np.array([0, 2, 4, 6, 8, 10], dtype=np.int64)
    periods = np.ones(len(integer_indices), dtype=np.float64)
    return H, np.asarray(H @ target), integer_indices, periods


@pytest.mark.skipif(not _cuda_available(), reason="CUDA extension/device unavailable")
def test_cuda_cpu_parity_and_check_interval_consistency() -> None:
    H, g, integer_indices, periods = _cuda_fixture()
    cpu = solve_lattice_qp(
        H,
        g,
        integer_indices=integer_indices,
        lattice_steps=periods,
        rounding="multiple",
        backend="cpu",
        tolerance=2e-7,
        intermediate_tolerance=1e-4,
        pcg_check_interval=4,
    )

    results = []
    for interval in (1, 4, 16):
        result = solve_lattice_qp(
            H,
            g,
            integer_indices=integer_indices,
            lattice_steps=periods,
            rounding="multiple",
            backend="cuda",
            tolerance=2e-7,
            intermediate_tolerance=1e-4,
            pcg_check_interval=interval,
        )
        results.append(result)
        np.testing.assert_allclose(result.x, cpu.x, rtol=2e-5, atol=3e-5)
        assert result.integrality_residual <= 1e-12
        assert result.relative_residual <= 1.05 * 2e-7
        assert result.converged

    for result in results[1:]:
        np.testing.assert_allclose(result.x, results[0].x, rtol=2e-6, atol=3e-6)
        assert result.rounding_batches == results[0].rounding_batches


def _scaled_block_fixture() -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build two scaled 4-node components with deliberately competing edges.

    In the first component, ``(0, 1)`` has coupling 45 and ``(0, 2)`` has
    coupling 80.  A raw-magnitude selector therefore prefers ``(0, 2)``,
    while the normalized scores are 0.2025 versus 0.000064 and select
    ``(0, 1)``.  Node 2's intended partner is node 3 (100 versus 80), so the
    raw mutual-pair pass leaves only ``(2, 3)``.  The second component is the
    same construction at a different scale.  This makes the test sensitive
    to accidentally reverting from algebraic-strength pairing to raw
    magnitude pairing, without relying on wall-clock timing.
    """

    diagonal = np.array(
        [1.0, 1.0e4, 1.0e8, 1.0, 2.0, 2.0e4, 2.0e8, 2.0],
        dtype=np.float64,
    )
    H = sparse.diags(diagonal, format="lil")
    couplings = (
        (0, 1, 45.0),
        (0, 2, 80.0),  # larger raw magnitude, weaker normalized coupling
        (2, 3, 100.0),
        (4, 5, 70.0),
        (4, 6, 120.0),  # larger raw magnitude, weaker normalized coupling
        (6, 7, 160.0),
    )
    for row, column, value in couplings:
        H[row, column] = value
        H[column, row] = value
    H = H.tocsr()
    assert np.all(np.linalg.eigvalsh(H.toarray()) > 0.0)

    target = np.array([0.2, 0.1, -0.3, 0.4, -0.2, 0.3, 0.4, -0.1])
    # No integer variables isolates the PCG preconditioner in this regression.
    integer_indices = np.empty(0, dtype=np.int64)
    periods = np.empty(0, dtype=np.float64)
    explicit_pairs = np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int64)
    return H, np.asarray(H @ target), integer_indices, periods, explicit_pairs


@pytest.mark.skipif(not _cuda_available(), reason="CUDA extension/device unavailable")
def test_cuda_normalized_inferred_blocks_match_explicit_blocks() -> None:
    H, g, integer_indices, periods, explicit_pairs = _scaled_block_fixture()
    common = dict(
        integer_indices=integer_indices,
        lattice_steps=periods,
        rounding="multiple",
        backend="cuda",
        tolerance=2e-7,
        intermediate_tolerance=1e-4,
        pcg_check_interval=4,
    )
    inferred = solve_lattice_qp(H, g, block_pairs=None, **common)
    explicit = solve_lattice_qp(H, g, block_pairs=explicit_pairs, **common)

    np.testing.assert_allclose(inferred.x, explicit.x, rtol=3e-6, atol=3e-6)
    assert inferred.relative_residual <= 1.05 * 2e-7
    assert explicit.relative_residual <= 1.05 * 2e-7
    # With normalized mutual pair selection, inferred blocks are the same
    # algebraic blocks as the explicit reference and converge in the same
    # number of PCG iterations.  A raw-magnitude implementation omits the
    # (0, 1) and (4, 5) blocks and fails this strict regression.
    assert inferred.linear_iterations == explicit.linear_iterations
    assert inferred.rounding_batches == explicit.rounding_batches == 0


@pytest.mark.skipif(not _cuda_available(), reason="CUDA extension/device unavailable")
def test_repeated_cuda_solves_are_stable() -> None:
    H, g, integer_indices, periods = _cuda_fixture()
    kwargs = dict(
        integer_indices=integer_indices,
        lattice_steps=periods,
        rounding="multiple",
        backend="cuda",
        tolerance=2e-7,
        intermediate_tolerance=1e-4,
        pcg_check_interval=4,
    )
    reference = solve_lattice_qp(H, g, **kwargs)
    for _ in range(5):
        result = solve_lattice_qp(H, g, **kwargs)
        np.testing.assert_array_equal(result.x, reference.x)
        assert result.rounding_batches == reference.rounding_batches
        assert result.linear_iterations == reference.linear_iterations
        assert result.relative_residual <= 1.05 * 2e-7
