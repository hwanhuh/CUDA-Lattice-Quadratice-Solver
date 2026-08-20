"""Prepared-state session contract tests."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from scipy import sparse

import lattice_qp
import lattice_qp._session as session_api
from lattice_qp import (
    LatticeQPSession,
    LatticeQPSessionStats,
    RoundingPolicy,
    solve_lattice_qp,
)

from _rounding_policy_helpers import cuda_available, coupled_fixture


def _session_fixture() -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    H, g, indices, periods = coupled_fixture()
    return H, g, indices, periods


def _assert_same_result(left, right) -> None:
    np.testing.assert_array_equal(left.x, right.x)
    np.testing.assert_array_equal(left.cone_values, right.cone_values)
    np.testing.assert_array_equal(left.cone_violations, right.cone_violations)
    for name in (
        "objective",
        "relaxation_objective",
        "relaxation_relative_residual",
        "relative_residual",
        "integrality_residual",
    ):
        assert getattr(left, name) == pytest.approx(getattr(right, name))
    for name in (
        "rounding",
        "final_solve",
        "rounding_batches",
        "linear_solves",
        "linear_iterations",
        "continuation_solves",
        "cone_correction_count",
        "cone_violation_count",
        "cone_max_violation",
        "cone_feasible",
    ):
        assert getattr(left, name) == getattr(right, name)


def test_session_one_shot_parity_and_repeated_rhs_x0_without_state_leakage() -> None:
    H, g1, indices, periods = _session_fixture()
    g2 = np.asarray(H @ np.array([-0.91, 0.42, 1.66, -0.13]))
    options = dict(
        rounding="multiple",
        final_solve="pcg",
        backend="cpu",
        tolerance=2.0e-7,
        intermediate_tolerance=1.0e-4,
        pcg_check_interval=4,
    )
    one_shot = solve_lattice_qp(
        H,
        g1,
        integer_indices=indices,
        lattice_steps=periods,
        x0=np.zeros(H.shape[0]),
        **options,
    )
    with LatticeQPSession(
        H, integer_indices=indices, lattice_steps=periods
    ) as session:
        first = session.solve(g1, x0=np.zeros(H.shape[0]), **options)
        _assert_same_result(first, one_shot)
        second = session.solve(g2, x0=np.zeros(H.shape[0]), **options)
        expected = solve_lattice_qp(
            H,
            g2,
            integer_indices=indices,
            lattice_steps=periods,
            x0=np.zeros(H.shape[0]),
            **options,
        )
        _assert_same_result(second, expected)
        assert session.stats.solve_calls == 2


def test_session_accepts_policy_variations_and_reset_stats() -> None:
    H, g, indices, periods = _session_fixture()
    policy = RoundingPolicy(selection="all", projection="away_from_zero")
    with LatticeQPSession(
        H, integer_indices=indices, lattice_steps=periods
    ) as session:
        result = session.solve(g, rounding=policy, final_solve="none", backend="cpu")
        expected = solve_lattice_qp(
            H,
            g,
            integer_indices=indices,
            lattice_steps=periods,
            rounding=policy,
            final_solve="none",
            backend="cpu",
        )
        _assert_same_result(result, expected)
        assert session.stats.solve_calls == 1
        assert isinstance(session.stats, LatticeQPSessionStats)
        with pytest.raises(AttributeError):
            session.stats.solve_calls = 100  # type: ignore[misc]
        session.reset_stats()
        assert session.stats.solve_calls == 0
        assert session.stats.cuda_system_creations == 0
        assert session.stats.cuda_system_reuses == 0


def test_session_accepts_new_cone_policy_each_call_without_projected_state_leakage() -> None:
    H = sparse.eye(2, format="csr")
    g = np.array([0.2, 0.8])
    indices = np.array([0, 1], dtype=np.int64)
    steps = np.ones(2)
    common = dict(
        incidence_indptr=np.array([0, 2], dtype=np.int64),
        incidence_indices=np.array([0, 1], dtype=np.int64),
        base_cones=np.array([0], dtype=np.int64),
        minimum_cones=np.array([0], dtype=np.int64),
    )
    first_policy = lattice_qp.ConeRoundingPolicy(
        incidence_coefficients=np.array([1, -1], dtype=np.int64),
        selection="sequential",
        **common,
    )
    second_policy = lattice_qp.ConeRoundingPolicy(
        incidence_coefficients=np.array([2, -2], dtype=np.int64),
        selection="all",
        projection="floor",
        **common,
    )
    with LatticeQPSession(
        H, integer_indices=indices, lattice_steps=steps
    ) as session:
        session.solve(g, rounding=first_policy, backend="cpu")
        second = session.solve(g, rounding=second_policy, backend="cpu")
    expected = solve_lattice_qp(
        H,
        g,
        integer_indices=indices,
        lattice_steps=steps,
        rounding=second_policy,
        backend="cpu",
    )
    _assert_same_result(second, expected)


def test_session_owns_problem_arrays_defensively() -> None:
    H, g, indices, periods = _session_fixture()
    original_H = H.copy()
    original_indices = indices.copy()
    original_periods = periods.copy()
    session = LatticeQPSession(
        H, integer_indices=indices, lattice_steps=periods
    )
    H.data[:] = 100.0
    indices[:] = indices[::-1]
    periods[:] = 7.0
    try:
        actual = session.solve(g, backend="cpu")
    finally:
        session.close()
    expected = solve_lattice_qp(
        original_H,
        g,
        integer_indices=original_indices,
        lattice_steps=original_periods,
        backend="cpu",
    )
    _assert_same_result(actual, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "bad"},
        {"final_solve": "bad"},
        {"tolerance": 0.0},
    ],
)
def test_session_rejects_invalid_solve_options(kwargs) -> None:
    H, g, indices, periods = _session_fixture()
    with LatticeQPSession(
        H, integer_indices=indices, lattice_steps=periods
    ) as session:
        with pytest.raises((TypeError, ValueError)):
            session.solve(g, **kwargs)


def test_session_rejects_invalid_problem_and_lifecycle_is_idempotent() -> None:
    H, g, indices, periods = _session_fixture()
    with pytest.raises((TypeError, ValueError)):
        LatticeQPSession(H, integer_indices=np.array([H.shape[0]]))
    session = LatticeQPSession(H, integer_indices=indices, lattice_steps=periods)
    session.close()
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.solve(g, backend="cpu")


@pytest.mark.parametrize("argument", [np.array([1.0]), np.array([np.nan] * 4)])
def test_session_rejects_invalid_rhs(argument) -> None:
    H, _, indices, periods = _session_fixture()
    with LatticeQPSession(
        H, integer_indices=indices, lattice_steps=periods
    ) as session:
        with pytest.raises((TypeError, ValueError)):
            session.solve(argument, backend="cpu")


def test_auto_cpu_fallback_does_not_report_cuda_resource_reuse(monkeypatch) -> None:
    H, g, indices, periods = _session_fixture()

    class NoDeviceCore:
        def __init__(self) -> None:
            self.availability_calls = 0

        def cuda_available(self) -> bool:
            self.availability_calls += 1
            return False

    fake_core = NoDeviceCore()
    monkeypatch.setattr(session_api, "_core", fake_core)
    with LatticeQPSession(
        H, integer_indices=indices, lattice_steps=periods
    ) as session:
        first = session.solve(g, backend="auto")
        second = session.solve(g, backend="auto")
        assert first.backend == second.backend == "cpu_fallback"
        assert fake_core.availability_calls == 1
        assert session.stats.cuda_system_creations == 0
        assert session.stats.cuda_system_reuses == 0


def test_transient_cuda_initialization_error_is_retried(monkeypatch) -> None:
    H, g, indices, periods = _session_fixture()

    class FailingCore:
        def __init__(self) -> None:
            self.availability_calls = 0
            self.preparation_calls = 0

        def cuda_available(self) -> bool:
            self.availability_calls += 1
            return True

        def PreparedCudaSolver(self, *args) -> None:  # noqa: N802
            self.preparation_calls += 1
            raise RuntimeError("transient CUDA allocation failure")

    fake_core = FailingCore()
    monkeypatch.setattr(session_api, "_core", fake_core)
    with LatticeQPSession(
        H, integer_indices=indices, lattice_steps=periods
    ) as session:
        first = session.solve(g, backend="auto")
        second = session.solve(g, backend="auto")
        assert first.backend == second.backend == "cpu_fallback"
        assert fake_core.availability_calls == 2
        assert fake_core.preparation_calls == 2
        assert session.stats.cuda_system_creations == 0
        assert session.stats.cuda_system_reuses == 0


def test_overlapping_session_solves_are_rejected_deterministically(monkeypatch) -> None:
    H, g, indices, periods = _session_fixture()
    entered = threading.Event()
    release = threading.Event()
    original = session_api._solve_normalized_problem

    def blocked_solve(*args, **kwargs):
        entered.set()
        if not release.wait(5.0):
            raise AssertionError("test did not release the first solve")
        return original(*args, **kwargs)

    monkeypatch.setattr(session_api, "_solve_normalized_problem", blocked_solve)
    session = LatticeQPSession(
        H, integer_indices=indices, lattice_steps=periods
    )
    errors: list[BaseException] = []

    def run_first() -> None:
        try:
            session.solve(g, backend="cpu")
        except BaseException as error:  # pragma: no cover - assertion below
            errors.append(error)

    worker = threading.Thread(target=run_first)
    worker.start()
    assert entered.wait(5.0)
    try:
        with pytest.raises(RuntimeError, match="concurrent|in progress|busy"):
            session.solve(g, backend="cpu")
    finally:
        release.set()
        worker.join(timeout=10.0)
        session.close()
    assert not errors


@pytest.mark.skipif(not cuda_available(), reason="CUDA extension/device unavailable")
def test_cuda_session_reuses_native_system_and_is_stable() -> None:
    H, g, indices, periods = _session_fixture()
    with LatticeQPSession(
        H, integer_indices=indices, lattice_steps=periods
    ) as session:
        reference = session.solve(g, backend="cuda")
        for _ in range(5):
            result = session.solve(g, backend="cuda")
            _assert_same_result(result, reference)
        stats = session.stats
        assert stats.solve_calls == 6
        assert stats.cuda_system_creations == 1
        assert stats.cuda_system_reuses >= 5


@pytest.mark.skipif(not cuda_available(), reason="CUDA extension/device unavailable")
def test_cuda_reset_stats_preserves_resource_and_reports_reuse() -> None:
    H, g, indices, periods = _session_fixture()
    with LatticeQPSession(
        H, integer_indices=indices, lattice_steps=periods
    ) as session:
        session.solve(g, backend="cuda")
        session.reset_stats()
        session.solve(g, backend="cuda")
        assert session.stats.solve_calls == 1
        assert session.stats.cuda_system_creations == 0
        assert session.stats.cuda_system_reuses == 1


@pytest.mark.skipif(not cuda_available(), reason="CUDA extension/device unavailable")
def test_cuda_session_alternating_inputs_and_policies_match_fresh_solves() -> None:
    H = sparse.eye(2, format="csr")
    indices = np.array([0, 1], dtype=np.int64)
    periods = np.ones(2, dtype=np.float64)
    calls = [
        (np.array([0.2, 0.8]), np.array([0.0, 0.0]), RoundingPolicy("sequential")),
        (
            np.array([-0.7, 0.3]),
            np.array([0.45, -0.25]),
            RoundingPolicy("all", "away_from_zero"),
        ),
    ]
    incidence = dict(
        incidence_indptr=np.array([0, 2], dtype=np.int64),
        incidence_indices=np.array([0, 1], dtype=np.int64),
        base_cones=np.array([0], dtype=np.int64),
        minimum_cones=np.array([0], dtype=np.int64),
    )
    cone_policies = [
        lattice_qp.ConeRoundingPolicy(
            incidence_coefficients=np.array([1, -1], dtype=np.int64),
            selection="sequential",
            **incidence,
        ),
        lattice_qp.ConeRoundingPolicy(
            incidence_coefficients=np.array([2, -2], dtype=np.int64),
            selection="all",
            projection="floor",
            **incidence,
        ),
    ]
    calls.extend(
        [
            (np.array([0.6, -0.2]), np.array([-0.1, 0.4]), cone_policies[0]),
            (np.array([-0.8, 0.9]), np.array([0.25, -0.35]), cone_policies[1]),
        ]
    )
    with LatticeQPSession(
        H, integer_indices=indices, lattice_steps=periods
    ) as session:
        for g, x0, policy in calls:
            actual = session.solve(
                g,
                x0=x0,
                rounding=policy,
                backend="cuda",
            )
            expected = solve_lattice_qp(
                H,
                g,
                integer_indices=indices,
                lattice_steps=periods,
                x0=x0,
                rounding=policy,
                backend="cuda",
            )
            _assert_same_result(actual, expected)
        stats = session.stats
        assert stats.solve_calls == len(calls)
        assert stats.cuda_system_creations == 1
        assert stats.cuda_system_reuses == len(calls) - 1


def test_session_context_manager_closes_on_exception() -> None:
    H, _, indices, periods = _session_fixture()
    with pytest.raises(RuntimeError, match="sentinel"):
        with LatticeQPSession(
            H, integer_indices=indices, lattice_steps=periods
        ) as session:
            raise RuntimeError("sentinel")
    with pytest.raises(RuntimeError, match="closed"):
        session.solve(np.zeros(H.shape[0]), backend="cpu")
