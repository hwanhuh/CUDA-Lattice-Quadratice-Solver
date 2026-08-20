"""Cone-aware stateful rounding and wrapper contract tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import lattice_qp
from lattice_qp import solve_lattice_qp

from _rounding_policy_helpers import assert_integral, cuda_available


def cone_solver():
    solver = getattr(lattice_qp, "solve_cone_miq", None)
    if solver is None:
        pytest.fail("solve_cone_miq is not available")
    return solver


def cone_fixture():
    return {
        "H": sparse.eye(4, format="csr"),
        "g": np.array([0.2, 0.8, 1.2, -0.2]),
        "integer_indices": np.arange(4, dtype=np.int64),
        "lattice_steps": np.ones(4),
        "incidence_indptr": np.array([0, 2, 4, 6], dtype=np.int64),
        "incidence_indices": np.array([0, 1, 1, 2, 2, 3], dtype=np.int64),
        "incidence_coefficients": np.array([1, -1, 1, -1, 1, -1], dtype=np.int64),
        "base_cones": np.zeros(3, dtype=np.int64),
        "minimum_cones": np.zeros(3, dtype=np.int64),
    }


def cone_policy(fixture, selection="multiple", projection="nearest"):
    policy_type = getattr(lattice_qp, "ConeRoundingPolicy", None)
    if policy_type is None:
        pytest.fail("ConeRoundingPolicy is not available")
    return policy_type(
        incidence_indptr=fixture["incidence_indptr"],
        incidence_indices=fixture["incidence_indices"],
        incidence_coefficients=fixture["incidence_coefficients"],
        base_cones=fixture["base_cones"],
        minimum_cones=fixture["minimum_cones"],
        selection=selection,
        projection=projection,
    )


def cone_call(fixture, *, selection="multiple", projection="nearest", backend="cpu", **kwargs):
    solver = cone_solver()
    cone_fields = {
        "incidence_indptr", "incidence_indices", "incidence_coefficients",
        "base_cones", "minimum_cones",
    }
    args = {key: value for key, value in fixture.items() if key not in cone_fields}
    args.update({"cone_policy": cone_policy(fixture, selection, projection), "backend": backend})
    args.update(kwargs)
    return solver(**args)


def cone_stats(result):
    return (
        bool(result.cone_feasible),
        int(result.cone_correction_count),
        int(result.cone_violation_count),
        float(result.cone_max_violation),
    )


def test_cone_endpoint_wrapper_matches_equivalent_incidence_policy() -> None:
    H = sparse.eye(2, format="csr")
    g = np.array([0.2, 0.8])
    indices = np.array([0, 1], dtype=np.int64)
    steps = np.ones(2)
    base = np.zeros(2, dtype=np.int64)
    minimum = np.zeros(2, dtype=np.int64)
    wrapped = cone_solver()(
        H,
        g,
        integer_indices=indices,
        lattice_steps=steps,
        edge_endpoints=np.array([[0, 1], [0, 1]], dtype=np.int64),
        base_cones=base,
        minimum_cones=minimum,
        selection="sequential",
        backend="cpu",
    )
    policy = lattice_qp.ConeRoundingPolicy(
        incidence_indptr=np.array([0, 2, 4], dtype=np.int64),
        incidence_indices=np.array([0, 1, 0, 1], dtype=np.int64),
        incidence_coefficients=np.array([-1, -1, 1, 1], dtype=np.int64),
        base_cones=base,
        minimum_cones=minimum,
        selection="sequential",
    )
    direct = solve_lattice_qp(
        H,
        g,
        integer_indices=indices,
        lattice_steps=steps,
        rounding=policy,
        backend="cpu",
    )
    np.testing.assert_array_equal(wrapped.x, direct.x)
    assert cone_stats(wrapped)[0]


def test_cone_rounding_reports_feasibility_and_terminal_corrections() -> None:
    fixture = cone_fixture()
    result = cone_call(fixture, selection="sequential")
    feasible, corrections, violations, max_violation = cone_stats(result)
    assert feasible and violations == 0 and max_violation <= 1.0e-12
    assert corrections >= 0
    assert result.rounding_batches == len(fixture["integer_indices"])
    assert_integral(result, fixture["integer_indices"], fixture["lattice_steps"])


def test_terminal_incident_commit_repairs_last_active_variable() -> None:
    fixture = cone_fixture()
    fixture.update({
        "H": sparse.eye(2, format="csr"), "g": np.array([0.1, 0.8]),
        "integer_indices": np.array([0, 1], dtype=np.int64), "lattice_steps": np.ones(2),
        "incidence_indptr": np.array([0, 2], dtype=np.int64),
        "incidence_indices": np.array([0, 1], dtype=np.int64),
        "incidence_coefficients": np.array([1, -1], dtype=np.int64),
        "base_cones": np.array([0], dtype=np.int64), "minimum_cones": np.array([0], dtype=np.int64),
    })
    result = cone_call(fixture, selection="sequential")
    assert cone_stats(result)[:1] == (True,)
    assert cone_stats(result)[1] >= 1
    np.testing.assert_array_equal(result.x, np.array([0.0, 0.0]))


def test_conflicting_terminal_constraints_report_violations() -> None:
    fixture = cone_fixture()
    fixture.update({
        "H": sparse.eye(1, format="csr"), "g": np.array([0.8]),
        "integer_indices": np.array([0], dtype=np.int64), "lattice_steps": np.ones(1),
        "incidence_indptr": np.array([0, 1, 2], dtype=np.int64),
        "incidence_indices": np.array([0, 0], dtype=np.int64),
        "incidence_coefficients": np.array([1, -1], dtype=np.int64),
        "base_cones": np.array([0, 0], dtype=np.int64),
        "minimum_cones": np.array([0, 1], dtype=np.int64),
    })
    result = cone_call(fixture, selection="sequential")
    feasible, _, violations, max_violation = cone_stats(result)
    assert not feasible and violations > 0 and max_violation > 0.0


def test_double_cover_matches_explicitly_scaled_coefficients() -> None:
    base = cone_fixture()
    scaled = dict(base)
    scaled["incidence_coefficients"] = 2 * base["incidence_coefficients"]
    plain = cone_call(scaled, backend="cpu", double_cover=False)
    covered = cone_call(base, backend="cpu", double_cover=True)
    np.testing.assert_array_equal(plain.x, covered.x)
    assert cone_stats(plain)[0] and cone_stats(covered)[0]


def test_cone_incidence_uses_normalized_nonunit_lattice_coordinates() -> None:
    fixture = cone_fixture()
    periods = np.array([1.0, 2.0, 3.0, 4.0])
    q_target = np.array([0.2, 0.8, 1.2, -0.2])
    H = sparse.diags(1.0 / periods**2, format="csr")
    fixture.update({"lattice_steps": periods, "H": H, "g": np.asarray(H @ (periods * q_target))})
    result = cone_call(fixture, selection="sequential")
    feasible, _, violations, max_violation = cone_stats(result)
    assert feasible and violations == 0 and max_violation <= 1.0e-12
    assert_integral(result, fixture["integer_indices"], periods)


def test_equal_residue_multi_commit_order_is_deterministic() -> None:
    fixture = cone_fixture()
    fixture["g"] = np.array([0.2, 0.2, 0.8, 0.8])
    first = cone_call(fixture, selection="multiple")
    second = cone_call(fixture, selection="multiple")
    np.testing.assert_array_equal(first.x, second.x)
    assert first.rounding_batches == second.rounding_batches
    assert first.cone_correction_count == second.cone_correction_count


@pytest.mark.parametrize(
    "bad",
    [np.array([[0, 1, 2]]), np.array([[0, 1], [1, 1]]), np.array([[0, 4]])],
)
def test_invalid_incidence_indices_are_rejected(bad) -> None:
    fixture = cone_fixture()
    fixture["incidence_indices"] = bad
    with pytest.raises((TypeError, ValueError), match="incidence|index|shape|pair"):
        cone_call(fixture)


def test_invalid_incidence_coefficients_are_rejected() -> None:
    fixture = cone_fixture()
    for bad in (np.ones((2, 2), dtype=np.int64), np.array([1.0, 0.5]), np.array([1, -1, np.nan, 0, 1, -1])):
        fixture["incidence_coefficients"] = bad
        with pytest.raises((TypeError, ValueError), match="coefficient|shape|finite|integer"):
            cone_call(fixture)


def test_policy_rejects_mismatched_csr_vectors_at_construction() -> None:
    fixture = cone_fixture()
    with pytest.raises(ValueError, match="CSR array sizes"):
        lattice_qp.ConeRoundingPolicy(
            incidence_indptr=fixture["incidence_indptr"],
            incidence_indices=fixture["incidence_indices"],
            incidence_coefficients=fixture["incidence_coefficients"][:-1],
            base_cones=fixture["base_cones"],
            minimum_cones=fixture["minimum_cones"],
        )
    with pytest.raises(ValueError, match="matching lengths"):
        lattice_qp.ConeRoundingPolicy(
            incidence_indptr=fixture["incidence_indptr"],
            incidence_indices=fixture["incidence_indices"],
            incidence_coefficients=fixture["incidence_coefficients"],
            base_cones=fixture["base_cones"],
            minimum_cones=fixture["minimum_cones"][:-1],
        )


@pytest.mark.parametrize("impostor", ["greedy", lattice_qp.RoundingPolicy()])
def test_cone_wrapper_rejects_non_cone_policies(impostor) -> None:
    with pytest.raises(TypeError, match="ConeRoundingPolicy"):
        cone_solver()(
            sparse.eye(1, format="csr"),
            np.zeros(1),
            integer_indices=np.array([0]),
            cone_policy=impostor,
            backend="cpu",
        )


def test_cone_audit_saturates_an_unrepresentable_violation() -> None:
    limit = np.iinfo(np.int64)
    policy = lattice_qp.ConeRoundingPolicy(
        incidence_indptr=np.array([0, 0]),
        incidence_indices=np.empty(0, dtype=np.int64),
        incidence_coefficients=np.empty(0, dtype=np.int64),
        base_cones=np.array([limit.min]),
        minimum_cones=np.array([limit.max]),
    )
    result = solve_lattice_qp(
        sparse.eye(1, format="csr"),
        np.zeros(1),
        integer_indices=np.empty(0, dtype=np.int64),
        rounding=policy,
        backend="cpu",
    )
    assert result.cone_max_violation == limit.max
    np.testing.assert_array_equal(result.cone_violations, [limit.max])


def test_unrepresentable_terminal_correction_keeps_basic_projection() -> None:
    limit = np.iinfo(np.int64)
    policy = lattice_qp.ConeRoundingPolicy(
        incidence_indptr=np.array([0, 1]),
        incidence_indices=np.array([0]),
        incidence_coefficients=np.array([1]),
        base_cones=np.array([limit.min]),
        minimum_cones=np.array([limit.max]),
        selection="all",
    )
    common = {
        "integer_indices": np.array([0]),
        "rounding": policy,
        "final_solve": "none",
    }
    cpu = solve_lattice_qp(
        sparse.eye(1, format="csr"), np.zeros(1), backend="cpu", **common
    )
    np.testing.assert_array_equal(cpu.x, [0.0])
    assert not cpu.cone_feasible
    assert cpu.cone_correction_count == 0
    if cuda_available():
        cuda = solve_lattice_qp(
            sparse.eye(1, format="csr"), np.zeros(1), backend="cuda", **common
        )
        np.testing.assert_array_equal(cuda.x, cpu.x)
        assert cone_stats(cuda) == cone_stats(cpu)


def test_cone_accumulation_overflow_is_rejected() -> None:
    limit = np.iinfo(np.int64)
    policy = lattice_qp.ConeRoundingPolicy(
        incidence_indptr=np.array([0, 1]),
        incidence_indices=np.array([0]),
        incidence_coefficients=np.array([1]),
        base_cones=np.array([limit.max]),
        minimum_cones=np.array([0]),
        selection="all",
    )
    with pytest.raises(RuntimeError, match="int64 range"):
        solve_lattice_qp(
            sparse.eye(1, format="csr"),
            np.ones(1),
            integer_indices=np.array([0]),
            rounding=policy,
            backend="cpu",
        )


def test_double_cover_coefficient_overflow_is_rejected() -> None:
    policy = lattice_qp.ConeRoundingPolicy(
        incidence_indptr=np.array([0, 1]),
        incidence_indices=np.array([0]),
        incidence_coefficients=np.array([np.iinfo(np.int64).max // 2 + 1]),
        base_cones=np.array([0]),
        minimum_cones=np.array([0]),
    )
    with pytest.raises(ValueError, match="double-cover.*int64"):
        cone_solver()(
            sparse.eye(1, format="csr"),
            np.zeros(1),
            integer_indices=np.array([0]),
            cone_policy=policy,
            double_cover=True,
            backend="cpu",
        )


@pytest.mark.skipif(not cuda_available(), reason="CUDA extension/device unavailable")
def test_cone_cpu_cuda_parity_and_stable_audit() -> None:
    fixture = cone_fixture()
    cpu = cone_call(fixture, selection="multiple", backend="cpu")
    cuda = cone_call(fixture, selection="multiple", backend="cuda")
    np.testing.assert_allclose(cuda.x, cpu.x, rtol=3.0e-5, atol=4.0e-5)
    assert cone_stats(cuda) == cone_stats(cpu)


def test_final_cone_stats_are_exposed() -> None:
    result = cone_call(cone_fixture())
    for name in ("cone_feasible", "cone_correction_count", "cone_violation_count", "cone_max_violation"):
        assert hasattr(result, name)
