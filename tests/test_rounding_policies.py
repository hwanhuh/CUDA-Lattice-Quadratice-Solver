"""Selection, projection, and legacy rounding-policy contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from lattice_qp import solve_lattice_qp

from _rounding_policy_helpers import (
    assert_integral,
    coupled_fixture,
    diagonal_fixture,
    policy_kwargs,
    rounding_policy,
    cuda_available,
)


@pytest.mark.parametrize(
    ("projection", "expected"),
    [
        ("nearest", [-2.0, -1.0, 0.0, 0.0, 1.0, 1.0]),
        ("floor", [-2.0, -2.0, -1.0, 0.0, 0.0, 1.0]),
        ("ceil", [-1.0, -1.0, 0.0, 1.0, 1.0, 2.0]),
        ("away_from_zero", [-2.0, -2.0, -1.0, 1.0, 1.0, 2.0]),
    ],
)
def test_projection_policies_have_signed_lattice_semantics(projection, expected) -> None:
    H, g, indices, periods = diagonal_fixture()
    result = solve_lattice_qp(
        H,
        g,
        integer_indices=indices,
        lattice_steps=periods,
        final_solve="none",
        backend="cpu",
        **policy_kwargs(rounding_policy("all", projection)),
    )
    np.testing.assert_array_equal(result.x, np.asarray(expected))
    assert_integral(result, indices, periods)
    assert result.rounding_batches == 1


@pytest.mark.parametrize("selection", ["sequential", "multiple", "all"])
def test_selection_policies_are_deterministic_and_report_batch_cardinality(selection) -> None:
    H, g, indices, periods = coupled_fixture()
    kwargs = {
        "integer_indices": indices,
        "lattice_steps": periods,
        "backend": "cpu",
        **policy_kwargs(rounding_policy(selection)),
    }
    first = solve_lattice_qp(H, g, **kwargs)
    second = solve_lattice_qp(H, g, **kwargs)
    np.testing.assert_array_equal(first.x, second.x)
    assert first.rounding_batches == second.rounding_batches
    assert first.linear_solves == second.linear_solves
    assert_integral(first, indices, periods)
    if selection == "sequential":
        assert first.rounding_batches == len(indices)
    elif selection == "all":
        assert first.rounding_batches == 1
    else:
        assert 1 <= first.rounding_batches <= len(indices)


def test_multiple_selection_re_solves_between_stateful_batches() -> None:
    H, g, indices, periods = coupled_fixture()
    common = {
        "integer_indices": indices,
        "lattice_steps": periods,
        "backend": "cpu",
        "multiple_rounding_threshold": 0.5,
    }
    multiple = solve_lattice_qp(
        H, g, **common, **policy_kwargs(rounding_policy("multiple", threshold=0.5))
    )
    all_at_once = solve_lattice_qp(
        H, g, **common, **policy_kwargs(rounding_policy("all"))
    )
    assert 1 <= multiple.rounding_batches <= len(indices)
    assert all_at_once.rounding_batches == 1
    assert multiple.linear_solves >= multiple.rounding_batches
    assert_integral(multiple, indices, periods)


def test_legacy_rounding_names_map_to_typed_defaults() -> None:
    H, g, indices, periods = coupled_fixture()
    common = {"integer_indices": indices, "lattice_steps": periods, "backend": "cpu"}
    legacy_multiple = solve_lattice_qp(H, g, rounding="multiple", **common)
    explicit_multiple = solve_lattice_qp(
        H, g, **common, **policy_kwargs(rounding_policy("multiple"))
    )
    legacy_greedy = solve_lattice_qp(H, g, rounding="greedy", **common)
    explicit_all = solve_lattice_qp(H, g, **common, **policy_kwargs(rounding_policy("all")))
    np.testing.assert_array_equal(legacy_multiple.x, explicit_multiple.x)
    np.testing.assert_array_equal(legacy_greedy.x, explicit_all.x)
    assert legacy_multiple.rounding == "multiple"
    assert legacy_greedy.rounding == "greedy"


def test_cpu_backend_preserves_the_public_initial_guess() -> None:
    H, target, _, _ = diagonal_fixture()
    result = solve_lattice_qp(
        H,
        target,
        integer_indices=np.empty(0, dtype=np.int64),
        lattice_steps=np.empty(0),
        x0=target,
        backend="cpu",
    )
    np.testing.assert_array_equal(result.x, target)
    assert result.linear_iterations == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"selection": "bad"}, "selection"),
        ({"projection": "bad"}, "projection"),
        ({"selection": "all", "projection": "bad"}, "projection"),
    ],
)
def test_invalid_policy_names_are_rejected(kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        rounding_policy(**kwargs)


@pytest.mark.skipif(not cuda_available(), reason="CUDA extension/device unavailable")
def test_policy_results_are_deterministic_and_match_between_cpu_and_cuda() -> None:
    H, g, indices, periods = coupled_fixture()
    kwargs = {
        "integer_indices": indices,
        "lattice_steps": periods,
        "tolerance": 2.0e-7,
        "intermediate_tolerance": 1.0e-4,
        **policy_kwargs(rounding_policy("multiple")),
    }
    cpu = solve_lattice_qp(H, g, backend="cpu", **kwargs)
    cuda_results = [solve_lattice_qp(H, g, backend="cuda", **kwargs) for _ in range(2)]
    for cuda in cuda_results:
        np.testing.assert_allclose(cuda.x, cpu.x, rtol=3.0e-5, atol=4.0e-5)
        assert cuda.rounding_batches == cpu.rounding_batches
        assert cuda.integrality_residual <= 1.0e-12
    np.testing.assert_array_equal(cuda_results[0].x, cuda_results[1].x)
