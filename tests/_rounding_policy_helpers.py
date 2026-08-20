"""Small public-API fixtures shared by rounding-policy tests."""

from __future__ import annotations

import numpy as np
from scipy import sparse

import lattice_qp


def diagonal_fixture():
    values = np.array([-1.7, -1.2, -0.2, 0.2, 0.8, 1.2], dtype=np.float64)
    return (
        sparse.eye(len(values), format="csr"),
        values,
        np.arange(len(values), dtype=np.int64),
        np.ones(len(values), dtype=np.float64),
    )


def coupled_fixture():
    H = sparse.csr_matrix(
        [[2.4, 0.55, 0.20, 0.00], [0.55, 2.1, 0.45, 0.10],
         [0.20, 0.45, 2.8, 0.35], [0.00, 0.10, 0.35, 1.7]],
        dtype=np.float64,
    )
    target = np.array([0.23, 0.79, 1.21, -0.64], dtype=np.float64)
    return H, np.asarray(H @ target), np.arange(4, dtype=np.int64), np.ones(4)


def cuda_available() -> bool:
    try:
        core = lattice_qp._api._core  # type: ignore[attr-defined]
        return core is not None and bool(core.cuda_available())
    except (AttributeError, ImportError, OSError, RuntimeError):
        return False


def rounding_policy(selection="multiple", projection="nearest", threshold=0.5):
    policy_type = getattr(lattice_qp, "RoundingPolicy", None)
    if policy_type is None:
        raise RuntimeError("RoundingPolicy is not available")
    return policy_type(selection=selection, projection=projection, threshold=threshold)


def policy_kwargs(policy):
    return {"rounding": policy}


def assert_integral(result, indices, periods) -> None:
    assert result.integrality_residual <= 1.0e-12
    np.testing.assert_allclose(
        result.x[indices] / periods,
        np.rint(result.x[indices] / periods),
        rtol=0.0,
        atol=1.0e-12,
    )
