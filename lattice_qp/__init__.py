"""CUDA-accelerated policy-driven lattice-QP solver."""

from ._api import LatticeQPResult, solve_lattice_qp
from .geometry import solve_cone_miq
from .policies import ConeRoundingPolicy, RoundingPolicy

__all__ = [
    "ConeRoundingPolicy",
    "LatticeQPResult",
    "RoundingPolicy",
    "solve_cone_miq",
    "solve_lattice_qp",
]
__version__ = "0.2.0"
