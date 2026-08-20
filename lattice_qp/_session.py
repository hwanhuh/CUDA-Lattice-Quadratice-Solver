"""Persistent prepared-state API for repeated lattice-QP solves."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from scipy import sparse

from ._api import (
    _core,
    _problem_with_vectors,
    _solve_normalized_problem,
    _validate_runtime_options,
)
from ._problem import NormalizedProblem, normalize_problem
from ._result import LatticeQPResult
from ._rounding import resolve_rounding
from .policies import ConeRoundingPolicy, RoundingPolicy


@dataclass(frozen=True)
class LatticeQPSessionStats:
    """Read-only snapshot of accepted solve attempts and CUDA reuse."""

    solve_calls: int
    cuda_system_creations: int
    cuda_system_reuses: int


class LatticeQPSession:
    """Prepared solver for repeated RHS solves over one immutable lattice QP.

    CUDA resources are initialized lazily on the first CUDA-capable solve and
    retained until :meth:`close`. A session is intentionally single-flight:
    overlapping ``solve``, ``close``, stats, or reset operations fail instead
    of racing a GIL-released native solve. Sessions created before ``fork``
    must be replaced in the child process.
    """

    def __init__(
        self,
        H: sparse.spmatrix,
        *,
        integer_indices: np.ndarray,
        lattice_steps: np.ndarray | float = 1.0,
        block_pairs: np.ndarray | None = None,
        check_symmetry: bool = True,
    ) -> None:
        # Force independent ownership: scipy may otherwise return the caller's
        # CSR object unchanged and normalize it in place.
        matrix = sparse.csr_matrix(H, dtype=np.float64, copy=True)
        dimension = matrix.shape[0] if matrix.ndim == 2 else 0
        normalized = normalize_problem(
            matrix,
            np.zeros(dimension, dtype=np.float64),
            integer_indices,
            lattice_steps,
            None,
            block_pairs,
            check_symmetry=check_symmetry,
        )
        self._template = NormalizedProblem(
            normalized.H.copy(),
            normalized.g,
            normalized.integer_indices.copy(),
            normalized.lattice_steps.copy(),
            normalized.x0,
            normalized.block_pairs.copy(),
        )
        self._guard = threading.Lock()
        self._creator_pid = os.getpid()
        self._closed = False
        self._native_session: object | None = None
        self._native_initialization_error: RuntimeError | None = None
        self._solve_calls = 0
        self._retired_cuda_stats = {
            "solve_calls": 0,
            "cuda_system_creations": 0,
            "cuda_system_reuses": 0,
        }

    def _ensure_process(self) -> None:
        if os.getpid() != self._creator_pid:
            raise RuntimeError(
                "LatticeQPSession cannot be used after fork; create a new "
                "session in the child process"
            )

    def _ensure_native_session(self) -> object:
        if self._native_session is None:
            if _core is None:
                raise RuntimeError("lattice_qp CUDA extension is not installed")
            if self._native_initialization_error is not None:
                raise RuntimeError(str(self._native_initialization_error)) from (
                    self._native_initialization_error
                )
            problem = self._template
            if not _core.cuda_available():
                error = RuntimeError("no CUDA device is available")
                # Device absence is stable for this process/session. Cache it
                # so repeated auto solves do not repeat availability probes.
                self._native_initialization_error = error
                raise error
            # Allocation/driver failures may be transient (for example, an
            # OOM can clear after another owner releases memory), so let the
            # next solve retry those rather than permanently poisoning the
            # session.
            self._native_session = _core.PreparedCudaSolver(
                np.asarray(problem.H.indptr, dtype=np.int32),
                np.asarray(problem.H.indices, dtype=np.int32),
                np.asarray(problem.H.data, dtype=np.float64),
                problem.integer_indices,
                problem.lattice_steps,
                problem.block_pairs,
            )
        return self._native_session

    def solve(
        self,
        g: np.ndarray,
        *,
        x0: np.ndarray | None = None,
        rounding: str | RoundingPolicy | ConeRoundingPolicy = "multiple",
        final_solve: Literal["pcg", "none"] = "pcg",
        backend: Literal["auto", "cuda", "cpu"] = "auto",
        tolerance: float = 2.0e-6,
        maximum_iterations: int = 20_000,
        intermediate_tolerance: float = 1.0e-3,
        intermediate_maximum_iterations: int = 50,
        multiple_rounding_threshold: float = 0.5,
        pcg_check_interval: int = 4,
    ) -> LatticeQPResult:
        """Solve for a new linear term while reusing prepared CUDA state."""

        if not self._guard.acquire(blocking=False):
            raise RuntimeError(
                "concurrent operations on one LatticeQPSession are not supported"
            )
        try:
            self._ensure_process()
            if self._closed:
                raise RuntimeError("LatticeQPSession is closed")
            (
                maximum_iterations,
                intermediate_maximum_iterations,
                pcg_check_interval,
            ) = _validate_runtime_options(
                final_solve=final_solve,
                backend=backend,
                tolerance=tolerance,
                maximum_iterations=maximum_iterations,
                intermediate_tolerance=intermediate_tolerance,
                intermediate_maximum_iterations=intermediate_maximum_iterations,
                multiple_rounding_threshold=multiple_rounding_threshold,
                pcg_check_interval=pcg_check_interval,
            )
            problem = _problem_with_vectors(self._template, g, x0)
            # Resolve before allocating a CUDA system, so a malformed per-call
            # policy cannot create expensive session resources as a side effect.
            resolve_rounding(
                rounding,
                multiple_rounding_threshold,
                len(problem.integer_indices),
            )
            self._solve_calls += 1
            started = perf_counter()
            prepared_cuda: object = None
            native_error: Exception | None = None
            if backend in {"auto", "cuda"}:
                if _core is not None:
                    try:
                        prepared_cuda = self._ensure_native_session()
                    except ValueError:
                        raise
                    except RuntimeError as error:  # pragma: no cover - hardware dependent
                        native_error = error
                        if backend == "cuda":
                            raise
            result = _solve_normalized_problem(
                problem,
                rounding=rounding,
                final_solve=final_solve,
                backend=backend,
                tolerance=tolerance,
                maximum_iterations=maximum_iterations,
                intermediate_tolerance=intermediate_tolerance,
                intermediate_maximum_iterations=intermediate_maximum_iterations,
                multiple_rounding_threshold=multiple_rounding_threshold,
                pcg_check_interval=pcg_check_interval,
                prepared_cuda=prepared_cuda,
                native_error=native_error,
                started=started,
            )
            return result
        finally:
            self._guard.release()

    @property
    def stats(self) -> LatticeQPSessionStats:
        """Return a snapshot of public calls and actual CUDA resource reuse."""

        if not self._guard.acquire(blocking=False):
            raise RuntimeError(
                "session statistics are unavailable during another operation"
            )
        try:
            cuda_stats = dict(self._retired_cuda_stats)
            if self._native_session is not None:
                current = self._native_session.stats()
                for name in cuda_stats:
                    cuda_stats[name] += int(current[name])
            return LatticeQPSessionStats(
                solve_calls=self._solve_calls,
                cuda_system_creations=cuda_stats["cuda_system_creations"],
                cuda_system_reuses=cuda_stats["cuda_system_reuses"],
            )
        finally:
            self._guard.release()

    def reset_stats(self) -> None:
        """Reset counters without releasing or rebuilding prepared resources."""

        if not self._guard.acquire(blocking=False):
            raise RuntimeError(
                "session statistics cannot be reset during another operation"
            )
        try:
            self._solve_calls = 0
            for name in self._retired_cuda_stats:
                self._retired_cuda_stats[name] = 0
            if self._native_session is not None:
                self._native_session.reset_stats()
        finally:
            self._guard.release()

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Release prepared native resources; subsequent solves are invalid."""

        if not self._guard.acquire(blocking=False):
            raise RuntimeError(
                "LatticeQPSession cannot be closed during another operation"
            )
        try:
            if self._closed:
                return
            if self._native_session is not None:
                current = self._native_session.stats()
                for name in self._retired_cuda_stats:
                    self._retired_cuda_stats[name] += int(current[name])
                self._native_session = None
            self._closed = True
        finally:
            self._guard.release()

    def __enter__(self) -> LatticeQPSession:
        self._ensure_process()
        if self._closed:
            raise RuntimeError("LatticeQPSession is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter dependent
        try:
            self.close()
        except Exception:
            pass


__all__ = ["LatticeQPSession", "LatticeQPSessionStats"]
