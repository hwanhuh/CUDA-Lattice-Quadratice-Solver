"""Benchmark PCG host-check intervals on a deterministic sparse lattice QP."""

from __future__ import annotations

import argparse
import json
from statistics import median

import numpy as np
from scipy import sparse

from lattice_qp import solve_lattice_qp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=12_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--intervals", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--diagonal", type=float, default=2.05)
    parser.add_argument(
        "--integer-stride",
        type=int,
        default=6,
        help="Use 0 for a continuous-QP/long-PCG benchmark",
    )
    arguments = parser.parse_args()

    rng = np.random.default_rng(20260814)
    dimension = arguments.dimension
    off_diagonal = np.full(dimension - 1, -1.0)
    matrix = sparse.diags(
        [off_diagonal, np.full(dimension, arguments.diagonal), off_diagonal],
        [-1, 0, 1],
        format="csr",
    )
    target = rng.normal(size=dimension)
    linear = matrix @ target
    if arguments.integer_stride > 0:
        integer_indices = np.arange(
            arguments.integer_stride // 2,
            dimension,
            arguments.integer_stride,
            dtype=np.int64,
        )
    else:
        integer_indices = np.empty(0, dtype=np.int64)

    # Exclude one-time CUDA context initialization from all measurements.
    solve_lattice_qp(
        matrix[:128, :128],
        linear[:128],
        integer_indices=np.arange(3, 128, 6, dtype=np.int64),
        rounding="greedy",
        final_solve="none",
        backend="cuda",
    )
    for interval in arguments.intervals:
        solve_lattice_qp(
            matrix,
            linear,
            integer_indices=integer_indices,
            rounding="multiple",
            final_solve="pcg",
            backend="cuda",
            pcg_check_interval=interval,
        )

    samples: dict[int, list[float]] = {
        interval: [] for interval in arguments.intervals
    }
    results = {}
    for repeat in range(arguments.repeats):
        order = (
            arguments.intervals
            if repeat % 2 == 0
            else list(reversed(arguments.intervals))
        )
        for interval in order:
            result = solve_lattice_qp(
                matrix,
                linear,
                integer_indices=integer_indices,
                rounding="multiple",
                final_solve="pcg",
                backend="cuda",
                pcg_check_interval=interval,
            )
            samples[interval].append(result.solve_seconds)
            results[interval] = result

    for interval in arguments.intervals:
        result = results[interval]
        print(
            json.dumps(
                {
                    "pcg_check_interval": interval,
                    "median_seconds": median(samples[interval]),
                    "samples_seconds": samples[interval],
                    "rounding_batches": result.rounding_batches,
                    "linear_solves": result.linear_solves,
                    "linear_iterations": result.linear_iterations,
                    "pcg_host_synchronizations": result.pcg_host_synchronizations,
                    "relative_residual": result.relative_residual,
                    "relaxation_relative_residual": (
                        result.relaxation_relative_residual
                    ),
                    "objective": result.objective,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
