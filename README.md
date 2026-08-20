# Lattice-QP

`Lattice-QP` is a CUDA-accelerated relax-and-fix solver for sparse convex lattice quadratic programs:

```math
\min_x \frac{1}{2}x^T Hx-g^Tx,
\qquad x_i \in p_i\mathbb Z.
```

The package deliberately has a narrow contract. `H` is a symmetric positive semidefinite sparse matrix,
equality constraints have already been eliminated, and the selected projected problems must be bounded.
It does not implement branch-and-bound, inequalities, binary variables, or an optimality-gap certificate.
The returned point is a feasible deterministic multiple-rounding candidate, not a globally optimal MIQP certificate.

## Usage

```python
from lattice_qp import solve_lattice_qp

result = solve_lattice_qp(
    H,
    g,
    integer_indices=indices,
    lattice_steps=periods,
    x0=initial_guess,
    rounding="multiple",
    final_solve="pcg",
    block_pairs=None,
    pcg_check_interval=4,
)

print(result.x, result.objective, result.relative_residual)
```

`block_pairs=None` asks the CUDA backend to form disjoint algebraic 2-by-2 Jacobi blocks
from mutual strongest off-diagonal couplings.
Explicit disjoint pairs may be supplied as an `(n, 2)` integer array.
Variables whose partner is fixed automatically fall back to scalar Jacobi.

CUDA PCG keeps convergence and breakdown state on the device. Host checks use
two single-iteration probes, then ramp to 2 and at most
`pcg_check_interval` iterations, instead
of copying the recursive residual and synchronizing the CUDA stream after every
iteration. The default maximum interval is 4; values from 1 through 64 are
accepted. The short first chunk avoids speculative work in the many projected
systems that converge immediately. Once a solve converges inside a longer
chunk, device-side guards freeze the solution for the rest of that chunk. Every
solve still recomputes an explicit residual, and the public API independently
verifies the true residual on the host. `result.pcg_host_synchronizations`
reports the synchronization count for profiling.

The `multiple` policy sorts normalized rounding residues,
fixes at least one variable and then the largest prefix whose cumulative residue is at most `multiple_rounding_threshold` (default `0.5`),
and re-solves between batches.
Intermediate projected solves use tolerance `1e-3` and at most 50 PCG iterations by default.
The final all-fixed continuous problem is solved to `2e-6` by default.

`backend="auto"` attempts CUDA and falls back to SciPy PCG. Use
`backend="cuda"` when fallback would hide a deployment error.

The deterministic synchronization benchmark can be run with:

```bash
python benchmarks/benchmark_pcg_sync.py
python benchmarks/benchmark_pcg_sync.py --diagonal 2.0005 --integer-stride 0
```

## Lineage

Its multiple-rounding policy is based on the policy described and implemented by CoMISo.
