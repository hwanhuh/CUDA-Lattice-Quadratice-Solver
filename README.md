# Lattice-QP

`Lattice-QP` is a CUDA-accelerated, policy-driven relax-and-fix solver for sparse convex lattice quadratic programs:

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

### Reusing a prepared problem

For geometry outer loops that solve the same matrix structure with changing
right-hand sides, `LatticeQPSession` keeps immutable copies of the normalized
sparse problem, integer coordinates, lattice steps, and optional block pairs.
When `backend="cuda"` is selected, it also keeps the native CUDA system alive
across calls. The per-call solver options match `solve_lattice_qp`; `g` and
`x0` may change between calls:

```python
from lattice_qp import LatticeQPSession

with LatticeQPSession(
    H,
    integer_indices=indices,
    lattice_steps=periods,
    block_pairs=None,
) as session:
    first = session.solve(g0, x0=x0, backend="cuda")
    second = session.solve(g1, x0=first.x, backend="cuda")
    print(session.stats.solve_calls)
```

The session is also explicitly closable with an idempotent `close()` method.
`session.stats` is a read-only snapshot exposing `solve_calls`,
`cuda_system_creations`, and `cuda_system_reuses`; call `reset_stats()` to
reset those counters without releasing the prepared resources. A session is
single-owner and rejects overlapping `solve()` calls.

CUDA sessions reuse the analyzed CSR matrix, block-Jacobi preconditioner,
device matrix, handles, and PCG work buffers. CPU sessions retain the
normalized structural inputs but currently rebuild SciPy projected systems
and preconditioners for each solve.

The session does not change the solver's optimization scope: it remains a
deterministic relax-and-fix heuristic for one fixed sparse lattice-QP. Use
`solve_lattice_qp` when a one-shot call is sufficient, and use a session when
repeated solves share the same `H`, integer coordinates, lattice steps, and
block structure.

Selection and lattice projection are independent. Typed policies make that
choice explicit while the original string options remain supported:

```python
from lattice_qp import RoundingPolicy

policy = RoundingPolicy(
    selection="sequential",       # sequential, multiple, or all
    projection="away_from_zero", # nearest, floor, ceil, or away_from_zero
)

result = solve_lattice_qp(
    H,
    g,
    integer_indices=indices,
    lattice_steps=periods,
    rounding=policy,
)
```

`rounding="multiple"` remains an alias for multiple selection with nearest
projection. `rounding="greedy"` retains its historical behavior and fixes all
integer coordinates at once with nearest projection.

## Cone-aware geometry problems

`solve_cone_miq` adds the stateful terminal correction used by cross-field and
period-jump problems. Cone incidence is evaluated on normalized integer
coordinates `q[j] = x[integer_indices[j]] / lattice_steps[j]`, not on the
physical values of `x`:

```python
from lattice_qp import solve_cone_miq

result = solve_cone_miq(
    H,
    g,
    integer_indices=period_jump_indices,
    edge_endpoints=edge_cone_pairs,  # (tail cone, head cone) per integer DOF
    base_cones=base_cone_indices,
    minimum_cones=minimum_cone_indices,
    double_cover=False,
    selection="multiple",
    backend="auto",
)

assert result.cone_feasible
print(result.cone_correction_count, result.cone_max_violation)
```

Advanced callers can construct `ConeRoundingPolicy` directly from a CSR
incidence matrix. Each signed integer coefficient contributes `a[c, j] * q[j]`
to cone row `c`; coefficients such as `+/-2` therefore support double-cover
formulations without mesh-specific objects in the solver core. Conflicting
terminal bounds are returned deterministically and exposed through
`cone_feasible`, `cone_violation_count`, `cone_max_violation`, `cone_values`,
and `cone_violations` rather than being silently hidden.

This helper solves one already-linear sparse lattice QP. It is not the complete
nonlinear Penner-coordinate algorithm from [*Seamless Parametrization in Penner
Coordinates*](https://arxiv.org/abs/2407.21342): callers remain responsible for
angle/holonomy Jacobian updates, line search, intrinsic flips, and any outer
Newton iteration.

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

The `multiple` selection policy sorts normalized rounding residues,
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
