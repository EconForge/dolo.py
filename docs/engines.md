# Computational Engines

The global solution algorithms (`time_iteration`, `newton`) and the residual functions
(`F`, `dF_1`, `F_and_dF_1`, `dF_2`) take an `engine` argument that selects how the Euler
residuals and their derivatives are computed:

| `engine` | implementation | device |
|---|---|---|
| `"auto"` (default) | JAX if it can be imported, else vectorized | JAX's default device |
| `"jax"` | JAX | JAX's default device (a GPU if one is visible) |
| `"gpu"` (or `"cuda"`) | JAX | the first GPU (error if there is none) |
| `"cpu"` | JAX | the CPU |
| `"vectorized"` (or `"numpy"`) | NumPy / numba | the host |

Both implementations compute the same objects, those of Dolo.jl. For controls $x$ on the grid
and a decision rule $\varphi$ (a spline per exogenous regime interpolating the controls):

$$F(x, \varphi)_{i,n} = \sum_k w_{ik}\, f\big(m_i, s_n, x_{in}, M_{ik}, S_{ikn}, \varphi_{r(k)}(S_{ikn})\big),
\qquad S_{ikn} = g(m_i, s_n, x_{in}, M_{ik})$$

where $i$ is the exogenous regime today, $n$ the grid point, and $k$ the node of the process
tomorrow, with weight $w_{ik}$, value $M_{ik}$ and regime $r(k)$. The algorithms need:

- $F$, and $dF_1 = \partial F / \partial x$: block diagonal, one $n_x \times n_x$ block per
  point;
- $dF_2 = \partial F / \partial \varphi$: a linear operator
  $L.v = \sum_k F_x[i,k,n] \cdot v_{r(k)}(S_{ikn})$ acting on variations $v$ of the decision
  rule; it is stored as the blocks $F_x$ and the points $S$;
- $J \backslash L$ (preconditioning by $dF_1$), and solutions of $(I - L)\,dx = b$ (Neumann
  series or GMRES).

All engines use the same algorithms, with the same defaults and stopping rules as Dolo.jl
(see [Defaults](#defaults-identical-to-dolojl)).

---

## Which code runs

| | JAX engines | vectorized engine |
|---|---|---|
| no complementarity conditions (or `ignore_constraints=True`), all regimes with the same number of nodes | **device loops**: everything on the device | host loops |
| complementarity conditions | host loops (Fischer–Burmeister smoothing, `ncpsolve` for the inner solver of time iteration): `time_iteration` evaluates the residuals and Jacobians with the JAX kernels and copies them to the host, `newton` uses the vectorized residuals | host loops |

The *device loops* (`_time_iteration_device` in `algos/time_iteration.py`, `_newton_device` in
`algos/newton.py`) keep the controls, the spline coefficients, the Jacobians and the operator
$L$ on the device. Only error norms (scalars) go to the host during the iterations; the host
`DecisionRule` returned in the result is fitted once, at the end. The other loops work on host
(NumPy) arrays.

---

## Vectorized engine (NumPy / numba)

Code: `numeric/tau.py` (`euler_residuals_tau`), `algos/newton.py` (`euler_residuals`,
`Operator`, `neumann_solve`), `algos/time_iteration.py`.

- **Model equations** are evaluated on `(N, n)` arrays, one call per (regime, node): NumPy
  expressions emitted from the equations (or the numba functions of the legacy compiler).
- **Decision rule**: cubic splines, coefficients by prefiltering. The approximators use the JAX
  implementation (`interpolation_jax`) by default, on JAX's default device: the values are
  copied there and back at each evaluation.
- **dF_1** by finite differences of $F$ (one evaluation of the residuals per control);
  **dF_2** by finite differences of $f$ with respect to the next-period controls
  (`euler_residuals(..., with_jres=True)`), stored as dense blocks.
- **Linear solves** on the host: `scipy.sparse.linalg.gmres` (with Dolo.jl's settings: from
  $x_0 = b$, one cycle, absolute tolerance) or a Neumann series; each product $L.v$ refits a
  spline to $v$.

It is the reference for the checks of the JAX engine, and `newton` uses it for problems with
complementarity conditions whatever the engine. It is not tuned for speed: on integration_B (20³×9 points, 8
controls), $F$ takes about 140 ms and the finite-difference $dF_1$ about 1.3 s (RTX 2080 host,
2026-10-20).

---

## JAX engine

Code:

| module | content |
|---|---|
| `numeric/residual_jax.py` | `CompiledResidual`: entry point (`F`, `dF_1`, `F_and_dF_1`, `dF_2`, `prefilter`), point functions, the previous vmapped kernels (`fast=False`) |
| `numeric/residual_kernels_jax.py` | `GridKernels`: staged kernels for F, (F, dF_1), dF_2; structural sparsity and colored Jacobians; small jitted solver steps |
| `numeric/policy_operator_jax.py` | the operator $L$: location of the points, prefilter, spline evaluation, contraction, $J\backslash L$, GMRES |
| `numeric/residuals.py` | `PolicyLinearOperator` (both engines) and the residual wrappers |
| `numeric/interpolation_jax.py` | splines (prefilter matrices, basis) |

All arrays are `(n_regimes, N, n_x)`, one regime being a regime axis of length one. Float64 only
for now.

### Kernels for F, dF_1 and dF_2 (`GridKernels`)

1. **Model equations as JAX point functions.** The equations are emitted as `jax.numpy`
   expressions of one point; they are `vmap`ped over the grid points and the next-period nodes.
   Terms that depend only on today's variables are computed once per point, not once per node
   (`vmap` does not batch what does not depend on the node).
2. **Separate compiled stages.** $S = g(\cdot)$, the decision rule at $S$, and the arbitrage part
   are three jitted functions. Compiled as one function, XLA fuses the spline evaluation (gathers)
   into its consumers and recomputes it for each output: several times slower.
3. **Spline evaluation.** For each point: index of the cell (one flat index into the coefficients
   of all regimes) and the 1-D basis weights; the value is the nested sum
   $\sum_p w_{0p} \sum_q w_{1q} \sum_r w_{2r}\, c_{pqr}$ (84 multiply-adds per output in 3-D) on
   coefficients gathered by flat index. No $4\times4\times4$ patches are materialized.
4. **Only the used columns.** The decision rule is interpolated only for the controls that $f$
   depends on next period (6 of 8 on integration_B).
5. **Linearized decision rule** (as Dolo.jl's `Linearized`). For $dF_1$, the decision rule at
   $S(x)$ is replaced by $X_0 + G\,(S(x) - S_0)$, with the value $X_0$ and gradient $G$ of the
   spline at $S_0$ computed by the spline code (the gradient: the same nested sum with the
   derivative of the basis in one dimension). Autodiff never differentiates the gathers.
6. **Sparse (colored) forward-mode Jacobians.** Columns of a Jacobian that never share a row
   are differentiated together, with one `jvp` per group of columns ("color"): 5 instead of 8
   for $dF_1$, 4 instead of 8 for $\partial f / \partial X$ (which gives $dF_2$) on integration_B.
   The sparsity pattern is **structural**: it is read from the traced program (`jaxpr`), by
   propagating which inputs each value depends on, with the values of the parameters (so that
   $x^0$ and $0 \cdot x$ do not depend on $x$). It is never inferred from numerical zeros: on
   integration_B, $\partial f / \partial p_f'$ is zero at the steady-state guess (because
   $db_f = 0$ there) but not elsewhere.
7. **Bounded memory.** At most $2^{20}$ (regime, node, point) triples are processed at once;
   beyond, regimes go in sequence (`lax.map` with `batch_size`), sums over nodes are done per
   regime. 20³×9 runs in one batch; 40³×9 fits in 6 GB.
8. **dF_2 compressed.** `dF_2_compressed` returns only the columns of the blocks $F_x$ that can
   be nonzero; the dense blocks are built only on request (`CompiledResidual.dF_2`,
   `PolicyLinearOperator.F_x`).

### The operator L (`PolicyLinearOperator`, `policy_operator_jax`)

1. **Points located once.** $S$ does not change while $L$ is applied (in a linear solve): the
   cell index and the basis weights of each $S_{ikn}$ are computed when the operator is built.
2. **Column compression.** Columns of $F_x$ that are zero everywhere are dropped: $L.v$
   prefilters and interpolates only these columns of $v$. $J^{-1} F_x$ keeps them zero.
3. **$L.v$ in three jitted stages**: prefilter (three small matrix products per regime), nested
   spline evaluation, and the contraction $\sum_k F_x \cdot v(S)$ written elementwise (faster
   than `einsum`'s batched products).
4. **$J \backslash L$**: each $n_x \times n_x$ block of $J$ is inverted once
   (`jnp.linalg.inv`), the products are elementwise (a batched solve with $K \cdot n_c$
   right-hand sides needs transposed copies); one regime at a time beyond $2^{20}$ blocks.
   Scaling the operator multiplies a factor, not the blocks.
5. **Linear solves on the device.** Neumann series (one scalar to the host per term), or GMRES
   written as Dolo.jl's `gmres!`: one cycle of at most `maxit` Krylov vectors from $x_0 = b$,
   modified Gram–Schmidt on the device (the basis is updated in place), Givens rotations of the
   small Hessenberg matrix on the host (one column copied per iteration), absolute tolerance on
   the 2-norm of the residual. The number of iterations is kept in `last_iterations`.

### Solvers (device loops)

- `CompiledResidual.prefilter` is one jitted computation for all regimes; device arrays stay on
  the device.
- Small steps of the loops (block Newton direction $J^{-1} r$, updates $x + a\,y$, norms) are
  jitted functions: one dispatch each, no constants copied from the host.
- Only the error norms are copied to the host (checked with `jax.transfer_guard`). The scalar
  arguments of the jitted functions (step sizes, tolerances) are the only other transfers.

### Things that did not work (on these GPUs)

- Pallas (Triton) kernels need a GPU of compute capability ≥ 8.0; the RTX 2080 and GTX 1660 Ti
  are 7.5.
- XLA computes each output element independently: the spline value with its 3 derivatives costs
  about 4 evaluations, whatever the formulation; multi-output fusions or materialized tensor-product
  weights were slower.
- An unrolled Gauss–Jordan inverse of the 8×8 blocks is faster than `jnp.linalg.inv` (2.6 vs
  5.9 ms) but takes minutes to compile; written as a loop over pivots it is slower (11.5 ms).
- `jax.scipy.sparse.linalg.gmres` is as fast as the GMRES above, but its tolerance is relative
  and it reorthogonalizes: it does not reproduce Dolo.jl's iterations.

---

## Defaults (identical to Dolo.jl)

| | option | default |
|---|---|---|
| `time_iteration` | `maxit`, `tol_eps`, `tol_eta` | 500, 1e-8, 1e-6 |
| | `inner` (Newton solver of $F(x_1, \varphi_0) = 0$) | `maxit=10`, `max_backsteps=5`, `tol=tol_eps` |
| | `accelerate` (when given) | `wait=10`, `linsolve="neumann"`, `tol=1e-10`, `maxit=1000` (Neumann terms) or 50 (GMRES Krylov vectors), `safeguarded=False` |
| `newton` | `maxit`, `tol_eps`, `tol_eta` | 20, 1e-8, None (no step criterion) |
| | `linsolve` | `"neumann"`, `tol=1e-10`, `maxit=1000` (Neumann) or 50 (GMRES) |
| | `linesearch` | None (full Newton steps); `"backtracking"`: `max_backsteps=10`; `"golden_section"`: `maxiter=5` |

Stopping rules, as in Dolo.jl: time iteration stops before a step when the residual is below
`tol_eps` (or not finite) and after a step when the step is below `tol_eta`; Newton stops when
the residual is below `tol_eps` (or not finite), and goes back to the previous point and stops if
no step decreases the residual. Iterations count the steps taken. `improved_time_iteration`
(legacy name of `newton`) keeps GMRES and backtracking.

Full Newton steps (the default) need a good starting point: from the default initial guess they
diverge on `rbc` and `rbc_mc`, where `linesearch="backtracking"` converges.

---

## Performance

integration_B (two countries, Epstein–Zin, 3 endogenous states, VAR(1) discretized to 9 Markov
states, 8 controls), 20³×9 = 72 000 points, Float64, same options as Dolo.jl's benchmark (GMRES,
linear tolerance 1e-10, backtracking for Newton), 2026-10-21:

| | Dolo.jl, RTX 2080 | dolo.py JAX, RTX 2080 | Dolo.jl, GTX 1660 Ti | dolo.py JAX, GTX 1660 Ti |
|---|---|---|---|---|
| time iteration (243 iterations) | 53.8 s | 42.5 s | 91 s | 70.9 s |
| accelerated time iteration (15) | 8.1 s | 6.7 s | 13.3 s | 11.0 s |
| Newton (6) | 3.62 s | 3.66 s | 5.8 s | 5.76 s |

Kernels on the RTX 2080 (20³×9): $F$ 12.8 ms (Dolo.jl 19.1), $F$ and $dF_1$ 53 ms (Dolo.jl
$F + dF_1$: 85), $dF_2$ 22 ms (51.5), $L.v$ 7.1 ms (5.8), $J\backslash L$ 10.2 ms (13.6),
prefilter 1.5 ms (0.5).

When measuring JAX code: calls are asynchronous (wait with `jax.block_until_ready` before
reading a clock); compilation happens once per array shape (warm up on the grid that is timed);
`jax.transfer_guard("log")` shows the copies between host and device.

Tests: `tests/test_residual_kernels_jax.py` (dependency analysis, coloring, kernels against the
point kernels), `tests/test_solvers_device.py` (device loops against the vectorized engine,
GMRES), `tests/test_df_2.py` (operator).
