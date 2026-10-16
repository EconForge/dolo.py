# dolo.py — dolang → dynspec transition: status & handover notes

Branch: `feat/dyno-importer` (all the work so far is on this branch, on top of `master`).

Date of last update: 2026-10-16. Read this instead of re-reading the whole conversation.

---

## 1. The goal and the design decisions (agreed with Pablo)

Rewrite dolo's model *import* so that:

- D1 — dolo depends on **dyno.py** (`dynopy`) and uses `dyno.dynspec` for
  equation parsing, conformity checking and recipes. During development
  pixi uses the local checkout at `../dyno.py` as an editable path
  dependency; `min-deps` pins the published `dynopy==0.1.13`.
  **Do not touch dyno.py code for now.**
- D2 — equation evaluation/differentiation uses **JAX**. Two stages
  happened: (a) `EquationsEvaluator` walking lark trees at trace time
  ("interpreted"), then (b) expression **emission** (`_emit`): each tree
  is compiled once at build time into a closure assembling the arithmetic
  directly, so the traced jaxpr is a plain nested expression graph
  (hand-written quality).
- D3 — a **second model object** (`ModelV2`) with the SAME API as the
  legacy `Model`, so `dolo.algos` run unchanged. The legacy importer
  stays intact and is the fallback.
- D4 — dependencies: jax, `dynopy`, numba are real runtime dependencies.
  **dolang is NOT removed**; it stays as a dependency, with new code
  going through the temporary shim `dolo.compiler.langspec`.
- Long-run file format: dolo models may be written as `.dyno` files where
  **tags denote equation types** (`[definitions] :: {…}`,
  `[transition] :: {…}`, `[arbitrage] :: {…}`, `[value] :: {…}`, …),
  with `@name:`, `@options:`, `@domain:` one-liners. `@` and YAML
  sections are equivalent (`@name: X` == `name: X`).
- Equation parsing must eventually be 100% dynspec, with `⟂` added to the
  dynspec grammar. Everything else is flagged TEMPORARY.

## 2. What exists now (files)

Under `src/dolo/compiler/`:

| file | status | content |
|---|---|---|
| `model_v2.py` | new | `ModelV2` with `from_yaml(...)` (composed node tree) and `from_dyno(...)` (`.dyno` via `dyno.larkfiles.DynoFile`). API: `symbols`, `variables`, `equations`, `definitions`, `calibration`, `exogenous`, `domain`, `endo_grid`, `discretize`, `functions`, `x_bounds`, `residuals`, `eval_formula`, `infos`, `__str__`. Calibration solved iteratively with the dynspec `FormulaEvaluator` (`_solve_symbolic_dict`). `.dyno` symbol inference: parameters = top-level assignments, states = transition LHGs, auxiliaries = definitions LHGs, controls = free vars in any group (arbitrage scanned first). |
| `functions_v2.py` | new | JAX function backend. `GROUP_LAYOUTS`: transition, arbitrage, auxiliary, arbitrage_lb/ub, controls_lb/ub, value, felicity, expectation, half_transition, direct_response, direct_response_egm, reverse_state, arbitrage_exp (all mirror `dolo/compiler/recipes.yaml` positional signatures, p last). `_emit(tree)` compiles lark trees once at build time into **backend-independent expression closures**, instantiated twice: jnp (jit/vmap/jacrev kernels) and numpy (low-latency point/batched path for numpy inputs). `make_function` builds `jax.jit` point kernel, `jax.vmap` batch kernel, point path via batch-of-1 with p unmapped, and `diff=True` via one cached `jax.jacrev` over concatenated variable args. `check_equation_group`/`build_definitions_block` from dynspec do conformity + DAG. |
| `equation_preprocess.py` | TEMP | Normalizes dolo-only syntax before dynspec parsing: `⟂` / legacy `|` complementarity splitting, `==`→`=`, `x(1)` timing, untimed symbols → `[t]`. To be deleted once dynspec supports it natively. |
| `langspec.py` | TEMP | Choke point around dolang: currently re-exports only `eval_data` (the single dolang utility the new path still needs); gone once that call site is reimplemented. |
| `model_import.py` | extended | `yaml_import` unchanged; new `yaml_import_v2` and `dyno_import`. Exported from `dolo/__init__.py`. |
| `model.py`, `factories.py`, `recipes.py/.yaml` … | legacy | untouched, still the default and the fallback. |

Algo-compatibility adapters (in `ModelV2`, no algo changes):
- `functions` is a `LoosyDict` with equivalences `equilibrium`→arbitrage,
  `optimality`→arbitrage, `controls_lb`→arbitrage_lb, `controls_ub`→arbitrage_ub
  (so `value_iteration` finds bounds). `LoosyDict.__getitem__` resolves
  **real keys before equivalences**, so standalone `controls_lb`/
  `controls_ub` equation groups are not hijacked by the aliases.
- `ModelV2.functions` also sets `__original_functions__` and
  `__original_gufunctions__` (egm reads the latter); the gufunctions
  are our allocating + returning functions (compatible).
- `endo_grid` accepts both `options.grid.n` and legacy `options.grid.orders`.
- Standalone bounds groups (`controls_lb:`/`controls_ub:` sections, e.g.
  `controls_lb: {i = 0}`) build real functions with signature
  (m[t], s[t], p) — dolo's documented convention; only `ModelV2` supports
  them (the legacy importer has no recipe for them).

Entry points / how to try:

```python
from dolo import yaml_import_v2, dyno_import
m = yaml_import_v2("examples/models/rbc.yaml")        # classic YAML
d = dyno_import("examples/dyno/rbc_dolo.dyno")        # .dyno with tags
```

`.dyno` examples: `examples/dyno/rbc_dolo.dyno` (copy of dyno.py's demo);
`examples/dyno/rbc_dolo_comp.dyno` is a **target-syntax mock** (`⟂`
complementarities) for the dyno-side grammar work — not yet parseable.

## 3. Tests

`tests/test_model_v2.py` is the *migration contract*:

- parity with the legacy importer for all `examples/models/*.yaml`
  (symbols, calibration, residual functions),
- **every** function group compared against the legacy function with
  args reconstructed from `recipes.yaml` specs (incl. value/felicity on
  a synthetic model string),
- `diff=True` jacobians vs legacy finite-diff (atol 1e-5); pole-tolerant
  comparison (`±inf`/`nan` treated alike),
- **batched-call shape regression** (DEV_NOTES §6.2): every group returns
  exactly `(N, n_out)` for row-vectorized args, with parameters either
  tiled or left 1-d (unmapped) — guards the p-broadcast bug class;
  inconsistent batch sizes raise.
- **algorithm smoke tests on ModelV2** (§6.3): `perturb` + `simulate`
  (identical driving process → near-bit-equal paths), `time_iteration`
  (5×5 grid → decision rules equal to 1e-8), `improved_time_iteration`
  (decision rules equal), `perfect_foresight`/`deterministic_solve`
  (bit-identical trajectories via vectorized `diff=True`), `egm`
  (decision rule exactly equal), `value_iteration` (bounded growth
  model, converged, bit-identical policy on both importers; also runs
  on standalone `controls_lb/controls_ub` groups).
- vectorized `diff=True` (row-vectorized args) matches legacy `numdiff`
  bit-for-bit (jacobians for every argument but p).
- `.dyno` import, `@options`/`@domain` handling, and `discretize()` on
  `.dyno` models.
- standalone `controls_lb`/`controls_ub` groups (real functions, not
  aliases) + alias still working when the group is absent.
- `eval_formula` on ModelV2 against the legacy evaluator (timing
  notation + dataframe shifts).
- `_repr_html_` renders equations with dynspec latex.

Full suite: `pixi run --environment dev pytest -q` → 54 passed.

Previously verified manually (now covered by the tests above):
`time_iteration`, `perturb`, `simulate` and `egm` run on `ModelV2` and
produce identical results to legacy.

## 4. Dependencies (pixi/pyproject)

- `pyproject.toml` added: `dynopy>=0.1.13,<0.2`, `jax>=0.4.38,<0.5`,
  `numba>=0.60.0,<0.70`.
- `pixi.toml`:
  - `[pypi-dependencies]`: `dolo` + `dynopy = { path = "../dyno.py",
    editable = true }` + `jax >=0.4.38,<0.5`.
  - `[feature.min-deps.pypi-dependencies]`: `dynopy ==0.1.13`, `jax ==0.4.38`.
  - conda `[dependencies]`: `rich = "<15"` (the Jupyter/MkDocs stack pins
    conda rich 15 while dynopy needs rich<15).
- Why jax is on PyPI and capped `<0.5`: conda-forge has no jaxlib for
  win-64; jaxlib dropped macosx_x86_64 wheels in 0.5.0 and the workspace
  still locks osx-64. jaxlib 0.4.38 has wheels for every platform×py312/313.
- `ci/check_deps.py` now understands `[pypi-dependencies]` (incl.
  min-deps pypi) and table-valued path entries; verified green.

## 5. Performance: everything learned

### Measurement pitfalls (fixed in the benches)
- `block_until_ready()` must be enforced — XLA defers CPU work; without it
  jax looked 30× too fast.
- dtype: hand-written reference ran f32 while everything else f64;
- broadcast-tiled (identical) input rows distort timings; use random rows;
- `jnp.array[None, :]` fancy indexing outside jit costs ~200µs/slice
  (JIT-rewriting take path) — expand inside the traced code;
- `jax.device_put` of a *small* numpy array costs ~55-65µs on this CPU
  jaxlib — it dominates point calls fed from numpy (the algos do that);
- vmap wrapper must not broadcast `p` if the kernel declares it unmapped
  (it produced (N, n_p, n_eq) outputs silently — caught while reworking the
  bench; production `functions_v2` maps all args incl. p, so unaffected);
- jit cache is per-shape: new batch sizes retrace+recompile (~35-55ms).

### The actual ranking (5M×2 rows, f64, random, sync-forced, 8-core box)

| variant | 5M rows | 1 core (`taskset`) |
|---|---:|---:|
| numba gufunc single-thread | 476 ms | — |
| numba gufunc `target="parallel"` (what dolang ships) | 67 ms | 476 ms (7.3× scaling) |
| jax jit (emitted == handcoded quality) | 208-233 ms | 627 ms (2.8× scaling) |

Conclusions:

1. **Kernel quality is fine**: single-thread, jax ≈ 2-2.8× faster than
   numba (user's independent bench — 107 vs 38 ns/pt — agrees).
2. The ranking flips only because dolang's guvectorize uses
   `target="parallel"` (prange over rows, 7-8× scaling) while XLA-CPU's
   fusion-level parallelism reaches ~2.8× on this workload.
3. Point-call latency — **numpy fast path implemented**: point *and*
   row-vectorized calls made entirely from numpy arrays now evaluate the
   same emitted expressions through numpy closures (no device transfers,
   no jit dispatch): `f` point 23µs (was ~470, legacy 5.5), `g` point
   9µs (was ~470, legacy 4.9). Macro effect: `time_iteration` on a 5×5
   grid 14.5s → 6.8s, `egm` 6.6s → 2.6s. JAX is still used for jax
   array inputs, tracers and `diff=True`.
   Still open: default-grid `time_iteration` (100×100) at 28.5s vs
   legacy 20.8s — residual dominated by shared numba/dispatch infra plus
   numpy-batched kernels being ~6× the numba gufunc at 10k rows (jax
   beats numpy only above ~1M rows). Closes either with pre-placed
   device arrays or the backend choice below.
4. `diff=True`: legacy 48µs (finite-diff on the compiled kernel), emitted
   ~0.6ms (one cached jacrev) — acceptable, only called in outer loops
   (perturb setup, perfect_foresight, euler residuals).

### Open performance question
> "It might be that we will still offer a numba option after all…"

Given (1)-(2), a reasonable end state is a **backend choice**:
`ModelV2.functions` selecting between (a) JAX kernels and (b) numba
gufuncs compiled from the *same emitted expression source*. The natural
implementation: extend `emit()` to target either jnp closures or numba
source/`numba.njit` + `guvectorize(target=...)`, behind one switch
(e.g. `ModelV2(..., backend="jax"|"numba")`). Decision not yet taken.

Benchmarks to run/inspect:
- `bench/dolang_vs_dynspec.py` — the self-contained side-by-side
  (gufunc source, handcoded source, 1-thread vs 8-thread columns,
  micro + large-batch tables, rich, warnings silenced).
- `bench/bench_functions.py` — legacy Model vs ModelV2 micro
  (+ `--macro` time_iteration).

## 6. Known gaps / next steps

1. **value_iteration** (fixed 2026-10-16):
   - *Bounds aliasing*: fixed earlier. `LoosyDict.__getitem__` used to
     resolve equivalences *before* real keys, so explicit `controls_lb`
     groups were hijacked by the `controls_lb`→`arbitrage_lb` alias.
     Real keys now win; regression test in `test_standalone_bounds_groups`.
   - *Algorithm-side fixes* (in `src/dolo/algos/value_iteration.py`,
     benefit both importers):
     - bounds are read through `model.x_bounds` instead of hardcoded
       `functions["controls_lb"]` — legacy models with ⟂
       complementarities can now run value_iteration at all;
     - the initial constant policy is clipped to [lb, ub] on the grid
       (the calibrated controls may be infeasible and drive states
       negative);
     - the Bellman operator clips transition states to the grid box
       before interpolation (`update_value`, `choice_value`), so
       boundary-exploring policies can't feed unbounded polynomial
       extrapolation into the operator — previously diverged to ~1e280.
   - Result: on a bounded growth model both importers converge
     (v_err ~1e-8, x_err 0) with **bit-identical** policies and value
     functions (`test_value_iteration_parity`). Still model-side: value
     iteration is only well-posed with upper bounds on controls and a
     domain covering the attainable states.
2. **Shape-regression tests**: done (see §3).
3. **Algorithm smoke tests on ModelV2** (time_iteration/perturb/simulate/
   improved_time_iteration/deterministic_solve/egm/value_iteration): all
   done (§3).
4. **Wrapper marshaling**: done — plain-numpy point/batched calls go
   through the numpy closure family; already-device `jax float64` arrays
   skip `jnp.asarray(dtype=...)`. Returning device arrays behind an opt-in
   is still open.
5. **Point-call path**: done via the numpy closure family (§5.3). Optional
   follow-ups: pre-placed device arrays, or the numba backend choice.
6. **dyno.py-side grammar work** (in a dyno.py branch, later): add `⟂`
   (+ maybe dolo timing conventions) and unify recipes, then delete
   `equation_preprocess.py` and `langspec.py`.
   **Handover ready**: `examples/dyno/rbc_dolo_comp.dyno` (target-syntax
   mock, does not parse yet) + `docs/dynospec_requirements.md` (exact
   grammar/API/recipe changes + acceptance criteria); written to hand to
   a dyno-side agent. Decision recorded there: untimed variables are a
   **non-goal** for dynospec (a name without `[t]` stays a constant);
   that normalization remains a dolo-YAML-legacy shim on our side.
7. **ModelV2 display**: `_repr_html_` added on top of dynspec latex
   (`x[t-1]` → `x_{t-1}`, spelled greek → greek commands, residuals
   column); `__str__` follows the legacy layout (header, equations with
   `N : residual : equation` rows, definitions block, red non-zero
   residuals); no dolang.
8. `eval_formula` on ModelV2 moved to the dynspec evaluator
   (`dyno.dynspec.FormulaEvaluator`): `k(0)`-style timing rewritten by
   `equation_preprocess.rewrite_timing_calls`, timed names resolved as
   shifted dataframe columns, untimed names as calibration constants,
   `k[t±d]` dynspec timing accepted. Parity test added. The legacy
   `Model.eval_formula` (dolang-based) stays as is.
   Greek/symbol-key normalization needs nothing dynspec-side: the
   legacy `stringify_symbol` only appends dolang's trailing `_`
   convention, which dynspec never uses (greek letters are native
   grammar names and calibration keys match symbols verbatim).
   Also: `langspec.py` trimmed to the single used export (`eval_data`);
   the module disappears when that one call site is reimplemented.
9. `discretize()` on `.dyno` models requires `@domain`; test added
   (`test_dyno_options_and_domain`). Old `Model` and `dolo.yaml_import`
   remain the default — decide when/if to switch.
10. min-deps CI only pins jax from PyPI; keep `check-deps` green when
    adding/removing deps (run `python3 ci/check_deps.py`).

## 7. Everyday commands

```bash
pixi install --environment dev
pixi run --environment dev pytest -q
pixi run --environment dev python bench/dolang_vs_dynspec.py
pixi run --environment dev python bench/bench_functions.py --macro
python3 ci/check_deps.py
```

Commit trail on `feat/dyno-importer` (most → least recent):
f0828bb __str__ polish · 675bfbb langspec trim · 5132e09 vectorized diff ·
219f9de value_iteration fixes · 29a277f numpy closure family ·
8a510c8 dynspec-latex _repr_html_ · 434faa0 dynspec eval_formula ·
f987b3e algo smoke + shape tests, bounds alias fix ·
b2df39b threading decomposition · 742395d rich example + p-broadcast fix ·
aa9b427 handcoded ref · b66355b emitter + honest benches · 81fc9b3 5M scaling ·
2e72cdc side-by-side example · 468806e algo compat + kernel speedups ·
261e605 all dtcc groups + @options · b53e454 ModelV2 initial.
