# Refactoring Review: dolang → dynspec Migration (`feat/dyno-importer`)

**Date:** 2026-10-17  
**Branch:** `feat/dyno-importer`  
**Base:** `master`

---

## 1. Executive Summary

This refactoring successfully transitions `dolo.py` away from `dolang` toward `dyno.dynspec` for model import, equation parsing, recipe conformity checking, and callable compilation. 

- **Short-term goal achieved:** `ModelV2` runs all core algorithms (`perturb`, `simulate`, `time_iteration`, `improved_time_iteration`, `deterministic_solve`, `egm`, `value_iteration`) without loss of functionality.
- **Legacy syntax retired cleanly:** Explicit timing (`x[t±d]`) replaces historical conveniences (paren-timing `x(1)` and untimed variables defaulting to date $0$).
- **Dual execution paths:** Pure NumPy calls bypass JAX device transfers and dispatch overheads, mitigating latency penalties for CPU-bound inner loops.

---

## 2. Architecture & Design Strengths

1. **Parallel Model Architecture (`ModelV2` & `yaml_import_v2` / `dyno_import`):**
   - Preserves legacy `Model` and `yaml_import` intact as a safe fallback and parity benchmark.
   - Decouples legacy code from modern dynspec-based workflows.

2. **Build-Time Expression Emission (`functions_v2.py: _emit`):**
   - Eliminates AST-walking at JIT trace time.
   - Compiles Lark trees once at model build time into nested expression closures, producing clean arithmetic graphs with hand-written JAX tracing quality.

3. **Dual Execution Closures (`NP_FUNCTIONS` vs. `JNP_FUNCTIONS`):**
   - Emits expressions into both NumPy and JAX backends.
   - Point calls and row-vectorized NumPy evaluations bypass device marshaling (~470 µs → ~10–20 µs), preserving performance for SciPy-style algorithm loops.

4. **Rigorous Parity & Regression Contract (`tests/test_model_v2.py`):**
   - 57 passing tests verifying mathematical parity against legacy models, row-vectorized shape contracts, derivative parity, and full algorithmic convergence.

5. **Recipe-Driven Conformity & Complementarity (`dyno.dynspec.recipe`):**
   - Native parsing of `⟂` (and legacy `|`) complementarities via dynspec's AST.
   - Recipe-driven positional checking and automatic inference of controls and state boundaries.

---

## 3. Code Quality: Identified Issues & Latent Edge Cases

### High / Medium Priority

1. **Hardcoded Arity Binding (`functions_v2.py: _with_arity`):**
   - Only arities 3, 4, 5, 7, and 9 are currently implemented. If an equation group or recipe layout requires a different arity (e.g. 6, 8, or $\ge 10$), it raises `NotImplementedError`.
   - *Recommendation:* Generate callable signatures dynamically via `inspect.Signature` or dynamic wrapper compilation to support arbitrary arities.

2. **Variadic Math Operations in `_emit` (`functions_v2.py:414`):**
   - `max` and `min` map to `np.maximum` / `jnp.maximum`.
   - `np.maximum(x1, x2, out=...)` only accepts two positional operands. Expressions with $>2$ arguments (e.g. `max(a, b, c)`) fail because the 3rd argument is treated as the `out` parameter.
   - *Recommendation:* Wrap variadic `min`/`max` in `_emit` with `functools.reduce(fn, evaluated_args)`.

3. **Asymmetric Vectorized `diff=True` Dispatch (`functions_v2.py:717-749`):**
   - NumPy row-vectorized evaluations use finite differences (`_np_vec_diff`), which supports `deterministic_solve`.
   - JAX row-vectorized inputs raise `NotImplementedError("diff=True is not implemented for vectorized arguments.")`.
   - *Recommendation:* Either document this limitation clearly or implement vectorized Jacobians for JAX via `jax.vmap(jax.jacrev(...))`.

4. **Empty Sequence Risk in `_normalize_batched` (`functions_v2.py:658`):**
   - `N = max(a.shape[0] for a in args if a.ndim == 2)` raises `ValueError` if no argument has `ndim == 2`.
   - *Recommendation:* Add `default=1` (consistent with `_np_batched`).

5. **Repeated Lark AST Parsing in Calibration Solver (`model_v2.py:226`):**
   - `_solve_symbolic_dict` re-parses pending equations with `parser.parse()` on every triangular resolution iteration, and parses them again inside `_eval_scalar`.
   - *Recommendation:* Parse pending expressions into Lark trees once upfront.

6. **Cache Invalidation on `set_calibration` (`model_v2.py:652-658`):**
   - `set_calibration` invalidates `__calibration__` and `__exogenous__`, but not `__domain__` or `__options__` if they were already resolved with old parameter values.
   - *Recommendation:* Ensure all derived state depending on calibration is cleared on update.

---

## 4. API Canonicity & Ecosystem Alignment

### Dolo Compatibility
- **Duck-Typing Parity:** `ModelV2` preserves all public attributes and properties expected by `dolo.algos`: `symbols`, `variables`, `equations`, `definitions`, `calibration` (`CalibrationDict`), `exogenous`, `domain`, `endo_grid`, `discretize`, `residuals`, `x_bounds`, and `infos`.
- **`LoosyDict` Aliasing:** Safely maps `equilibrium`/`optimality` $\to$ `arbitrage` and fallback bounds without hijacking explicit `controls_lb`/`controls_ub` groups.
- **Value Iteration Fixes:** Reading bounds via `model.x_bounds` and clamping out-of-domain state extrapolation makes `value_iteration` canonical and robust across both importers.
- **Display Canonicity:** Terminal `__str__` matches legacy colored residual tables; `_repr_html_` renders clean KaTeX formulas directly via dynspec.

### Dyno Convergence
- **`.dyno` Grammar Alignment:** Block tags (`[definitions]`, `[transition]`, `[arbitrage]`, etc.) and header metadata (`@name:`, `@options:`, `@domain:`) are directly supported.
- **Explicit Timing:** Transitioning model files and notebooks to explicit `x[t±d]` notation removes a major syntax rift between `dolo` and `dyno`.

---

## 5. Actionable Next Steps

1. **Short Term:**
   - Generalize `_with_arity` in `functions_v2.py`.
   - Fix variadic `min`/`max` handling in `_emit`.
   - Add `default=1` guard to `_normalize_batched`.
   - Pre-parse expressions in `_solve_symbolic_dict`.

2. **Medium Term (Eliminating Temporary Shims):**
   - **`equation_preprocess.py`:** Delete once dynspec grammar natively accepts `==` as an alias for `=`.
   - **`langspec.py`:** Reimplement `eval_data` for tagged YAML objects (`!Normal`, `!Cartesian`), allowing `dolang` to be completely removed from dependencies.

3. **Long Term (Performance & Backend Choice):**
   - Single-core JAX execution is fast, but Numba gufuncs with `target="parallel"` scale better on large multi-core CPU batches.
   - Because `_emit` generates backend-independent expression closures, consider exposing a backend switch (e.g. `backend="jax"|"numba"`) so users can choose between JAX's differentiability and Numba's multi-threaded CPU throughput.

