"""
Self-contained comparison of the two function-compilation paths.

The same RBC arbitrage function `f(m, s, x, M, S, X, p)` (2 residuals,
with the auxiliary definitions `y, c, rk, w` inlined at dates t and t+1)
is built two ways:

A. **dolang** (the legacy path used by `dolo.compiler.factories`):
   parse equations with the dolang lark grammar -> `FlatFunctionFactory`
   -> generated source -> numba jit + `guvectorize` (a vectorized ufunc)
   -> `standard_function` wrapper (row-vectorization + finite-difference
   `diff=True`).

B. **dynspec + JAX** (the new path used by `dolo.compiler.functions_v2`):
   parse with `dyno.dynspec`'s lark grammar -> recipe conformity checking
   -> `EquationsEvaluator` (a lark-tree interpreter) whose function
   table is replaced by `jax.numpy` primitives -> wrapped in
   `jax.jit`/`jax.vmap` (vectorization) with `jax.jacrev` for
   `diff=True`.

Both functions expose the same calling convention:
    f(m, s, x, M, S, X, p)            -> (N,2) for (N,n) row-vectorized args
    f(m, s, x, M, S, X, p, diff=True) -> value + one jacobian per arg group

Run:  pixi run --environment dev python bench/dolang_vs_dynspec.py
"""

import time

import jax
import numpy as np

# ---------------------------------------------------------------------------
# the model: two arbitrage residuals + four auxiliary definitions
# ---------------------------------------------------------------------------

VARIABLES = {
    "exogenous": ["e_z"],
    "states": ["z", "k"],
    "controls": ["n", "i"],
    "auxiliaries": ["y", "c", "rk", "w"],
    "parameters": ["beta", "sigma", "eta", "chi", "delta", "alpha"],
}

DEFINITIONS = [
    "y[t] = exp(z[t])*k[t]^alpha*n[t]^(1-alpha)",
    "c[t] = y[t] - i[t]",
    "rk[t] = alpha*y[t]/k[t]",
    "w[t] = (1-alpha)*y[t]/n[t]",
]

EQUATIONS = [
    "chi*n[t]^eta*c[t]^sigma - w[t]",
    "1 - beta*(c[t]/c[t+1])^sigma*(1-delta+rk[t+1])",
]

# a numerical point (the rbc steady state)
M0 = np.array([0.0])
S0 = np.array([0.0, 9.35497829])
X0 = np.array([0.33, 0.23387446])
P0 = np.array([0.99, 5.0, 1.0, 23.957859, 0.025, 0.33])
N_ROWS = 7

ALL_VARS = [
    v for g in ("exogenous", "states", "controls", "auxiliaries") for v in VARIABLES[g]
]


def timings_table(name_leg, fn_leg, name_new, fn_new):
    """Time point / vectorized / diff calls; return (headers, rows)."""
    headers = ("call", name_leg, name_new, "ratio (×)")
    rows = []

    def bench(kind):
        if kind == "point":
            return (
                lambda: fn_leg(M0, S0, X0, M0, S0, X0, P0),
                lambda: fn_new(M0, S0, X0, M0, S0, X0, P0),
            )
        if kind == "diff":
            return (
                lambda: fn_leg(M0, S0, X0, M0, S0, X0, P0, diff=True),
                lambda: fn_new(M0, S0, X0, M0, S0, X0, P0, diff=True),
            )
        if kind == "vectorized":
            # larger batch: amortizes jax dispatch/device_put overhead
            n_vec = 1000
            M, S, X = (
                np.tile(M0, (n_vec, 1)),
                np.tile(S0, (n_vec, 1)),
                np.tile(X0, (n_vec, 1)),
            )
            return (
                lambda: fn_leg(M, S, X, M, S, X, P0),
                lambda: fn_new(M, S, X, M, S, X, P0),
            )
        raise ValueError(kind)

    for kind, n_loops in (("point", 300), ("vectorized", 30), ("diff", 20)):
        c_leg, c_new = bench(kind)
        # warm-up (numba/XLA compile)
        c_leg()
        c_new()
        t0 = time.perf_counter()
        for _ in range(n_loops):
            c_leg()
        t_leg = (time.perf_counter() - t0) / n_loops * 1e6
        t0 = time.perf_counter()
        for _ in range(n_loops):
            c_new()
        t_new = (time.perf_counter() - t0) / n_loops * 1e6
        rows.append(
            (kind, f"{t_leg:8.1f} µs", f"{t_new:8.1f} µs", f"{t_new / t_leg:4.1f}")
        )
    return headers, rows


def print_table(headers, rows):
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  " + "  ".join(h.ljust(w[i]) for i, h in enumerate(headers)))
    print("  " + "  ".join("-" * w[i] for i in range(len(headers))))
    for r in rows:
        print("  " + "  ".join(r[i].ljust(w[i]) for i in range(len(r))))


# ---------------------------------------------------------------------------
# A. the dolang path
# ---------------------------------------------------------------------------


def build_dolang_function():
    """Legacy compilation: lark parsing -> source generation -> numba."""
    from dolang.grammar import str_expression
    from dolang.symbolic import (
        Sanitizer,
        parse_string,
        stringify,
        stringify_symbol,
        time_shift,
    )
    from dolang.factory import FlatFunctionFactory
    from dolang.function_compiler import make_method_from_factory
    from dolang.vectorize import standard_function

    sanitize = Sanitizer(variables=ALL_VARS)

    # --- preamble: each definition, shifted to the dates the equations need
    preamble = {}
    for d in DEFINITIONS:
        lhs, rhs = [e.strip() for e in d.split("=", 1)]
        name = lhs.split("[")[0]
        for t in (0, 1):
            preamble[stringify_symbol((name, t))] = str_expression(
                stringify(time_shift(parse_string(rhs), t))
            )

    # --- equations -> content {out_i: expr}
    content = {
        stringify_symbol((f"out{i}", 0)): str_expression(
            stringify(sanitize.transform(parse_string(e)))
        )
        for i, e in enumerate(EQUATIONS)
    }

    # --- arguments: one list per positional group (parameters last)
    spec = [
        ("exogenous", 0, "m"),
        ("states", 0, "s"),
        ("controls", 0, "x"),
        ("exogenous", 1, "M"),
        ("states", 1, "S"),
        ("controls", 1, "X"),
        ("parameters", 0, "p"),
    ]
    arguments = {}
    for group, shift, argname in spec:
        if group == "parameters":
            arguments[argname] = [stringify_symbol(v) for v in VARIABLES[group]]
        else:
            arguments[argname] = [
                stringify_symbol((v, shift)) for v in VARIABLES[group]
            ]

    factory = FlatFunctionFactory(preamble, content, arguments, "arbitrage")
    # `fun` is a numba-jitted plain function; `gufun` is a numba guvectorize
    # (a true vectorized ufunc) with signature (n_m),(n_s),...->(n_out).
    fun, gufun = make_method_from_factory(factory, vectorize=True, debug=False)
    return standard_function(gufun, n_output=len(content)), factory


# ---------------------------------------------------------------------------
# B. the dynspec + JAX path
# ---------------------------------------------------------------------------


JNP_FUNCTIONS = {
    "exp": jax.numpy.exp,
    "log": jax.numpy.log,
    "sqrt": jax.numpy.sqrt,
    "abs": jax.numpy.abs,
    "sin": jax.numpy.sin,
    "cos": jax.numpy.cos,
    "tan": jax.numpy.tan,
    "sinh": jax.numpy.sinh,
    "cosh": jax.numpy.cosh,
    "tanh": jax.numpy.tanh,
    "asin": jax.numpy.arcsin,
    "acos": jax.numpy.arccos,
    "atan": jax.numpy.arctan,
    "max": jax.numpy.maximum,
    "min": jax.numpy.minimum,
    "log10": jax.numpy.log10,
    "log2": jax.numpy.log2,
    "floor": jax.numpy.floor,
    "ceil": jax.numpy.ceil,
    "pow": jax.numpy.pow,
}


def build_jax_function():
    """New path: dynspec parsing + recipe check -> JAX-evaluated kernel."""
    import jax
    import jax.numpy as jnp
    from dyno.dynspec.grammar import parser
    from dyno.dynspec.recipe import (
        DTCC_RECIPE,
        check_equation_group,
        extract_lhs_variable,
        extract_variables_from_equation,
    )
    from dyno.dynspec.analyze import EquationsEvaluator
    from dyno.dynspec.funcgen import build_definitions_block

    jax.config.update("jax_enable_x64", True)

    # --- parse definitions and equations into lark trees
    def parse_block(lines):
        return parser.parse("\n".join(lines), start="equation_block").children

    def_trees = parse_block(DEFINITIONS)
    eq_trees = parse_block(EQUATIONS)
    params = VARIABLES["parameters"]
    def_spec = DTCC_RECIPE.get_group_spec("definitions")
    f_spec = DTCC_RECIPE.get_group_spec("arbitrage")
    def_result = check_equation_group(def_trees, VARIABLES, def_spec, constants=params)
    f_result = check_equation_group(eq_trees, VARIABLES, f_spec, constants=params)
    assert def_result.ok, def_result
    assert f_result.ok, f_result

    def_result_block, defs_block = build_definitions_block(
        def_trees, VARIABLES, def_spec, constants=params
    )

    # --- interpretation kernel: one pass over the trees with jnp primitives
    layout = [
        ("exogenous", 0),
        ("states", 0),
        ("controls", 0),
        ("exogenous", 1),
        ("states", 1),
        ("controls", 1),
    ]

    def eval_definitions_at(ctx_vars, constants, shift):
        for eq in defs_block.equations:
            name = extract_lhs_variable(eq)
            staged = {}
            for v, shifts in ctx_vars.items():
                if shift in shifts:
                    staged[v] = {0: shifts[shift]}
            sub = {"constants": constants, "variables": staged}
            ev = EquationsEvaluator(sub, function_table={}, assign=True)
            ev.function_table.update(JNP_FUNCTIONS)
            ev.visit(eq)
            ctx_vars.setdefault(name, {})[shift] = staged[name][0]

    def core(m, s, x, M, S, X, p):
        # next date
        ctx_vars = {}
        for (group, shift), arr in zip(layout, (m, s, x, M, S, X)):
            for j, v in enumerate(VARIABLES[group]):
                ctx_vars.setdefault(v, {})[shift] = arr[j]
        constants = dict(zip(params, p))
        constants["inf"] = np.inf
        # inline the definitions at dates 0 and 1
        for shift in (0, 1):
            eval_definitions_at(ctx_vars, constants, shift)
        context = {"constants": constants, "variables": ctx_vars}
        evaluator = EquationsEvaluator(context=context, function_table={}, assign=False)
        evaluator.function_table.update(JNP_FUNCTIONS)
        return jnp.stack([evaluator.visit(eq) for eq in eq_trees])

    point = jax.jit(core)
    batch = jax.jit(jax.vmap(core))

    # cached jacrev over the concatenated variable arguments
    sizes = [len(VARIABLES[g]) for g, _ in layout]

    def split_eval(v, p):
        parts, k = [], 0
        for sz in sizes:
            parts.append(v[k : k + sz])
            k += sz
        return core(*parts, p)

    jacobian = jax.jit(jax.jacrev(split_eval, argnums=0))
    cum = np.cumsum(sizes)

    def f(*args, diff=False):
        jargs = [jnp.asarray(a, dtype=jnp.float64) for a in args]
        batched = any(a.ndim == 2 for a in jargs)
        if batched:
            if diff:
                raise NotImplementedError("diff=True is point-only here")
            N = max(a.shape[0] for a in jargs if a.ndim == 2)
            bargs = [
                jnp.broadcast_to(a, (N, a.shape[0])) if a.ndim == 1 else a
                for a in jargs
            ]
            return np.asarray(batch(*bargs))
        if diff:
            v = jnp.concatenate(jargs[:-1])
            J = jacobian(v, jargs[-1])
            parts = np.split(np.asarray(J), cum[:-1], axis=1)
            return [np.asarray(point(*jargs))] + list(parts)
        return np.asarray(point(*jargs))

    return f, (def_result, f_result, defs_block)


# ---------------------------------------------------------------------------
# run it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 66)
    print("One arbitrage function, two compilation paths")
    print("=" * 66)
    print("\nequations:")
    for e in EQUATIONS:
        print("   ", e)
    print("definitions:", DEFINITIONS)
    print(f"calibration point: m={M0}, s={S0}, x={X0}, p={P0}")

    print("\n" + "=" * 66)
    print("A. dolang: parse -> FlatFunctionFactory -> numba guvectorize")
    print("=" * 66)
    f_dolang, factory = build_dolang_function()
    print("\n  factory arguments (positional groups):")
    for k, v in factory.arguments.items():
        print(f"    {k}: {v}")
    print("\n  factory content (residual expressions):")
    for k, v in factory.content.items():
        print(f"    {k} = {v}")
    print(
        "\n  f_dolang is:",
        type(f_dolang).__name__,
        "| inner ufunc:",
        type(f_dolang.fun).__name__,
    )
    print("  point   :", f_dolang(M0, S0, X0, M0, S0, X0, P0))

    print("\n" + "=" * 66)
    print("B. dynspec + JAX: parse -> recipe check -> interpreted jnp kernel")
    print("=" * 66)
    f_jax, (def_res, f_res, defs_block) = build_jax_function()
    from dyno.dynspec.grammar import parser

    first = parser.parse(EQUATIONS[0], start="equation_block").children[0]
    print("\n  lark parse-tree of equation 1 (dynspec grammar):")
    for line in first.pretty().splitlines()[1:]:
        print("   ", line)
    print("\n  conformity results:")
    print("   ", str(def_res).replace("\n", "\n    "))
    print("   ", str(f_res).replace("\n", "\n    "))
    print(f"\n  definitions block (DAG order): {defs_block.variable_names}")
    print("  point   :", f_jax(M0, S0, X0, M0, S0, X0, P0))

    # --- agreement checks
    M, S, X = (
        np.tile(M0, (N_ROWS, 1)),
        np.tile(S0, (N_ROWS, 1)),
        np.tile(X0, (N_ROWS, 1)),
    )
    v_leg = f_dolang(M, S, X, M, S, X, P0)
    v_jax = f_jax(M, S, X, M, S, X, P0)
    d_leg = f_dolang(M0, S0, X0, M0, S0, X0, P0, diff=True)
    d_jax = f_jax(M0, S0, X0, M0, S0, X0, P0, diff=True)

    print("\n" + "=" * 66)
    print("agreement (same inputs)")
    print("=" * 66)
    print(f"  values {N_ROWS}x2 equal      : {np.allclose(v_leg, v_jax)}")
    for i, (a, b) in enumerate(zip(d_leg[1:], d_jax[1:])):
        print(f"  jacobian #{i} (analytic vs fd): max diff {np.abs(a - b).max():.2e}")

    print("\n" + "=" * 66)
    print("timing (warm, per call)")
    print("=" * 66)
    headers, rows = timings_table("dolang", f_dolang, "dynspec+jax", f_jax)
    print_table(headers, rows)
    print(
        "\n  note: the point-call gap is dominated by jax.device_put of the\n"
        "  small input arrays (~60us each on this CPU jaxlib), not by the\n"
        "  compiled kernel; vectorized calls amortize it."
    )
