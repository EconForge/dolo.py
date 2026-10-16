"""
Self-contained comparison of three compilation paths for one RBC arbitrage
function ``f(m, s, x, M, S, X, p)`` (two residuals; auxiliary definitions
``y, c, rk, w`` inlined at dates t and t+1):

1. **dolang** (legacy): lark parsing -> ``FlatFunctionFactory`` ->
   generated source -> numba ``guvectorize`` ufunc (warm).  The generated
   source is printed (dolang's ``make_method_from_factory(debug=True)``).
2. **dynspec + JAX, emitted**: parsing + recipe conformity checking by
   ``dyno.dynspec``, then each equation tree is compiled ONCE (build time)
   into a closure assembling the jnp arithmetic expression directly — the
   same emission stage as ``src/dolo/compiler/functions_v2.py::_emit`` —
   and wrapped in ``jax.jit``/``jax.vmap``.
3. **handcoded JAX**: the same arithmetic written by hand and jitted; its
   source is printed for side-by-side inspection.

All timings are warm, sync-forced (``block_until_ready``), on vector data;
the large-batch scaling table uses f64 random distinct rows.  Output is
rendered with `rich` and warnings are silenced.

Run:  pixi run --environment dev python bench/dolang_vs_dynspec.py
"""

import contextlib
import inspect
import io
import logging
import time
import warnings

import numpy as np

# --- silence deprecation / backend warnings ------------------------------
warnings.filterwarnings("ignore")
for _name in ("jax", "jax._src.xla_bridge", "absl", "tsl"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)
try:  # absl uses its own logger
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.FATAL)
except Exception:
    pass

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# the model
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

M0 = np.array([0.0])
S0 = np.array([0.0, 9.35497829])
X0 = np.array([0.33, 0.23387446])
P0 = np.array([0.99, 5.0, 1.0, 23.957859, 0.025, 0.33])
N_ROWS = 7

ALL_VARS = [
    v for g in ("exogenous", "states", "controls", "auxiliaries") for v in VARIABLES[g]
]

LAYOUT = [
    ("exogenous", 0),
    ("states", 0),
    ("controls", 0),
    ("exogenous", 1),
    ("states", 1),
    ("controls", 1),
]


def sync(call):
    """Run a call and force device completion (jax may defer CPU work)."""
    res = call()
    if hasattr(res, "block_until_ready"):
        res.block_until_ready()
    return res


def time_it(call, n):
    """Warm-then-time: best-of-3 average over n loops, in microseconds."""
    sync(call)
    best = np.inf
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n):
            sync(call)
        best = min(best, (time.perf_counter() - t0) / n)
    return best * 1e6


# ---------------------------------------------------------------------------
# 1. dolang path
# ---------------------------------------------------------------------------


def build_dolang_function():
    """Legacy compilation: parse -> FlatFunctionFactory -> numba gufunc."""
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

    preamble = {}
    for d in DEFINITIONS:
        lhs, rhs = [e.strip() for e in d.split("=", 1)]
        name = lhs.split("[")[0]
        for t in (0, 1):
            preamble[stringify_symbol((name, t))] = str_expression(
                stringify(time_shift(parse_string(rhs), t))
            )

    content = {
        stringify_symbol((f"out{i}", 0)): str_expression(
            stringify(sanitize.transform(parse_string(e)))
        )
        for i, e in enumerate(EQUATIONS)
    }

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

    # `debug=True` makes dolang print the generated python source:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _, gufun = make_method_from_factory(factory, vectorize=True, debug=True)
    generated_source = buf.getvalue()

    # A second, SINGLE-THREADED numba variant of the same generated source:
    # dolang's gufun uses `target="parallel"` (prange over rows); plain
    # numba.guvectorize defaults to one thread.  Keeping both isolates the
    # effect of threading.
    import numba

    ns = {
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "min": np.minimum,
        "max": np.maximum,
    }
    exec(generated_source, ns)
    fty = "void(f8[:],f8[:],f8[:],f8[:],f8[:],f8[:],f8[:],f8[:])"
    sig = "(a),(n),(n),(a),(n),(n),(q)->(n)"
    gufun_1thread = numba.guvectorize([fty], sig)(ns["arbitrage"])

    return (
        standard_function(gufun, n_output=len(content)),
        standard_function(gufun_1thread, n_output=len(content)),
        factory,
        generated_source,
    )


# ---------------------------------------------------------------------------
# 2. dynspec + JAX emitted path (same emission as functions_v2._emit)
# ---------------------------------------------------------------------------

_JNP = {
    "exp": __import__("jax").numpy.exp,
    "log": __import__("jax").numpy.log,
    "sqrt": __import__("jax").numpy.sqrt,
    "abs": __import__("jax").numpy.abs,
    "sin": __import__("jax").numpy.sin,
    "cos": __import__("jax").numpy.cos,
    "tan": __import__("jax").numpy.tan,
    "sinh": __import__("jax").numpy.sinh,
    "cosh": __import__("jax").numpy.cosh,
    "tanh": __import__("jax").numpy.tanh,
    "asin": __import__("jax").numpy.arcsin,
    "acos": __import__("jax").numpy.arccos,
    "atan": __import__("jax").numpy.arctan,
    "max": __import__("jax").numpy.maximum,
    "min": __import__("jax").numpy.minimum,
    "log10": __import__("jax").numpy.log10,
    "log2": __import__("jax").numpy.log2,
    "floor": __import__("jax").numpy.floor,
    "ceil": __import__("jax").numpy.ceil,
    "pow": __import__("jax").numpy.pow,
}


def emit(tree, assign=False):
    """Compile one dynspec tree into ``fn(ctx) -> value`` (build-time).

    Mirrors ``src/dolo/compiler/functions_v2.py::_emit``: `ctx` maps
    variable `(name, shift)` keys to arrays and plain-name keys to
    constants; no lark walking remains in the traced program.
    """
    data = tree.data
    c = tree.children
    if data == "number":
        v = float(c[0])
        return lambda ctx: v
    if data == "constant":
        name = str(c[0].children[0])
        return lambda ctx: ctx[name]
    if data == "variable":
        name = str(c[0].children[0])
        shift = int(str(c[2].children[0])) if c[2].children else 0
        return lambda ctx: ctx[(name, shift)]
    if data in ("add", "sub", "mul", "div"):
        a, b = emit(c[0]), emit(c[1])
        if data == "add":
            return lambda ctx: a(ctx) + b(ctx)
        if data == "sub":
            return lambda ctx: a(ctx) - b(ctx)
        if data == "mul":
            return lambda ctx: a(ctx) * b(ctx)
        return lambda ctx: a(ctx) / b(ctx)
    if data == "neg":
        a = emit(c[0])
        return lambda ctx: -a(ctx)
    if data == "pow":
        a, b = emit(c[0]), emit(c[1])
        return lambda ctx: a(ctx) ** b(ctx)
    if data == "call":
        fname = str(c[0].children[0])
        args = [emit(arg) for arg in c[1:]]
        fn = _JNP[fname]
        return lambda ctx: (
            fn(*[a(ctx) for a in args]) if len(args) > 1 else fn(args[0](ctx))
        )
    if data == "equality":
        if assign:
            return emit(c[1])
        lhs, rhs = emit(c[0]), emit(c[1])
        return lambda ctx: rhs(ctx) - lhs(ctx)
    if data == "bare_formula":
        return emit(c[0])
    raise NotImplementedError(f"node {data}")


def build_emitted_function():
    """dynspec parsing + recipe check -> emitted jnp kernel -> jit/vmap."""
    import jax
    import jax.numpy as jnp
    from dyno.dynspec.grammar import parser
    from dyno.dynspec.recipe import (
        DTCC_RECIPE,
        check_equation_group,
        extract_lhs_variable,
    )
    from dyno.dynspec.funcgen import build_definitions_block

    jax.config.update("jax_enable_x64", True)

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

    _, defs_block = build_definitions_block(
        def_trees, VARIABLES, def_spec, constants=params
    )

    eq_fns = [emit(eq, assign=False) for eq in eq_trees]
    def_fns = [
        (extract_lhs_variable(eq), emit(eq, assign=True)) for eq in defs_block.equations
    ]
    needed_shifts = [0, 1]

    def core(m, s, x, M, S, X, p):
        ctx = {"inf": np.inf}
        for i, pname in enumerate(params):
            ctx[pname] = p[i]
        for (group, shift), arr in zip(LAYOUT, (m, s, x, M, S, X)):
            for j, vname in enumerate(VARIABLES[group]):
                ctx[(vname, shift)] = arr[j]
        for shift in needed_shifts:
            loc = {}
            for key, val in ctx.items():
                if isinstance(key, tuple):
                    if key[1] == shift:
                        loc[(key[0], 0)] = val
                else:
                    loc[key] = val
            for vname, dfn in def_fns:
                res = dfn(loc)
                loc[(vname, 0)] = res
                ctx[(vname, shift)] = res
        return jnp.stack([efn(ctx) for efn in eq_fns], axis=-1)

    point = jax.jit(core)
    batch = jax.jit(jax.vmap(core, in_axes=[0] * 6 + [None]))

    sizes = [len(VARIABLES[g]) for g, _ in LAYOUT]
    cum = np.cumsum(sizes)

    def split_eval(v, p):
        parts, k = [], 0
        for sz in sizes:
            parts.append(v[k : k + sz])
            k += sz
        return jnp.reshape(core(*parts, p), (-1,))

    jacobian = jax.jit(jax.jacrev(split_eval, argnums=0))

    def f(*args, diff=False):
        jargs = [jnp.asarray(a, dtype=jnp.float64) for a in args]
        if any(a.ndim == 2 for a in jargs):
            if diff:
                raise NotImplementedError("diff=True is point-only here")
            N = max(a.shape[0] for a in jargs if a.ndim == 2)
            # p is NOT mapped by the vmapped kernel (in_axes=... + [None]):
            # pass it as-is instead of broadcasting it to (N, n_p).
            bargs = [
                jnp.broadcast_to(a, (N, a.shape[0])) if a.ndim == 1 else a
                for a in jargs[:-1]
            ] + [jargs[-1]]
            return np.asarray(batch(*bargs))
        if diff:
            v = jnp.concatenate(jargs[:-1])
            J = jacobian(v, jargs[-1])
            return [np.asarray(point(*jargs))] + list(
                np.split(np.asarray(J), cum[:-1], axis=1)
            )
        return np.asarray(point(*jargs))

    return f, (def_result, f_result, defs_block)


# ---------------------------------------------------------------------------
# 3. handcoded JAX reference
# ---------------------------------------------------------------------------


def hand_raw(m, s, x, M, S, X, p):
    """Hand-written arithmetic (same model).  Returns (N, 2) residuals."""
    import jax.numpy as jnp

    beta, sigma, eta, chi, delta, alpha = p
    z, k = s[..., 0], s[..., 1]
    n, i = x[..., 0], x[..., 1]
    Z, K = S[..., 0], S[..., 1]
    Np, Ip = X[..., 0], X[..., 1]
    y = jnp.exp(z) * k**alpha * n ** (1 - alpha)  # defs at t
    c = y - i
    rk = alpha * y / k
    w = (1 - alpha) * y / n
    Y = jnp.exp(Z) * K**alpha * Np ** (1 - alpha)  # defs at t+1
    C = Y - Ip
    RK = alpha * Y / K
    return jnp.stack(
        [
            chi * n**eta * c**sigma - w,
            1 - beta * (c / C) ** sigma * (1 - delta + RK),
        ],
        axis=-1,
    )


def build_hand_function():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    kernel = jax.jit(hand_raw)

    def f(*args, diff=False):
        if diff:
            raise NotImplementedError("handcoded reference is value-only")
        return np.asarray(kernel(*[jnp.asarray(a, dtype=jnp.float64) for a in args]))

    return f


# ---------------------------------------------------------------------------
# run it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    console.print(
        Rule(
            "[bold]one arbitrage function — three warm compilation paths[/bold]",
            style="blue",
        )
    )
    for e in EQUATIONS:
        console.print(f"  [dim]eq.[/dim] {e}")
    console.print(f"  [dim]definitions[/dim] = {DEFINITIONS}")
    console.print(f"  [dim]point[/dim] m={M0} s={S0} x={X0} p={P0}\n")

    # --- 1. dolang -----------------------------------------------------
    console.print(
        Rule(
            "[bold blue]1/3  dolang[/bold blue] → FlatFunctionFactory → numba guvectorize"
        )
    )
    f_dolang, f_dolang_1thr, factory, guf_src = build_dolang_function()
    console.print(
        "  factory arguments: "
        + ", ".join(f"{k}({len(v)})" for k, v in factory.arguments.items())
    )
    console.print(
        '  [dim]dolang compiles the gufunc with target="parallel" (prange over '
        "rows);\n"
        "  a single-threaded twin is built from the same source for the "
        "comparison.[/dim]"
    )
    console.print("  point:", f_dolang(M0, S0, X0, M0, S0, X0, P0), "\n")
    console.print(
        Panel(
            Syntax(guf_src, "python", line_numbers=True, word_wrap=True),
            title="generated gufunc source (dolang debug output)",
            border_style="grey50",
        )
    )

    # --- 2. dynspec + JAX, emitted ------------------------------------
    console.print(
        Rule(
            "[bold blue]2/3  dynspec + JAX[/bold blue] → recipe check → emitted expression → jit/vmap"
        )
    )
    f_jax, (def_res, f_res, defs_block) = build_emitted_function()
    console.print(
        f"  {str(def_res)}  |  {str(f_res)}  |  defs DAG: {defs_block.variable_names}"
    )
    console.print("  point:", f_jax(M0, S0, X0, M0, S0, X0, P0))
    console.print(
        "  [dim](each equation tree is compiled once at build time by emit();\n"
        "  the traced program is a plain nested arithmetic graph)[/dim]\n"
    )

    # --- 3. handcoded -------------------------------------------------
    console.print(
        Rule("[bold blue]3/3  handcoded JAX[/bold blue] → jax.jit of hand-written math")
    )
    f_hand = build_hand_function()
    console.print("  point:", f_hand(M0, S0, X0, M0, S0, X0, P0), "\n")
    console.print(
        Panel(
            Syntax(inspect.getsource(hand_raw), "python", line_numbers=True),
            title="source of the handcoded function",
            border_style="grey50",
        )
    )

    # --- agreement -----------------------------------------------------
    M, S, X = (
        np.tile(M0, (N_ROWS, 1)),
        np.tile(S0, (N_ROWS, 1)),
        np.tile(X0, (N_ROWS, 1)),
    )
    v_dolang = f_dolang(M, S, X, M, S, X, P0)
    v_jax = f_jax(M, S, X, M, S, X, P0)
    v_hand = f_hand(M, S, X, M, S, X, P0)
    d_dolang = f_dolang(M0, S0, X0, M0, S0, X0, P0, diff=True)
    d_jax = f_jax(M0, S0, X0, M0, S0, X0, P0, diff=True)

    console.print(Rule("[bold blue]agreement[/bold blue]"))
    console.print(f"  gufunc == emitted    : {np.allclose(v_dolang, v_jax)}")
    console.print(f"  gufunc == handcoded  : {np.allclose(v_dolang, v_hand)}")
    console.print(f"  emitted == handcoded : {np.allclose(v_jax, v_hand)}")
    maxd = max(np.abs(a - b).max() for a, b in zip(d_dolang[1:], d_jax[1:]))
    console.print(
        f"  max |jacobian diff| (emitted analytic vs gufunc finite-diff): {maxd:.2e}"
    )

    # --- timings -------------------------------------------------------
    console.print()
    table = Table(title="timing — warm calls, µs (best of 3)")
    table.add_column("call", justify="left", style="cyan")
    table.add_column("gufunc 8-thread", justify="right", style="green")
    table.add_column("gufunc 1-thread", justify="right", style="yellow")
    table.add_column("emitted jit", justify="right", style="magenta")
    table.add_column("handcoded jit", justify="right", style="magenta")
    table.add_column("emit/par", justify="right")

    pt = (
        lambda: f_dolang(M0, S0, X0, M0, S0, X0, P0),
        lambda: f_dolang_1thr(M0, S0, X0, M0, S0, X0, P0),
        lambda: f_jax(M0, S0, X0, M0, S0, X0, P0),
        lambda: f_hand(M0, S0, X0, M0, S0, X0, P0),
    )
    n_vec = 10_000
    Mv, Sv, Xv = (
        np.tile(M0, (n_vec, 1)),
        np.tile(S0, (n_vec, 1)),
        np.tile(X0, (n_vec, 1)),
    )
    vec = (
        lambda: f_dolang(Mv, Sv, Xv, Mv, Sv, Xv, P0),
        lambda: f_dolang_1thr(Mv, Sv, Xv, Mv, Sv, Xv, P0),
        lambda: f_jax(Mv, Sv, Xv, Mv, Sv, Xv, P0),
        lambda: f_hand(Mv, Sv, Xv, Mv, Sv, Xv, P0),
    )
    diff = (
        lambda: f_dolang(M0, S0, X0, M0, S0, X0, P0, diff=True),
        lambda: f_jax(M0, S0, X0, M0, S0, X0, P0, diff=True),
    )

    t0, t1, t2, t3 = [time_it(c, 300) for c in pt]
    table.add_row(
        "[bold]point[/bold] (300 loops)",
        f"{t0:.1f}",
        f"{t1:.1f}",
        f"{t2:.1f}",
        f"{t3:.1f}",
        f"{t2 / t0:.1f}×",
    )
    t0, t1, t2, t3 = [time_it(c, 30) for c in vec]
    table.add_row(
        f"[bold]vectorized {n_vec} rows[/bold] (30 loops)",
        f"{t0:.1f}",
        f"{t1:.1f}",
        f"{t2:.1f}",
        f"{t3:.1f}",
        f"{t2 / t0:.1f}×",
    )
    t0, t1 = [time_it(c, 30) for c in diff]
    table.add_row(
        "[bold]diff=True[/bold] (30 loops)",
        f"{t0:.1f}",
        "n/a",
        f"{t1:.1f}",
        "n/a",
        f"{t1 / t0:.1f}×",
    )
    console.print(table)
    console.print(
        "[dim]point-call gap for jax is dominated by device_put of small "
        "input arrays (~60µs each); vectorized calls amortize it.  The\n"
        "gufunc 1-thread column is the same generated numba source compiled "
        'without target="parallel" — on one thread jax is the faster\n'
        "kernel (see the scaling table).[/dim]"
    )

    # --- scaling --------------------------------------------------------
    console.print()
    stab = Table(
        title="large-batch scaling — one warm call per size, f64 random rows, sync-forced"
    )
    stab.add_column("N", justify="right", style="cyan")
    stab.add_column("gufunc 8-thr", justify="right", style="green")
    stab.add_column("gufunc 1-thr", justify="right", style="yellow")
    stab.add_column("emitted jit", justify="right", style="magenta")
    stab.add_column("handcoded jit", justify="right", style="magenta")
    stab.add_column("emit Mpts/s", justify="right")
    stab.add_column("equal", justify="right")
    rng = np.random.default_rng(0)
    for n in (10_000, 100_000, 1_000_000, 5_000_000):
        Mv = rng.normal(size=(n, 1))
        Sv = rng.normal(size=(n, 2))
        Xv = np.abs(rng.normal(size=(n, 2))) + 0.1
        args = (Mv, Sv, Xv, Mv, Sv, Xv, P0)
        v_dolang = f_dolang(*args)
        f_dolang_1thr(*args)
        w = f_jax(*args)
        w.block_until_ready() if hasattr(w, "block_until_ready") else None
        w = f_hand(*args)
        w.block_until_ready() if hasattr(w, "block_until_ready") else None
        t0 = time.perf_counter()
        f_dolang(*args)
        t_dol = time.perf_counter() - t0
        t0 = time.perf_counter()
        f_dolang_1thr(*args)
        t_dol1 = time.perf_counter() - t0
        t0 = time.perf_counter()
        v_jax = f_jax(*args)
        v_jax.block_until_ready() if hasattr(v_jax, "block_until_ready") else None
        t_jax = time.perf_counter() - t0
        t0 = time.perf_counter()
        v_hand = f_hand(*args)
        v_hand.block_until_ready() if hasattr(v_hand, "block_until_ready") else None
        t_h = time.perf_counter() - t0
        stab.add_row(
            f"{n:,}",
            f"{t_dol * 1e3:.1f} ms",
            f"{t_dol1 * 1e3:.1f} ms",
            f"{t_jax * 1e3:.1f} ms",
            f"{t_h * 1e3:.1f} ms",
            f"{n / t_jax / 1e6:.1f}",
            f"{np.allclose(v_dolang, v_jax, equal_nan=True)}",
        )
    console.print(stab)
    console.print(
        "[dim]on a single thread the jit kernel is ~2-3× faster than numba; "
        'the gufunc\'s 7-8× thread scaling (target="parallel") is what\n'
        "leaves jax behind at 8 cores — xla's fusion-level parallelism "
        "reaches ~2.8× on this CPU.[/dim]"
    )
