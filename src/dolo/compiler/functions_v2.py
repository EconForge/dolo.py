"""
Functions backend for :class:`dolo.compiler.model_v2.ModelV2`.

Equation trees are produced (and conformity-checked) by ``dyno.dynspec``;
this module compiles them into JAX-backed callables.

Two compilation stages exist in the migration history:

1. *interpreted* (first prototype): ``EquationsEvaluator`` walking each
   lark tree at trace time inside ``jax.jit``/``jax.vmap`` — the runtime
   was compiled, but the traced program carried dict-staging/gather ops
   that hurt fusion (~1.4x vs. hand-written jax on large batches);
2. *emitted* (current): each equation tree is compiled once at build
   time into a closure that assembles the arithmetic expression
   directly (see ``_emit``), so the traced jaxpr is a plain nested
   arithmetic graph, like hand-written jnp code.

The callables reproduce the legacy dolo function API exactly, so that
``dolo.algos`` does not need to change:

    g = model.functions["transition"]      g(m[-1], s[-1], x[-1], m[t], p)
    f = model.functions["arbitrage"]       f(m[t], s[t], x[t], m[t+1], s[t+1], x[t+1], p)
    a = model.functions["auxiliary"]       a(m[t], s[t], x[t], p)
    lb = model.functions["arbitrage_lb"]   lb(m[t], s[t], p)
    ub = model.functions["arbitrage_ub"]   ub(m[t], s[t], p)

where ``p`` (parameters) is always the last positional argument, point
arguments are 1-d arrays producing 1-d results, first-axis-vectorized
arguments produce row-vectorized results, and ``diff=True`` returns
``(value, jacobian-group1, jacobian-group2, ...)`` mimicking the legacy
source-differentiated functions.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from dyno.dynspec.analyze import EquationsEvaluator
from dyno.dynspec.funcgen import DefinitionsBlock, build_definitions_block
from dyno.dynspec.recipe import (
    DTCC_RECIPE,
    EquationGroupSpec,
    VariableSpec,
    check_equation_group,
    extract_lhs_variable,
    extract_variables_from_equation,
)

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# JAX function table: overrides the autodiff (DNumber) implementations that
# dyno.dynspec installs by default in `EquationsEvaluator`.  jnp arrays are
# not np.ndarrays, so the autodiff fallbacks ("use math.* for non-arrays")
# would break tracing if they were left in place.
# ---------------------------------------------------------------------------

JNP_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "exp": jnp.exp,
    "log": jnp.log,
    "sqrt": jnp.sqrt,
    "abs": jnp.abs,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tan": jnp.tan,
    "sinh": jnp.sinh,
    "cosh": jnp.cosh,
    "tanh": jnp.tanh,
    "asin": jnp.arcsin,
    "acos": jnp.arccos,
    "atan": jnp.arctan,
    "max": jnp.maximum,
    "min": jnp.minimum,
    "log10": jnp.log10,
    "log2": jnp.log2,
    "floor": jnp.floor,
    "ceil": jnp.ceil,
    "pow": jnp.pow,
}

# Named constants made available inside equation expressions.
SPECIAL_CONSTANTS = {"inf": np.inf}

# numpy mirror of JNP_FUNCTIONS: the emitted expression closures are
# independent of the backend, so the same trees compile twice — to jnp
# (jit/vmap/diff) and to plain numpy (the low-latency point-call path).
NP_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "max": np.maximum,
    "min": np.minimum,
    "log10": np.log10,
    "log2": np.log2,
    "floor": np.floor,
    "ceil": np.ceil,
    "pow": np.power,
}


class GroupLayout:
    """How a legacy function maps positional arguments to (group, shift)."""

    def __init__(
        self,
        recipe_group: str,
        layout: List[Tuple[str, int]],
        assign: bool,
        with_definitions: bool = False,
        output_group: str | None = None,
    ) -> None:
        self.recipe_group = recipe_group
        self.layout = layout
        self.assign = assign
        self.with_definitions = with_definitions
        self.output_group = output_group


def _group_spec(
    name: str,
    allowed: list[tuple[str, int]],
    target: str | None,
    recursive: bool,
) -> EquationGroupSpec:
    """Build a dolo-style group spec (translating `dolo/compiler/recipes.yaml`).

    The target variable group is added to the allowed shifts at 0 (its
    left-hand-side occurrence is collected as a variable too), and
    auxiliaries are allowed at shifts 0/1 like the legacy importer did.
    """
    varspecs = [VariableSpec(g, s) for g, s in allowed]
    if target is not None:
        varspecs.append(VariableSpec(target, 0))
    varspecs.extend([VariableSpec("auxiliaries", 0), VariableSpec("auxiliaries", 1)])
    varspecs.append(VariableSpec("parameters", 0))
    return EquationGroupSpec(
        name=name,
        allowed=varspecs,
        target=target,
        recursive=recursive,
        n_equations=None,
        optional=True,
    )


GROUP_LAYOUTS: Dict[str, GroupLayout] = {
    "transition": GroupLayout(
        recipe_group="transition",
        layout=[("exogenous", -1), ("states", -1), ("controls", -1), ("exogenous", 0)],
        assign=True,
        output_group="states",
    ),
    "arbitrage": GroupLayout(
        recipe_group="arbitrage",
        layout=[
            ("exogenous", 0),
            ("states", 0),
            ("controls", 0),
            ("exogenous", 1),
            ("states", 1),
            ("controls", 1),
        ],
        assign=False,
        with_definitions=True,
    ),
    "auxiliary": GroupLayout(
        recipe_group="definitions",
        layout=[("exogenous", 0), ("states", 0), ("controls", 0)],
        assign=True,
        output_group="auxiliaries",
    ),
    "arbitrage_lb": GroupLayout(
        recipe_group="arbitrage_bounds",
        layout=[("exogenous", 0), ("states", 0)],
        assign=False,
    ),
    "arbitrage_ub": GroupLayout(
        recipe_group="arbitrage_bounds",
        layout=[("exogenous", 0), ("states", 0)],
        assign=False,
    ),
    # standalone bounds groups (docs use `controls_lb`/`controls_ub`,
    # e.g. `controls_lb: {i = 0, n = 0}`): dyno's recipe treats each line
    # as `x[t] = <bound>` defining one control, so they compile with
    # assignment semantics; same (m, s, p) signature as arbitrage bounds
    "controls_lb": GroupLayout(
        recipe_group="controls_lb",
        layout=[("exogenous", 0), ("states", 0)],
        assign=True,
        output_group="controls",
    ),
    "controls_ub": GroupLayout(
        recipe_group="controls_ub",
        layout=[("exogenous", 0), ("states", 0)],
        assign=True,
        output_group="controls",
    ),
    # --- additional dtcc groups (dolo/compiler/recipes.yaml) ---------
    "value": GroupLayout(
        recipe_group="value",
        layout=[
            ("exogenous", 0),
            ("states", 0),
            ("controls", 0),
            ("values", 0),
            ("exogenous", 1),
            ("states", 1),
            ("controls", 1),
            ("values", 1),
        ],
        assign=True,
        with_definitions=True,
        output_group="values",
    ),
    "felicity": GroupLayout(
        recipe_group="felicity",
        layout=[("exogenous", 0), ("states", 0), ("controls", 0)],
        assign=True,
        with_definitions=True,
        output_group="rewards",
    ),
    "expectation": GroupLayout(
        recipe_group="expectation",
        layout=[("exogenous", 1), ("states", 1), ("controls", 1)],
        assign=True,
        with_definitions=True,
        output_group="expectations",
    ),
    "half_transition": GroupLayout(
        recipe_group="half_transition",
        layout=[("exogenous", -1), ("poststates", -1), ("exogenous", 0)],
        assign=True,
        with_definitions=True,
        output_group="states",
    ),
    "direct_response": GroupLayout(
        recipe_group="direct_response",
        layout=[("exogenous", 0), ("states", 0), ("expectations", 0)],
        assign=True,
        with_definitions=True,
        output_group="controls",
    ),
    "direct_response_egm": GroupLayout(
        recipe_group="direct_response_egm",
        layout=[("exogenous", 0), ("poststates", 0), ("expectations", 0)],
        assign=True,
        with_definitions=True,
        output_group="controls",
    ),
    "reverse_state": GroupLayout(
        recipe_group="reverse_state",
        layout=[("exogenous", 0), ("poststates", 0), ("controls", 0)],
        assign=True,
        with_definitions=True,
        output_group="states",
    ),
    "arbitrage_exp": GroupLayout(
        recipe_group="arbitrage_exp",
        layout=[("states", 0), ("controls", 0)],
        assign=False,
        with_definitions=True,
    ),
}

_EXTRA_GROUP_SPECS: Dict[str, EquationGroupSpec] = {
    "value": _group_spec(
        "value",
        [
            ("exogenous", 0),
            ("states", 0),
            ("controls", 0),
            ("values", 0),
            ("exogenous", 1),
            ("states", 1),
            ("controls", 1),
            ("values", 1),
        ],
        target="values",
        recursive=False,
    ),
    "felicity": _group_spec(
        "felicity",
        [("exogenous", 0), ("states", 0), ("controls", 0)],
        target="rewards",
        recursive=True,
    ),
    "expectation": _group_spec(
        "expectation",
        [("exogenous", 1), ("states", 1), ("controls", 1)],
        target="expectations",
        recursive=False,
    ),
    "half_transition": _group_spec(
        "half_transition",
        [("exogenous", -1), ("poststates", -1), ("exogenous", 0)],
        target="states",
        recursive=False,
    ),
    "direct_response": _group_spec(
        "direct_response",
        [("exogenous", 0), ("states", 0), ("expectations", 0)],
        target="controls",
        recursive=True,
    ),
    "direct_response_egm": _group_spec(
        "direct_response_egm",
        [("exogenous", 0), ("poststates", 0), ("expectations", 0)],
        target="controls",
        recursive=True,
    ),
    "reverse_state": _group_spec(
        "reverse_state",
        [("exogenous", 0), ("poststates", 0), ("controls", 0)],
        target="states",
        recursive=True,
    ),
    "arbitrage_exp": _group_spec(
        "arbitrage_exp",
        [("states", 0), ("controls", 0)],
        target=None,
        recursive=False,
    ),
}

_SPECS: Dict[str, EquationGroupSpec] = {
    name: DTCC_RECIPE.get_group_spec(name)
    for name in (
        "definitions",
        "transition",
        "arbitrage",
        "arbitrage_bounds",
        "controls_lb",
        "controls_ub",
    )
}
_SPECS.update(_EXTRA_GROUP_SPECS)


def _new_evaluator(context: Dict[str, Any], assign: bool) -> EquationsEvaluator:
    ev = EquationsEvaluator(context=context, function_table={}, assign=assign)
    ev.function_table.update(JNP_FUNCTIONS)
    return ev


# ---------------------------------------------------------------------------
# expression emission: compile each dynspec tree into a closure that builds
# the jax expression directly (no dict staging / gathers in the traced
# program).  All lark walking happens here, at build time.
# ---------------------------------------------------------------------------


def _emit(
    tree, assign: bool = False, functions: Dict[str, Callable] = JNP_FUNCTIONS
) -> Callable:
    """Compile one equation tree into ``fn(ctx) -> value``.

    ``ctx`` maps variable ``(name, shift)`` keys to arrays, and plain
    ``name`` keys to constants (parameters, ``inf``).  The returned
    closure contains no python book-keeping: evaluating it produces a
    plain nested arithmetic expression.  ``functions`` selects the math
    backend (jnp for tracing, numpy for the point-call path).
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
    if data == "value":
        name = str(c[0].children[0])
        t = int(str(c[1].children[0]))
        return lambda ctx: ctx.get(("value", name, t), ctx.get(name, np.nan))
    if data in ("add", "sub", "mul", "div"):
        a, b = _emit(c[0], functions=functions), _emit(c[1], functions=functions)
        if data == "add":
            return lambda ctx: a(ctx) + b(ctx)
        if data == "sub":
            return lambda ctx: a(ctx) - b(ctx)
        if data == "mul":
            return lambda ctx: a(ctx) * b(ctx)
        return lambda ctx: a(ctx) / b(ctx)
    if data == "neg":
        a = _emit(c[0], functions=functions)
        return lambda ctx: -a(ctx)
    if data == "pow":
        a, b = _emit(c[0], functions=functions), _emit(c[1], functions=functions)
        return lambda ctx: a(ctx) ** b(ctx)
    if data == "call":
        fname = str(c[0].children[0])
        args = [_emit(arg, functions=functions) for arg in c[1:]]
        fn = functions[fname]
        return lambda ctx: (
            fn(*[a(ctx) for a in args]) if len(args) > 1 else fn(args[0](ctx))
        )
    if data == "equality":
        if assign:
            return _emit(c[1], functions=functions)
        lhs, rhs = _emit(c[0], functions=functions), _emit(c[1], functions=functions)
        return lambda ctx: rhs(ctx) - lhs(ctx)
    if data == "complementarity":
        # `res ⟂ lb <= x <= ub`: the residual is the equation, bounds are
        # compiled as separate `*_lb` / `*_ub` groups (dynospec convention)
        return _emit(c[0], assign=assign, functions=functions)
    if data == "bare_formula":
        return _emit(c[0], functions=functions)
    raise NotImplementedError(f"Cannot compile dynspec node '{data}'.")


def _eval_definitions_at_shift(
    def_equations: Sequence[Any],
    variables: Dict[str, Dict[int, Any]],
    constants: Dict[str, Any],
    shift: int,
) -> None:
    """[not used anymore — expression emission replaced the interpreter;
    kept for reference until the migration is complete.]

    Evaluate definition equations with all variables at date ``shift``.
    """
    for eq in def_equations:
        name = extract_lhs_variable(eq)
        if name is None:
            continue
        staged: Dict[str, Dict[int, Any]] = {}
        for vname, shifts in variables.items():
            if shift in shifts:
                staged[vname] = {0: shifts[shift]}
            elif 0 in shifts:
                staged[vname] = {0: shifts[0]}
        sub_context = {
            "constants": dict(constants),
            "variables": staged,
            "values": {},
            "processes": {},
            "steady_states": {},
        }
        ev = _new_evaluator(sub_context, assign=True)
        ev.visit(eq)
        variables.setdefault(name, {})[shift] = staged[name][0]


def _with_arity(n_args: int, core: Callable) -> Callable:
    """Bind ``core(*args)`` to an explicit signature of ``n_args`` args.

    ``jax.vmap`` inspects the wrapped function's signature (for error
    reporting) and breaks on ``*args``-only signatures.
    """
    if n_args == 3:

        def f(a0, a1, a2):
            return core(a0, a1, a2)

    elif n_args == 4:

        def f(a0, a1, a2, a3):
            return core(a0, a1, a2, a3)

    elif n_args == 5:

        def f(a0, a1, a2, a3, a4):
            return core(a0, a1, a2, a3, a4)

    elif n_args == 7:

        def f(a0, a1, a2, a3, a4, a5, a6):
            return core(a0, a1, a2, a3, a4, a5, a6)

    elif n_args == 9:

        def f(a0, a1, a2, a3, a4, a5, a6, a7, a8):
            return core(a0, a1, a2, a3, a4, a5, a6, a7, a8)

    else:
        raise NotImplementedError(f"No explicit signature for {n_args} args.")
    return f


def make_function(
    *,
    name: str,
    layout: GroupLayout,
    variables: Dict[str, List[str]],
    parameters: List[str],
    equations: Sequence[Any],
    definitions_block: DefinitionsBlock | None = None,
) -> Callable:
    """Build a legacy-style dolo function from dynspec equation trees."""

    arg_groups = layout.layout
    n_variable_args = len(arg_groups)

    # --- emit equations at build time (all lark walking happens here) --
    def_names = set(definitions_block.variable_names) if definitions_block else set()
    needed_shifts: List[int] = []
    if layout.with_definitions and definitions_block is not None:
        shifts: set[int] = set()
        for eq in equations:
            for vname, sset in extract_variables_from_equation(eq).items():
                if vname in def_names:
                    shifts.update(sset)
        needed_shifts = sorted(shifts)

    if layout.assign:
        lhs_names = [extract_lhs_variable(eq) for eq in equations]
        if any(l is None for l in lhs_names):
            raise ValueError(
                f"{name}: assignment semantics need a variable lhs per equation"
            )
        eq_fns_j = [_emit(eq, assign=True) for eq in equations]
        eq_fns_n = [_emit(eq, assign=True, functions=NP_FUNCTIONS) for eq in equations]
    else:
        lhs_names = []
        eq_fns_j = [_emit(eq, assign=False) for eq in equations]
        eq_fns_n = [_emit(eq, assign=False, functions=NP_FUNCTIONS) for eq in equations]

    def_fns_j = None
    def_fns_n = None
    if definitions_block is not None and (
        layout.with_definitions or name == "auxiliary"
    ):
        # same trees, both backends
        def_fns_j = [
            (extract_lhs_variable(eq), _emit(eq, assign=True))
            for eq in definitions_block.equations
        ]
        def_fns_n = [
            (extract_lhs_variable(eq), _emit(eq, assign=True, functions=NP_FUNCTIONS))
            for eq in definitions_block.equations
        ]

    def _make_core(xm, eq_fns, def_fns):
        """Build the backend-specific core: identical bookkeeping, `xm` is
        either jnp (traced) or np (plain arrays)."""

        def core(*jargs):
            params = jargs[-1]
            # ctx: plain-name keys -> constants/parameters; (name, shift) -> vars
            ctx: dict = dict(SPECIAL_CONSTANTS)
            # 1-d parameters (point calls, or `p` unmapped under vmap) give
            # scalars; 2-d parameters (numpy-batched calls) give columns.
            if params.ndim == 1:
                pvals = [params[i] for i in range(params.shape[0])]
            else:
                pvals = [params[:, i] for i in range(params.shape[1])]
            for i, pname in enumerate(parameters):
                ctx[pname] = pvals[i]
            for (group, shift), arr in zip(arg_groups, jargs[:-1]):
                # under vmap each group arrives as a 1-d row (`arr[j]` is a
                # scalar); the numpy-batched core feeds (N, n_g) arrays, so
                # transpose first to index columns.
                arr_t = arr.T if arr.ndim == 2 else arr
                for j, vname in enumerate(variables[group]):
                    ctx[(vname, shift)] = arr_t[j]

            if needed_shifts and def_fns is not None:
                for shift in needed_shifts:
                    loc: dict = {}
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

            if layout.assign:
                for vname, efn in zip(lhs_names, eq_fns):
                    ctx[(vname, 0)] = efn(ctx)
                if layout.output_group is not None:
                    names = variables[layout.output_group]
                    if names:
                        return xm.stack([ctx[(vname, 0)] for vname in names])
                    return xm.zeros(0)
                return xm.zeros(0)

            if eq_fns:
                return xm.stack([efn(ctx) for efn in eq_fns])
            return xm.zeros(0)

        return _with_arity(n_variable_args + 1, core)

    # Emission happens at build time; the traced core is a plain nested
    # arithmetic expression.  The jax point path reuses the vmapped
    # kernel (batch of 1) because vmap execution is efficient there and
    # `p` stays unmapped.
    point_fn = _make_core(jnp, eq_fns_j, def_fns_j)
    n_total = n_variable_args + 1
    batch = jax.jit(jax.vmap(point_fn, in_axes=[0] * n_total))
    batch_pt = jax.jit(jax.vmap(point_fn, in_axes=[0] * n_variable_args + [None]))

    def _point_impl(*a):
        expanded = [jnp.expand_dims(x, 0) for x in a[:-1]]
        return batch_pt(*expanded, a[-1])[0]

    point = jax.jit(_with_arity(n_total, _point_impl))

    # Point calls on plain numpy arrays go through the numpy closures —
    # same emitted expressions, no device transfers, no jit dispatch
    # (DEV_NOTES §5.3 / §6.5: ~470µs -> ~µs for scipy-style algos).
    point_np = _make_core(np, eq_fns_n, def_fns_n)

    # --- `diff=True`: one jacrev of the concatenated variable args, cached
    # by jit (recomputing it per argument group retraces six functions per
    # call).  Like `point`, it goes through the vmap graph.
    sizes = [len(variables[g]) for g, _ in arg_groups]

    def _split_and_eval_b(v, vp):
        parts = []
        k = 0
        for sz in sizes:
            parts.append(v[None, k : k + sz])
            k += sz
        return batch_pt(*parts, vp)[0]

    _cum = np.cumsum(sizes)

    if sizes:
        _jacobian = jax.jit(jax.jacrev(_split_and_eval_b, argnums=0))

        def _diff_all(jargs):
            vp = jargs[-1]
            v = jnp.concatenate(jargs[:-1])
            J = _jacobian(v, vp)
            return [np.asarray(p) for p in jnp.split(J, _cum[:-1], axis=1)]

    else:

        def _diff_all(jargs):
            return []

    def _normalize_batched(args):
        """Validate/broadcast numpy row-vectorized arguments."""
        N = max(a.shape[0] for a in args if a.ndim == 2)
        bargs = []
        for a in args:
            if a.ndim == 1:
                bargs.append(np.broadcast_to(a, (N, a.shape[0])))
            elif a.ndim == 2:
                if a.shape[0] != N:
                    raise ValueError("Inconsistent number of points across arguments.")
                bargs.append(a)
            else:
                bargs.append(np.broadcast_to(a, (N,)))
        return bargs

    def _np_batched(bargs):
        # numpy closures produce (n_out, N); return (N, n_out).
        # Row-independent equations (e.g. literal bounds `0.0`/`-inf`)
        # yield scalars, hence 1-d stacks: repeat them over N.
        N = max((a.shape[0] for a in bargs if a.ndim == 2), default=1)
        res = np.asarray(point_np(*bargs)).T
        if res.ndim == 1:
            res = np.repeat(res[None, :], N, axis=0)
        return res

    def _np_vec_diff(args):
        """Vectorized diff=True on numpy data — mirrors the legacy
        `standard_function.__vecdiff__`: finite differences (eps=1e-8)
        w.r.t. every argument but the last (parameters)."""
        bargs = _normalize_batched(args)
        out = _np_batched(bargs)
        N, n_out = out.shape
        l_dout = []
        for i in range(len(bargs) - 1):
            a = bargs[i]
            dout = np.zeros((N, n_out, a.shape[1]))
            for j in range(a.shape[1]):
                xx = list(bargs)
                xx[i] = a.copy()
                xx[i][:, j] += 1e-8
                dout[:, :, j] = (_np_batched(xx) - out) / 1e-8
            l_dout.append(dout)
        return [out] + l_dout

    def fn(*args, diff=False):
        # numpy-only calls: plain arrays avoid all jax marshaling (no
        # device transfers, no jit dispatch — DEV_NOTES §5.3 / §6.5).
        # Covered: point (1-d) and row-vectorized (2-d) inputs.
        if not diff and all(isinstance(a, np.ndarray) for a in args):
            ndims = {a.ndim for a in args}
            if any(d > 2 for d in ndims):
                raise ValueError(
                    "dolo functions accept point (1-d) or row-vectorized (2-d) "
                    "arguments only."
                )
            if 2 not in ndims:
                return np.asarray(point_np(*args))
            return _np_batched(_normalize_batched(list(args)))

        # vectorized diff=True on numpy data: finite differences, legacy
        # `numdiff` semantics (jacobians for every argument but p).
        if (
            diff
            and all(isinstance(a, np.ndarray) for a in args)
            and any(a.ndim == 2 for a in args)
        ):
            if any(a.ndim > 2 for a in args):
                raise ValueError(
                    "dolo functions accept point (1-d) or row-vectorized (2-d) "
                    "arguments only."
                )
            return _np_vec_diff(list(args))

        # Skip the `jnp.asarray(..., dtype=float64)` marshaling for device
        # arrays that already satisfy the dtype — algos that hold jax
        # arrays across calls avoid the extra copy/pass cost (DEV_NOTES §6.4).
        jargs = []
        for a in args:
            if isinstance(a, jnp.ndarray) and a.dtype == jnp.float64:
                jargs.append(a)
            else:
                jargs.append(jnp.asarray(a, dtype=jnp.float64))
        ndims = {a.ndim for a in jargs}
        if any(d > 2 for d in ndims):
            raise ValueError(
                "dolo functions accept point (1-d) or row-vectorized (2-d) "
                "arguments only."
            )
        batched = 2 in ndims
        if batched:
            if diff:
                raise NotImplementedError(
                    "diff=True is not implemented for vectorized arguments."
                )
            N = max(a.shape[0] for a in jargs if a.ndim == 2)
            bjargs = []
            for a in jargs:
                if a.ndim == 1:
                    bjargs.append(jnp.broadcast_to(a, (N, a.shape[0])))
                elif a.ndim == 2:
                    if a.shape[0] != N:
                        raise ValueError(
                            "Inconsistent number of points across arguments."
                        )
                    bjargs.append(a)
                else:
                    bjargs.append(jnp.broadcast_to(a, (N,)))
            return np.asarray(batch(*bjargs))
        if diff:
            return [np.asarray(point(*jargs)), *_diff_all(jargs)]
        return np.asarray(point(*jargs))

    fn.__name__ = name
    fn.__doc__ = (
        f"Evaluate the {name} equation group of a dolo model "
        f"(JAX backend, interpreted evaluation). Args: "
        f"{[f'{g}[t{s:+d}]' if s else f'{g}[t]' for g, s in arg_groups]} + p."
    )
    return fn


def compile_group(
    *,
    name: str,
    equations: Sequence[Any],
    variables: Dict[str, List[str]],
    parameters: List[str],
    definitions_block: DefinitionsBlock | None = None,
) -> Callable:
    """Conformity-check a group and compile it into a callable."""
    layout = GROUP_LAYOUTS[name]
    spec = _SPECS[layout.recipe_group]
    group_equations = list(equations)

    result = check_equation_group(
        group_equations, variables, spec, constants=parameters
    )
    if not result.ok:
        raise ValueError(f"Equations of group '{name}' do not conform:\n{result}")

    # for recursive groups use the DAG order from the conformity result
    if result.dag_order is not None and layout.output_group is not None:
        order_map: Dict[str, int] = {}
        for i, eq in enumerate(group_equations):
            lhs = extract_lhs_variable(eq)
            if lhs is not None:
                order_map[lhs] = i
        group_equations = [group_equations[order_map[v]] for v in result.dag_order]

    return make_function(
        name=name,
        layout=layout,
        variables=variables,
        parameters=parameters,
        equations=group_equations,
        definitions_block=definitions_block,
    )


def build_definitions(
    *,
    equations: Sequence[Any],
    variables: Dict[str, List[str]],
    parameters: List[str],
) -> Tuple[Any, DefinitionsBlock | None]:
    """Check and order the auxiliary-variable definitions block."""
    spec = _SPECS["definitions"]
    result = check_equation_group(equations, variables, spec, constants=parameters)
    if not result.ok:
        raise ValueError(f"Definition equations do not conform:\n{result}")
    _, block = build_definitions_block(equations, variables, spec, constants=parameters)
    return result, block
