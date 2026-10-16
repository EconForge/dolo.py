"""
Functions backend for :class:`dolo.compiler.model_v2.ModelV2`.

Equation trees are produced (and conformity-checked) by ``dyno.dynspec``;
this module compiles them into callables using JAX as described in the
migration plan (design decision D2: *interpreted* evaluation first,
``jax.jit``/``jax.vmap`` around ``dyno.dynspec.analyze.EquationsEvaluator``
with a JAX function table — generating dedicated source code can come
later, behind the same interface).

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


# Bounds: residual-free expressions over (m, s) and parameters, one per
# control variable, in equation order.
_BOUND_SPEC = EquationGroupSpec(
    name="arbitrage_bounds",
    allowed=[
        VariableSpec("exogenous", 0),
        VariableSpec("states", 0),
        VariableSpec("parameters", 0),
    ],
    target=None,
    recursive=False,
    n_equations="controls",
)


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
    for name in ("definitions", "transition", "arbitrage")
}
_SPECS["arbitrage_bounds"] = _BOUND_SPEC
_SPECS.update(_EXTRA_GROUP_SPECS)


def _new_evaluator(context: Dict[str, Any], assign: bool) -> EquationsEvaluator:
    ev = EquationsEvaluator(context=context, function_table={}, assign=assign)
    ev.function_table.update(JNP_FUNCTIONS)
    return ev


def _eval_definitions_at_shift(
    def_equations: Sequence[Any],
    variables: Dict[str, Dict[int, Any]],
    constants: Dict[str, Any],
    shift: int,
) -> None:
    """Evaluate definition equations with all variables at date ``shift``.

    Definitions are written at date 0 (``aux[t] = f(x[t], ...)``); to obtain
    ``aux[t+shift]`` we stage every known variable at its ``shift`` value
    into the date-0 slot, run the block topologically, and store the results
    back at ``shift``.  Same semantics as
    ``dyno.dynspec.funcgen._evaluate_definitions_at_shift`` but JAX-native.
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

    def _core(*jargs):
        params = jargs[-1]
        ctx_vars: Dict[str, Dict[int, Any]] = {}
        for (group, shift), arr in zip(arg_groups, jargs[:-1]):
            for j, vname in enumerate(variables[group]):
                ctx_vars.setdefault(vname, {})[shift] = arr[j]
        constants = dict(SPECIAL_CONSTANTS)
        constants.update((p, params[i]) for i, p in enumerate(parameters))
        context = {
            "constants": constants,
            "variables": ctx_vars,
            "values": {},
            "processes": {},
            "steady_states": {},
        }

        if layout.with_definitions and definitions_block is not None:
            shifts: set[int] = set()
            for eq in equations:
                for vname, sset in extract_variables_from_equation(eq).items():
                    if vname in definitions_block.variable_names:
                        shifts.update(sset)
            for shift in sorted(shifts):
                _eval_definitions_at_shift(
                    definitions_block.equations, ctx_vars, constants, shift
                )

        ev = _new_evaluator(context, assign=layout.assign)
        results = [ev.visit(eq) for eq in equations]

        if layout.output_group is not None:
            names = variables[layout.output_group]
            if names:
                return jnp.stack([ctx_vars[vname][0] for vname in names])
            return jnp.zeros(0)
        if results:
            return jnp.stack(results)
        return jnp.zeros(0)

    point_fn = _with_arity(n_variable_args + 1, _core)
    point = jax.jit(point_fn)
    batch = jax.jit(jax.vmap(point_fn))

    def fn(*args, diff=False):
        jargs = [jnp.asarray(a, dtype=jnp.float64) for a in args]
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
            values = [np.asarray(point(*jargs))]
            for i in range(n_variable_args):

                def _jac(u, i=i):
                    args2 = list(jargs)
                    args2[i] = u
                    return point(*args2)

                values.append(np.asarray(jax.jacrev(_jac)(jargs[i])))
            return values
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
