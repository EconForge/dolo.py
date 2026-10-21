"""
Grid kernels of the Euler residuals F, and of their derivatives dF_1 and dF_2, for all exogenous
regimes at once (`CompiledResidual`).

For controls x (n_m, N, n_x) on the grid and coefficients C of the decision rule (one spline per
regime), with weights w[i, k] and nodes M[i, k] of the exogenous process (node k is regime r(k)):

    S[i, k, n] = g(m_i, s_n, x[i, n], M[i, k])
    X[i, k, n] = phi_{r(k)}(S[i, k, n])
    F[i, n]    = sum_k w[i, k] f(m_i, s_n, x[i, n], M[i, k], S[i, k, n], X[i, k, n])

Each of S, X and the arbitrage part is a separate jit: in a single jit, XLA fuses the spline
evaluation (gathers) into the consumers and recomputes it, which is several times slower.

The derivatives are forward-mode, with two savings:
- the decision rule is linearized at S, as in Dolo.jl (`Linearized`): its value and gradient
  w.r.t. S come from the spline evaluation, so autodiff never differentiates the gathers;
- Jacobians are sparse (on integration_B, f depends on 5 of the 8 next-period controls, and
  df/dX has 12 nonzero entries out of 64): columns that never share a row are differentiated
  together (one jvp per group, or "color"). The sparsity pattern is structural: it is read
  from the traced program (`jacobian_pattern`), not from values.
Only the columns of the decision rule that f depends on are interpolated.
"""

from __future__ import annotations

from functools import partial, reduce
from typing import Callable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from dolo.numeric import policy_operator_jax as poj

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------------------------
# Structural sparsity

# primitives whose output element depends only on the same element of the inputs
_ELEMENTWISE = {
    "abs", "acos", "acosh", "add", "and", "asin", "asinh", "atan", "atan2", "atanh", "cbrt",
    "ceil", "clamp", "convert_element_type", "copy", "copy_p", "cos", "cosh", "digamma", "div",
    "eq", "erf", "erf_inv", "erfc", "exp", "exp2", "expm1", "floor", "ge", "gt", "imag",
    "integer_pow", "is_finite", "le", "lgamma", "log", "log1p", "logistic", "lt", "max", "min",
    "mul", "ne", "neg", "nextafter", "not", "or", "pow", "real", "reduce_precision", "rem",
    "round", "rsqrt", "select_n", "sign", "sin", "sinh", "sqrt", "square", "stop_gradient",
    "sub", "tan", "tanh", "xor",
}
# primitives with a sub-program, and the parameter that holds it
_CALLS = {
    "pjit": "jaxpr",
    "closed_call": "call_jaxpr",
    "core_call": "call_jaxpr",
    "remat": "jaxpr",
    "checkpoint": "jaxpr",
    "custom_jvp_call": "call_jaxpr",
    "custom_vjp_call": "call_jaxpr",
    "custom_vjp_call_jaxpr": "fun_jaxpr",
}


def _zeros(shape):
    out = np.empty(shape, dtype=object)
    out.fill(0)
    return out


def _union_all(arrays):
    return reduce(lambda u, a: u | reduce(lambda p, q: p | q, a.ravel().tolist(), 0), arrays, 0)


def _is_zero(v):
    return isinstance(v, jax.core.Literal) and bool(np.all(np.asarray(v.val) == 0))


def _propagate(jaxpr, consts_deps, in_deps):
    """Dependencies of the outputs of `jaxpr` given those of its inputs.

    A dependency array has the shape of the value; each element is an int whose bit j is set
    when the element may depend on input j. Unknown primitives are handled conservatively
    (each output element depends on all input elements). Literal zeros are used: x^0 and 0*x
    do not depend on x (parameters are literals when the kernels are traced with their values,
    e.g. psi - gamma = 0 in integration_B)."""
    env = {}

    def read(v):
        if isinstance(v, jax.core.Literal):
            return _zeros(np.shape(v.val))
        return env[v]

    for v, d in zip(jaxpr.constvars, consts_deps):
        env[v] = d
    for v, d in zip(jaxpr.invars, in_deps):
        env[v] = d

    for eqn in jaxpr.eqns:
        ins = [read(v) for v in eqn.invars]
        name = eqn.primitive.name
        p = eqn.params
        outs = None
        if (
            (name == "pow" and _is_zero(eqn.invars[1]))
            or (name == "integer_pow" and p["y"] == 0)
            or (name == "mul" and any(_is_zero(v) for v in eqn.invars))
        ):
            outs = [_zeros(eqn.outvars[0].aval.shape)]
        elif name in _ELEMENTWISE and len(eqn.outvars) == 1:
            shape = eqn.outvars[0].aval.shape
            outs = [reduce(np.bitwise_or, [np.broadcast_to(a, shape) for a in ins])]
        elif name == "broadcast_in_dim":
            a = ins[0]
            shape = p["shape"]
            expanded = np.ones(len(shape), dtype=int)
            for src, dst in enumerate(p["broadcast_dimensions"]):
                expanded[dst] = a.shape[src]
            outs = [np.broadcast_to(a.reshape(tuple(expanded)), shape).copy()]
        elif name == "reshape":
            outs = [ins[0].reshape(p["new_sizes"])]
        elif name == "squeeze":
            outs = [np.squeeze(ins[0], axis=tuple(p["dimensions"]))]
        elif name == "expand_dims":
            outs = [np.expand_dims(ins[0], axis=tuple(p["dimensions"]))]
        elif name == "concatenate":
            outs = [np.concatenate(ins, axis=p["dimension"])]
        elif name == "slice":
            strides = p["strides"] or (1,) * len(p["start_indices"])
            idx = tuple(
                slice(s, l, st)
                for s, l, st in zip(p["start_indices"], p["limit_indices"], strides)
            )
            outs = [ins[0][idx]]
        elif name == "transpose":
            outs = [np.transpose(ins[0], p["permutation"])]
        elif name in _CALLS and _CALLS[name] in p:
            sub = p[_CALLS[name]]
            sub_jaxpr = sub.jaxpr if hasattr(sub, "jaxpr") else sub
            sub_consts = [_zeros(np.shape(c)) for c in getattr(sub, "consts", [])]
            outs = _propagate(sub_jaxpr, sub_consts, ins[: len(sub_jaxpr.invars)])
        if outs is None:
            dep = _union_all(ins)
            outs = []
            for v in eqn.outvars:
                o = _zeros(v.aval.shape)
                o[...] = dep
                outs.append(o)
        for v, o in zip(eqn.outvars, outs):
            o = np.asarray(o, dtype=object)
            env[v] = o if o.shape == tuple(v.aval.shape) else np.broadcast_to(o, v.aval.shape)
    return [read(v) for v in jaxpr.outvars]


def jacobian_pattern(fun: Callable, args: Sequence, argnum: int) -> np.ndarray:
    """Structural sparsity pattern (n_out, n_in) of the Jacobian of `fun(*args)` (a vector)
    w.r.t. `args[argnum]` (a vector): False where the output cannot depend on the input."""
    x = np.asarray(args[argnum])
    n_in = x.shape[0]

    def by_component(*xs):
        a = list(args)
        a[argnum] = jnp.stack(xs)
        return fun(*a)

    closed = jax.make_jaxpr(by_component)(*[x[i] for i in range(n_in)])
    in_deps = [_zeros(()) for _ in range(n_in)]
    for i, d in enumerate(in_deps):
        d[()] = 1 << i
    consts = [_zeros(np.shape(c)) for c in closed.consts]
    (out,) = _propagate(closed.jaxpr, consts, in_deps)
    out = out.reshape(-1)
    return np.array([[bool((int(o) >> j) & 1) for j in range(n_in)] for o in out])


def coloring(pattern: np.ndarray) -> List[List[int]]:
    """Groups of columns of `pattern` that share no row (greedy)."""
    groups: List[List[int]] = []
    for j in range(pattern.shape[1]):
        for g in groups:
            if not any((pattern[:, j] & pattern[:, i]).any() for i in g):
                g.append(j)
                break
        else:
            groups.append([j])
    return groups


class ColoredJacobian:
    """Forward-mode Jacobian of a function with a known sparsity pattern (n_out, n_in):
    one jvp per group of columns that share no row."""

    def __init__(self, pattern: np.ndarray):
        self.pattern = np.asarray(pattern, dtype=bool)
        n_in = self.pattern.shape[1]
        self.groups = coloring(self.pattern)
        seeds = np.zeros((len(self.groups), n_in))
        group_of = np.zeros(n_in, dtype=np.int32)
        for g, cols in enumerate(self.groups):
            seeds[g, cols] = 1.0
            group_of[cols] = g
        self.seeds = seeds
        self.group_of = group_of

    def value_and_compressed(self, fun, x):
        """f(x) and the jvps along the seeds: (n_out,), (n_groups, n_out)."""
        seeds = jnp.asarray(self.seeds, dtype=x.dtype)
        val, Y = jax.vmap(lambda v: jax.jvp(fun, (x,), (v,)))(seeds)
        return val[0], Y

    def expand(self, Y):
        """Jacobian (..., n_out, n_in) from compressed jvps (..., n_groups, n_out)."""
        J = jnp.swapaxes(jnp.take(Y, jnp.asarray(self.group_of), axis=-2), -1, -2)
        return J * jnp.asarray(self.pattern, dtype=Y.dtype)


# --------------------------------------------------------------------------------------------
# Kernels


def _over_regimes(fn, mapped, shared, batch):
    """fn(*mapped_i, *shared) for each regime i (leading axis of the `mapped` arrays): all at
    once (vmap) when `batch` is None, else `batch` regimes at a time (lax.map), which bounds
    the memory of the XLA temporaries."""
    if batch is None:
        return jax.vmap(lambda *m: fn(*m, *shared))(*mapped)
    return jax.lax.map(lambda m: fn(*m, *shared), tuple(mapped), batch_size=batch)


class GridKernels:
    """F, (F, dF_1) and dF_2 on (n_m, N, n_x) arrays of controls, for all regimes.

    Parameters
    ----------
    f_pt, g_pt : point functions f(m, s, x, M, S, X, p) and g(m, s, x, M, p).
    grid : ((min, max, n), ...) of the endogenous grid.
    order : 3 (cubic) or 1 (linear) interpolation.
    regimes : (K,) regime of each node of the exogenous process.
    m_vals, inodes, weights : (n_m, n_e), (n_m, K, n_e), (n_m, K).
    parms, s_nodes : parameters and grid nodes (N, d).
    max_points : the kernels process at most this many (regime, node, grid point) triples at
        once (more regimes are processed in sequence): bounds the XLA temporaries (about 0.8 kB
        per triple for F_and_dF_1 on integration_B).
    """

    def __init__(
        self, f_pt, g_pt, grid, order, regimes, m_vals, inodes, weights, parms, s_nodes, n_x,
        max_points=2**20,
    ):
        self.f_pt, self.g_pt = f_pt, g_pt
        self.grid = tuple((float(a), float(b), int(n)) for a, b, n in grid)
        self.order = order
        self.regimes = jnp.asarray(regimes, dtype=jnp.int32)[:, None]  # (K, 1)
        self.m_vals, self.inodes, self.weights = m_vals, inodes, weights
        self.parms, self.s_nodes = parms, s_nodes
        self.n_x = n_x
        n_m, K = inodes.shape[:2]
        N, d = s_nodes.shape
        self.d = d
        triples = n_m * K * N
        self.batch = None if triples <= max_points else max(1, max_points // (K * N))

        # structural sparsity, at an arbitrary point (it does not depend on the values of the
        # variables, but uses those of the parameters: see `_propagate`)
        m0, M0 = np.asarray(m_vals[0]), np.asarray(inodes[0, 0])
        s0, x0, S0, X0 = np.ones(d), np.ones(n_x), np.ones(d), np.ones(n_x)
        p0 = np.asarray(parms)
        pat_X = jacobian_pattern(lambda X: f_pt(m0, s0, x0, M0, S0, X, p0), (X0,), 0)
        # columns of the decision rule that f depends on: only these are interpolated
        self.xcols = tuple(int(j) for j in np.flatnonzero(pat_X.any(axis=0)))
        nc = len(self.xcols)

        def f_lin(m, s, u, M, S0, Xc0, Gc, p):
            """f with the decision rule linearized at S0: X = X0 + sum_j G_j (S_j(u) - S0_j)."""
            S = g_pt(m, s, u, M, p)
            dS = S - S0
            Xc = Xc0
            for j in range(d):
                Xc = Xc + Gc[j] * dS[j]
            return f_pt(m, s, u, M, S, self._scatter(Xc), p)

        G0 = [np.ones(nc)] * d
        pat_x = jacobian_pattern(
            lambda u: f_lin(m0, s0, u, M0, S0, np.ones(nc), G0, p0), (x0,), 0
        )
        self.jac_x = ColoredJacobian(pat_x)
        self.jac_X = ColoredJacobian(pat_X[:, list(self.xcols)])
        self._f_lin = f_lin
        self._build()

    def _scatter(self, Xc):
        """Full vector of next-period controls (..., n_x) from the used columns (..., nc); the
        other columns do not enter f."""
        X = jnp.zeros(Xc.shape[:-1] + (self.n_x,), dtype=Xc.dtype)
        return X.at[..., np.asarray(self.xcols, dtype=np.int32)].set(Xc)

    def _build(self):
        f_pt, g_pt, f_lin = self.f_pt, self.g_pt, self._f_lin
        order, grid, batch = self.order, self.grid, self.batch
        jac_x, jac_X, scatter = self.jac_x, self.jac_X, self._scatter
        xcols = np.asarray(self.xcols, dtype=np.int32)

        def over_nodes(one, axes_n, axes_k):
            """`one` mapped over grid points n and nodes k (within one regime)."""
            return jax.vmap(jax.vmap(one, axes_n), axes_k)

        @jax.jit
        def S_all(s, x, mv, IN, p):
            one = lambda m, sn, xn, M: g_pt(m, sn, xn, M, p)
            per_regime = over_nodes(one, (None, 0, 0, None), (None, None, None, 0))
            return jax.vmap(per_regime, (0, None, 0, 0))(mv, s, x, IN)

        @partial(jax.jit, static_argnames=("j",))
        def spline(C, S, regimes, j=-1):
            """Decision rule (columns xcols) at S (n_m, K, N, d); with j >= 0, its derivative
            w.r.t. S_j."""
            Cc = C[..., xcols]

            def per_regime(S_i):
                if j < 0:
                    base, W = poj.locate(S_i, regimes, grid, order)
                else:
                    base, W, dW = poj.locate(S_i, regimes, grid, order, derivative=True)
                    W = W.at[..., j, :].set(dW[..., j, :])
                return poj.evaluate(Cc, base, W, order)

            return _over_regimes(per_regime, (S,), (), batch)

        # within a regime: arguments (m, x_i, IN_i, W_i, S_i, Xc_i, ...) and s
        axes_n = (None, 0, None, None, 0, 0)
        axes_k = (None, None, 0, 0, 0, 0)

        @jax.jit
        def F_from(s, x, mv, IN, Wt, S, Xc, p):
            def per_regime(m, x_i, IN_i, W_i, S_i, Xc_i):
                def one(m, xn, M, w, Sk, Xk, sn):
                    return w * f_pt(m, sn, xn, M, Sk, scatter(Xk), p)

                r = over_nodes(one, axes_n + (0,), axes_k + (None,))(m, x_i, IN_i, W_i, S_i, Xc_i, s)
                return r.sum(axis=0)

            return _over_regimes(per_regime, (mv, x, IN, Wt, S, Xc), (), batch)

        @jax.jit
        def F_and_dF_from(s, x, mv, IN, Wt, S, Xc, Gs, p):
            def per_regime(m, x_i, IN_i, W_i, S_i, Xc_i, Gs_i):
                def one(m, xn, M, w, Sk, Xk, Gk, sn):
                    val, Y = jac_x.value_and_compressed(
                        lambda u: f_lin(m, sn, u, M, Sk, Xk, Gk, p), xn
                    )
                    return w * val, w * Y

                r, Y = over_nodes(one, axes_n + (0, 0), axes_k + (0, None))(
                    m, x_i, IN_i, W_i, S_i, Xc_i, Gs_i, s
                )
                return r.sum(axis=0), jac_x.expand(Y.sum(axis=0))

            return _over_regimes(per_regime, (mv, x, IN, Wt, S, Xc, Gs), (), batch)

        @jax.jit
        def dF_2_from(s, x, mv, IN, Wt, S, Xc, p):
            def per_regime(m, x_i, IN_i, W_i, S_i, Xc_i):
                def one(m, xn, M, w, Sk, Xk, sn):
                    _, Y = jac_X.value_and_compressed(
                        lambda u: f_pt(m, sn, xn, M, Sk, scatter(u), p), Xk
                    )
                    return w * Y

                Y = over_nodes(one, axes_n + (0,), axes_k + (None,))(m, x_i, IN_i, W_i, S_i, Xc_i, s)
                return jac_X.expand(Y)  # (K, N, n_x, nc)

            return _over_regimes(per_regime, (mv, x, IN, Wt, S, Xc), (), batch)

        self._S_all, self._spline = S_all, spline
        self._F_from, self._F_and_dF_from, self._dF_2_from = F_from, F_and_dF_from, dF_2_from

    def _args(self, x):
        return self.s_nodes, x, self.m_vals, self.inodes

    def F(self, x, C):
        """Residuals (n_m, N, n_x) for controls x (n_m, N, n_x) and coefficients C."""
        S = self._S_all(*self._args(x), self.parms)
        Xc = self._spline(C, S, self.regimes)
        return self._F_from(*self._args(x), self.weights, S, Xc, self.parms)

    def F_and_dF_1(self, x, C):
        """Residuals (n_m, N, n_x) and their Jacobian w.r.t. x (n_m, N, n_x, n_x)."""
        S = self._S_all(*self._args(x), self.parms)
        # value and gradient of the decision rule: one jit each (faster than a single one)
        Xc = self._spline(C, S, self.regimes)
        Gs = tuple(self._spline(C, S, self.regimes, j=j) for j in range(self.d))
        return self._F_and_dF_from(*self._args(x), self.weights, S, Xc, Gs, self.parms)

    def dF_2_compressed(self, x, C):
        """Columns `xcols` (the others are zero) of the blocks F_x (n_m, K, N, n_x, len(xcols))
        of the derivative w.r.t. the decision rule, and the next-period states S (n_m, K, N, d)."""
        S = self._S_all(*self._args(x), self.parms)
        Xc = self._spline(C, S, self.regimes)
        return self._dF_2_from(*self._args(x), self.weights, S, Xc, self.parms), S

    def dF_2(self, x, C):
        """Blocks F_x (n_m, K, N, n_x, n_x) of the derivative w.r.t. the decision rule, and the
        next-period states S (n_m, K, N, d)."""
        F_c, S = self.dF_2_compressed(x, C)
        return poj.densify(F_c, self.xcols, self.n_x), S
