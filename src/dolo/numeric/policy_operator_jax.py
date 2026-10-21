"""
JAX kernels of the policy linear operator L = dF_2 (`PolicyLinearOperator`).

For a variation v of the decision rule (one array (N, n_x) per exogenous regime),

    (L.v)[i, n] = scale * sum_k F_x[i, k, n] . v_{r(k)}(S[i, k, n])

where v_r is the spline (cubic or linear) interpolating v in regime r, S[i, k, n] the next-period
state of node k from grid point n in regime i, and r(k) the regime of node k.

S does not change while L is applied (e.g. during a linear solve), so the location of each S in
the grid (flat index of its cell and 1-D basis weights) is computed once, by `locate`. Columns of
F_x that are zero everywhere are dropped (`compress`): L.v then only needs these columns of v,
which saves prefilters and spline evaluations (for integration_B, 5 of 8 columns are used).

Each step is jitted separately: compiled as a single function, XLA fuses the spline evaluation
into the contraction and recomputes it for each output (5× slower on integration_B).
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from dolo.numeric.interpolation_jax import (
    Ad,
    Ad_deriv0,
    Ad_deriv1,
    Ad_val0,
    Ad_val1,
    get_filter_matrix,
)

jax.config.update("jax_enable_x64", True)


def _cubic_basis(lam):
    """The 4 cubic B-spline weights at lam, with C1 linear extrapolation outside [0, 1]
    (as `eval_cubic_point_jax`)."""
    l2 = lam * lam
    l3 = l2 * lam
    out = []
    for j in range(4):
        inner = Ad[j, 0] * l3 + Ad[j, 1] * l2 + Ad[j, 2] * lam + Ad[j, 3]
        left = Ad_val0[j] + Ad_deriv0[j] * lam
        right = Ad_val1[j] + Ad_deriv1[j] * (lam - 1.0)
        out.append(jnp.where(lam < 0.0, left, jnp.where(lam > 1.0, right, inner)))
    return out


def _linear_basis(lam):
    return [1.0 - lam, lam]


def coeffs_shape(grid_n: Tuple[int, ...], order: int) -> Tuple[int, ...]:
    """Shape of the interpolation coefficients of one function on the grid."""
    return tuple(g + 2 for g in grid_n) if order == 3 else tuple(grid_n)


def _strides(shape):
    return tuple(int(np.prod(shape[d + 1 :])) for d in range(len(shape)))


@partial(jax.jit, static_argnames=("grid", "order", "derivative"))
def locate(S, regimes, grid, order, derivative=False):
    """Locate points S in the grid.

    Parameters
    ----------
    S : (..., d) points.
    regimes : int array broadcastable to S.shape[:-1], regime of the spline evaluated at each point.
    grid : static tuple ((min, max, n), ...) of the endogenous grid.
    order : 3 (cubic) or 1 (linear).

    Returns
    -------
    base : (...) int32, flat index of the first coefficient of the cell, in the coefficients of
        all regimes stacked (n_regimes, *coeffs_shape).
    weights : (..., d, order + 1) 1-D basis weights.
    dweights : (..., d, order + 1), only if `derivative`: derivatives of the weights w.r.t. S
        (the gradient of the spline w.r.t. S_d is the nested sum with dweights in dimension d).
    """
    a = jnp.asarray([g[0] for g in grid], dtype=S.dtype)
    b = jnp.asarray([g[1] for g in grid], dtype=S.dtype)
    n = np.asarray([g[2] for g in grid], dtype=np.int32)
    shape = coeffs_shape(tuple(int(v) for v in n), order)
    strides = _strides(shape)
    u = (S - a) * ((n - 1.0) / (b - a))
    i = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n - 2)
    lam = u - i
    base = jnp.asarray(regimes, dtype=jnp.int32) * int(np.prod(shape))
    for d in range(len(grid)):
        base = base + i[..., d] * strides[d]
    basis = _cubic_basis if order == 3 else _linear_basis
    weights = jnp.stack(
        [jnp.stack(basis(lam[..., d]), axis=-1) for d in range(len(grid))], axis=-2
    )
    if not derivative:
        return base, weights
    delta = (n - 1.0) / (b - a)
    dweights = jnp.stack(
        [
            jnp.stack(
                jax.jvp(basis, (lam[..., d],), (jnp.ones_like(lam[..., d]),))[1], axis=-1
            )
            * delta[d]
            for d in range(len(grid))
        ],
        axis=-2,
    )
    return base, weights, dweights


@partial(jax.jit, static_argnames=("grid_n", "cols", "order"))
def prefilter(v, grid_n, cols, order):
    """Interpolation coefficients (n_regimes, *coeffs_shape, len(cols)) of the columns `cols` of
    v (n_regimes, N, n_x)."""
    V = v[..., np.asarray(cols, dtype=np.int32)]
    V = V.reshape((v.shape[0],) + tuple(grid_n) + (len(cols),))
    if order == 3:
        for ax, g in enumerate(grid_n):
            W = jnp.asarray(get_filter_matrix(g), dtype=V.dtype)  # (g + 2, g)
            V = jnp.moveaxis(jnp.tensordot(W, V, axes=([1], [ax + 1])), 0, ax + 1)
    return V


@partial(jax.jit, static_argnames=("order",))
def evaluate(C, base, weights, order):
    """Values (..., n_c) at the points located by `locate` of the splines with coefficients
    C (n_regimes, *coeffs_shape, n_c): sum_p w0p sum_q w1q ... c, one dimension at a time.
    `weights` is (..., d, order + 1), or a list of d arrays (..., order + 1)."""
    shape = C.shape[1:-1]
    strides = _strides(shape)
    cf = C.reshape(-1, C.shape[-1])
    d = len(shape)
    if not isinstance(weights, (list, tuple)):
        weights = [weights[..., dim, :] for dim in range(d)]

    def nested(dim, offset):
        if dim == d:
            return cf[base + offset]
        acc = 0.0
        for p in range(order + 1):
            acc = acc + weights[dim][..., p, None] * nested(dim + 1, offset + p * strides[dim])
        return acc

    return nested(0, 0)


@jax.jit
def contract(Fc, X, scale):
    """scale * sum_k Fc[i, k, n] . X[i, k, n]: Fc (n_m, K, N, n_x, n_c), X (n_m, K, N, n_c)
    -> (n_m, N, n_x). Written elementwise (faster than einsum's batched products)."""
    acc = jnp.zeros(Fc.shape[:-1], dtype=Fc.dtype)
    for j in range(Fc.shape[-1]):
        acc = acc + Fc[..., j] * X[..., j, None]
    return scale * acc.sum(axis=1)


@jax.jit
def compress(F_x):
    """Mask of the columns of F_x (..., n_x, n_x) that are not zero everywhere."""
    return jnp.any(F_x != 0, axis=tuple(range(F_x.ndim - 1)))


@jax.jit
def precondition(J, Fc):
    """J^{-1} Fc for block-diagonal J (n_m, N, n_x, n_x) and Fc (n_m, K, N, n_x, n_c): each
    block J[i, n] is inverted once, and the products are written elementwise (a batched solve
    with the K * n_c right-hand sides needs transposed copies of Fc: slower, more memory)."""
    Jinv = jnp.linalg.inv(J)[:, None]  # (n_m, 1, N, n_x, n_x)
    out = jnp.zeros_like(Fc)
    for q in range(Fc.shape[-2]):
        out = out + Jinv[..., :, q, None] * Fc[..., q, None, :]
    return out


def apply(v, Fc, base, weights, scale, grid_n, cols, order):
    """L.v for v (n_m, N, n_x)."""
    if len(cols) == 0:
        return jnp.zeros_like(v)
    C = prefilter(v, grid_n, cols, order)
    X = evaluate(C, base, weights, order)
    return contract(Fc, X, scale)


@partial(
    jax.jit, static_argnames=("grid_n", "cols", "order", "restart", "maxiter")
)
def gmres_I_minus_L(b, Fc, base, weights, scale, tol, grid_n, cols, order, restart, maxiter):
    """Solve (I - L) x = b with GMRES on the device, starting from x0 = b. Stops when
    |b - (I - L) x| <= tol |b| (as scipy's gmres with rtol=tol, atol=0)."""

    def matvec(x):
        return x - apply(x, Fc, base, weights, scale, grid_n, cols, order)

    x, _ = jax.scipy.sparse.linalg.gmres(
        matvec,
        b,
        x0=b,
        tol=tol,
        atol=0.0,
        restart=restart,
        maxiter=maxiter,
        solve_method="incremental",
    )
    return x
