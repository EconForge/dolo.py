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


def _precondition(J, Fc):
    Jinv = jnp.linalg.inv(J)[:, None]  # (n_m, 1, N, n_x, n_x)
    out = jnp.zeros_like(Fc)
    for q in range(Fc.shape[-2]):
        out = out + Jinv[..., :, q, None] * Fc[..., q, None, :]
    return out


@partial(jax.jit, static_argnames=("by_regime",))
def precondition(J, Fc, by_regime=False):
    """J^{-1} Fc for block-diagonal J (n_m, N, n_x, n_x) and Fc (n_m, K, N, n_x, n_c): each
    block J[i, n] is inverted once, and the products are written elementwise (a batched solve
    with the K * n_c right-hand sides needs transposed copies of Fc: slower, more memory).
    `by_regime`: one regime at a time, to bound the temporaries (slower)."""
    if not by_regime:
        return _precondition(J, Fc)
    return jax.lax.map(lambda a: _precondition(a[0][None], a[1][None])[0], (J, Fc))


def densify(Fc, cols, n_x, scale=1.0):
    """Blocks (..., n_x, n_x) from their columns `cols` (..., n_x, len(cols)), the others zero
    (a stack: a scatter into zeros needs as much memory again for XLA temporaries)."""
    pos = {c: i for i, c in enumerate(cols)}
    zero = jnp.zeros(Fc.shape[:-1], dtype=Fc.dtype)
    return jnp.stack([scale * Fc[..., pos[j]] if j in pos else zero for j in range(n_x)], axis=-1)


# above this number of blocks, the preconditioning goes one regime at a time
MAX_BLOCKS = 2**20


@jax.jit
def axmy(x, y):
    """x - y."""
    return x - y


def apply(v, Fc, base, weights, scale, grid_n, cols, order):
    """L.v for v (n_m, N, n_x)."""
    if len(cols) == 0:
        return jnp.zeros_like(v)
    C = prefilter(v, grid_n, cols, order)
    X = evaluate(C, base, weights, order)
    return contract(Fc, X, scale)


# --------------------------------------------------------------------------------------------
# GMRES as Dolo.jl's `gmres!` (src/algos/linsolve.jl): one cycle of at most m Krylov vectors,
# modified Gram-Schmidt, Givens rotations, absolute tolerance on the 2-norm of the residual.
# The Krylov basis and the matrix-vector products stay on the device; the small (m+1) x m
# Hessenberg problem is solved on the host (one transfer of a column per iteration). As fast
# as `jax.scipy.sparse.linalg.gmres` here (whose tolerance is relative, with
# reorthogonalization), with the iterations and the stopping rule of Dolo.jl.


@partial(jax.jit, donate_argnums=(0,))
def _arnoldi_step(V, w, j):
    """Orthogonalize w against V[0..j] (modified Gram-Schmidt), store w / |w| in V[j+1] (when
    j + 1 < len(V)). Returns V and the column h[0..j+1] (zeros beyond)."""

    def body(i, carry):
        w, h = carry
        v_i = jax.lax.dynamic_index_in_dim(V, i, keepdims=False)
        h_i = jnp.vdot(v_i, w)
        return w - h_i * v_i, h.at[i].set(h_i)

    h = jnp.zeros(V.shape[0], dtype=w.dtype)
    w, h = jax.lax.fori_loop(0, j + 1, body, (w, h))
    h_norm = jnp.linalg.norm(w)
    h = h.at[j + 1].set(h_norm)
    V = jax.lax.dynamic_update_index_in_dim(V, w / h_norm, jnp.minimum(j + 1, V.shape[0] - 1), 0)
    return V, h


@partial(jax.jit, donate_argnums=(0,))
def _krylov_init(V, r, beta):
    return V.at[0].set(r / beta)


@jax.jit
def _krylov_vector(V, j):
    return jax.lax.dynamic_index_in_dim(V, j, keepdims=False)


@jax.jit
def _krylov_update(x, V, y):
    """x + sum_i y_i V[i] (y padded with zeros)."""
    return x + jnp.tensordot(y, V, axes=1)


def gmres(matvec, b, x0, tol, m):
    """Solve A x = b, A given by `matvec` on flat device vectors, starting from x0: one cycle of
    at most m Krylov vectors, until |b - A x|_2 < tol. Returns x and the number of iterations."""
    n = b.shape[0]
    x = x0
    r = b - matvec(x)
    beta = float(jnp.linalg.norm(r))
    if beta < tol:
        return x, 0
    V = _krylov_init(jnp.zeros((m + 1, n), dtype=b.dtype), r, beta)
    H = np.zeros((m + 1, m))
    cs = np.zeros(m)
    sn = np.zeros(m)
    g = np.zeros(m + 1)
    g[0] = beta
    eps = np.finfo(np.float64).eps
    k = 0
    for j in range(m):
        w = matvec(_krylov_vector(V, j))
        V, h_col = _arnoldi_step(V, w, j)
        h_col = np.asarray(h_col)
        H[: j + 2, j] = h_col[: j + 2]
        h = H[j + 1, j]
        # previous rotations on the new column
        for i in range(j):
            t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
            H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
            H[i, j] = t
        # new rotation eliminating H[j+1, j]
        rho = np.hypot(H[j, j], H[j + 1, j])
        cs[j] = H[j, j] / rho
        sn[j] = H[j + 1, j] / rho
        H[j, j] = rho
        H[j + 1, j] = 0.0
        g[j + 1] = -sn[j] * g[j]
        g[j] = cs[j] * g[j]
        k = j + 1
        # converged, or lucky breakdown (the Krylov space is invariant)
        if abs(g[j + 1]) < tol or h <= eps * beta:
            break
    y = np.zeros(m + 1)
    for i in range(k - 1, -1, -1):
        y[i] = (g[i] - H[i, i + 1 : k] @ y[i + 1 : k]) / H[i, i]
    return _krylov_update(x, V, jnp.asarray(y, dtype=b.dtype)), k
