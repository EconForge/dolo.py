"""
Pure JAX cubic spline interpolation with C1 linear extrapolation on Cartesian grids.

This module mirrors the behavior of EconForge/interpolation.py, providing:
1. Prefiltering: solves tridiagonal systems with natural boundary conditions (f''(a) = f''(b) = 0).
2. Evaluation: C^1 linear extrapolation (tangent at boundary) for points outside grid bounds,
   preventing exponential/polynomial divergence in dynamic programming algorithms.
3. Compatibility with jax.jit, jax.vmap, and autodiff (jax.grad, jax.jacobian, jax.vjp).
"""

from __future__ import annotations

import functools
from typing import Any, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


# Cubic B-spline matrix (basis coefficients for [t^3, t^2, t, 1])
# Ad[j, :] gives polynomial coefficients for j-th basis function on interval [0, 1]
Ad = jnp.array(
    [
        [-1.0 / 6.0, 3.0 / 6.0, -3.0 / 6.0, 1.0 / 6.0],
        [3.0 / 6.0, -6.0 / 6.0, 0.0 / 6.0, 4.0 / 6.0],
        [-3.0 / 6.0, 3.0 / 6.0, 3.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 0.0 / 6.0, 0.0 / 6.0, 0.0 / 6.0],
    ],
    dtype=jnp.float64,
)

# Precomputed boundary values and derivatives of basis functions:
# At lam = 0:
# Value is Ad[:, 3]
# Derivative is Ad[:, 2]
Ad_val0 = Ad[:, 3]
Ad_deriv0 = Ad[:, 2]

# At lam = 1:
# Value is sum(Ad, axis=1)
# Derivative is 3*Ad[:, 0] + 2*Ad[:, 1] + Ad[:, 2]
Ad_val1 = jnp.sum(Ad, axis=1)
Ad_deriv1 = 3.0 * Ad[:, 0] + 2.0 * Ad[:, 1] + Ad[:, 2]


@functools.lru_cache(maxsize=128)
def get_filter_matrix(M: int) -> jnp.ndarray:
    """
    Computes the prefiltering matrix W of shape (M + 2, M) such that
    coeffs = W @ values solves the natural cubic spline tridiagonal system.

    Parameters
    ----------
    M : int
        Number of grid points along the dimension.

    Returns
    -------
    W : jnp.ndarray of shape (M + 2, M)
    """
    if M < 2:
        raise ValueError(f"Grid size along each dimension must be at least 2, got {M}")

    # Construct the (M + 2, M + 2) system matrix A
    A = np.zeros((M + 2, M + 2), dtype=np.float64)
    # Row 0: f''(a) = 0 -> c_0 - 2*c_1 + c_2 = 0
    A[0, 0:3] = [1.0, -2.0, 1.0]
    # Row M + 1: f''(b) = 0 -> c_{M-1} - 2*c_M + c_{M+1} = 0
    A[M + 1, M - 1 : M + 2] = [1.0, -2.0, 1.0]
    # Internal rows: 1/6 * c_i + 2/3 * c_{i+1} + 1/6 * c_{i+2} = y_i
    for i in range(M):
        A[i + 1, i : i + 3] = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]

    A_inv = np.linalg.inv(A)
    # The rhs vector has shape (M + 2,) with zeros at index 0 and M+1, and values at 1..M.
    # Therefore, coeffs = A_inv[:, 1:M+1] @ values.
    W = A_inv[:, 1 : M + 1]
    return W


def prefilter_cubic_jax(
    grid_or_shape: Union[Any, Sequence[int], Tuple[int, ...]],
    values: Union[jnp.ndarray, np.ndarray],
    d: int | None = None,
) -> jnp.ndarray:
    """
    Computes prefiltered cubic B-spline coefficients on a uniform Cartesian grid in JAX.

    Parameters
    ----------
    grid_or_shape :
        Either a Grid object (with attribute `n`), or a tuple/list of grid specifications
        `((min, max, n), ...)`, or a tuple of integers specifying grid dimensions `(n_0, ..., n_{d-1})`.
    values : jnp.ndarray or np.ndarray
        Array of values on the grid. Shape must start with (n_0, ..., n_{d-1}),
        optionally followed by trailing value dimensions (for vector-valued functions).
    d : int, optional
        Number of grid dimensions. If None, inferred from `grid_or_shape` or `values.ndim`.

    Returns
    -------
    coeffs : jnp.ndarray
        Filtered B-spline coefficients of shape (n_0 + 2, ..., n_{d-1} + 2, ...).
    """
    shape = _extract_grid_shape(grid_or_shape)
    if d is None:
        d = len(shape)

    coeffs = jnp.asarray(values, dtype=jnp.float64)

    for axis in range(d):
        M = shape[axis]
        W = jnp.asarray(get_filter_matrix(M), dtype=jnp.float64)  # (M + 2, M)

        # Contract axis of coeffs with axis 1 of W:
        # coeffs has shape (..., M, ...). We dot W with that axis.
        coeffs = jnp.tensordot(coeffs, W.T, axes=([axis], [0]))
        coeffs = jnp.moveaxis(coeffs, -1, axis)

    return coeffs


def eval_cubic_basis_1d(lam: jnp.ndarray, extrap_mode: str = "linear") -> jnp.ndarray:
    """
    Evaluates the 4 cubic B-spline basis functions for relative coordinate lam.

    Parameters
    ----------
    lam : scalar or array
        Relative coordinate inside grid interval (lam = u - i).
    extrap_mode : str
        'linear': C^1 linear extrapolation outside [0, 1] using boundary tangent.
        'nearest': clamping (evaluated as standard basis).

    Returns
    -------
    phi : jnp.ndarray of shape (4,) or (..., 4)
    """
    # Polynomial vector [lam^3, lam^2, lam, 1.0]
    u = jnp.stack([lam**3, lam**2, lam, jnp.ones_like(lam)], axis=-1)
    phi_int = jnp.tensordot(u, Ad, axes=([-1], [1]))

    if extrap_mode == "linear":
        # Tangent at lam = 0: val0 + deriv0 * lam
        phi_left = (
            Ad_val0 + jnp.outer(lam, Ad_deriv0).squeeze()
            if lam.ndim > 0
            else Ad_val0 + Ad_deriv0 * lam
        )
        # Tangent at lam = 1: val1 + deriv1 * (lam - 1.0)
        lam_m1 = lam - 1.0
        phi_right = (
            Ad_val1 + jnp.outer(lam_m1, Ad_deriv1).squeeze()
            if lam.ndim > 0
            else Ad_val1 + Ad_deriv1 * lam_m1
        )

        return jnp.where(
            lam[..., None] < 0.0 if lam.ndim > 0 else lam < 0.0,
            phi_left,
            jnp.where(
                lam[..., None] > 1.0 if lam.ndim > 0 else lam > 1.0,
                phi_right,
                phi_int,
            ),
        )
    elif extrap_mode == "nearest":
        return phi_int
    else:
        raise ValueError(
            f"Unsupported extrap_mode: {extrap_mode}. Expected 'linear' or 'nearest'."
        )


def eval_cubic_point_jax(
    coeffs: jnp.ndarray,
    x: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    n: jnp.ndarray,
    extrap_mode: str = "linear",
) -> jnp.ndarray:
    """
    Evaluates an N-dimensional cubic spline at a single coordinate vector x in JAX.

    Parameters
    ----------
    coeffs : jnp.ndarray
        Filtered coefficients of shape (n_0 + 2, ..., n_{d-1} + 2, ...).
    x : jnp.ndarray of shape (d,)
        Point coordinates.
    a : jnp.ndarray of shape (d,)
        Lower bounds along each dimension.
    b : jnp.ndarray of shape (d,)
        Upper bounds along each dimension.
    n : jnp.ndarray of shape (d,)
        Number of nodes along each dimension.
    extrap_mode : str
        'linear' or 'nearest'.

    Returns
    -------
    val : jnp.ndarray
        Interpolated scalar or vector of values.
    """
    d = a.shape[0]
    delta = (n - 1.0) / (b - a)

    if extrap_mode == "nearest":
        x = jnp.clip(x, a, b)

    u = (x - a) * delta
    # Stop gradient on integer cell indices so indexing doesn't introduce spurious zero gradients
    i = jax.lax.stop_gradient(jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n - 2))
    lam = u - i

    start_indices = tuple(jnp.int32(i[k]) for k in range(d)) + (jnp.int32(0),) * (
        coeffs.ndim - d
    )
    slice_sizes = (4,) * d + coeffs.shape[d:]
    patch = jax.lax.dynamic_slice(coeffs, start_indices, slice_sizes)

    # Contract patch with 1D basis functions sequentially along each dimension
    for k in range(d):
        lam_k = lam[k]
        u_poly = jnp.stack([lam_k**3, lam_k**2, lam_k, 1.0])
        phi_int = Ad @ u_poly
        if extrap_mode == "linear":
            phi_left = Ad_val0 + Ad_deriv0 * lam_k
            phi_right = Ad_val1 + Ad_deriv1 * (lam_k - 1.0)
            phi_k = jnp.where(
                lam_k < 0.0, phi_left, jnp.where(lam_k > 1.0, phi_right, phi_int)
            )
        else:
            phi_k = phi_int

        # Contract phi_k with axis 0 of patch
        patch = jnp.tensordot(phi_k, patch, axes=([0], [0]))

    return patch


@functools.partial(jax.jit, static_argnames=("extrap_mode",))
def _eval_cubic_batch_jit(
    coeffs: jnp.ndarray,
    pts: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    n: jnp.ndarray,
    extrap_mode: str = "linear",
) -> jnp.ndarray:
    eval_pt = functools.partial(
        eval_cubic_point_jax,
        coeffs,
        a=a,
        b=b,
        n=n,
        extrap_mode=extrap_mode,
    )
    return jax.vmap(eval_pt)(pts)


def eval_cubic_jax(
    grid: Any,
    coeffs: Union[jnp.ndarray, np.ndarray],
    coords: Union[jnp.ndarray, np.ndarray],
    extrap_mode: str = "linear",
) -> jnp.ndarray:
    """
    Evaluates an N-dimensional cubic spline on uniform Cartesian grid for coordinates in JAX.

    Parameters
    ----------
    grid :
        Grid object (e.g. UniformCartesianGrid) or tuple of `(min, max, n)`.
    coeffs : jnp.ndarray or np.ndarray
        Prefiltered cubic B-spline coefficients.
    coords : jnp.ndarray or np.ndarray
        Coordinate array of shape (d,) or (N, d).
    extrap_mode : str
        'linear' (default) or 'nearest'.

    Returns
    -------
    result : jnp.ndarray
        Interpolated values of shape (...) or (N, ...).
    """
    a, b, n = _extract_grid_bounds(grid)
    coeffs_jax = jnp.asarray(coeffs, dtype=jnp.float64)
    pts = jnp.asarray(coords, dtype=jnp.float64)

    is_1d_pt = (pts.ndim == 1) and (len(a) > 1 or pts.shape[0] == 1)
    if pts.ndim == 0 or (pts.ndim == 1 and len(a) == 1 and pts.shape[0] != 1):
        # 1D grid with batch of 1D scalar query points
        pts = pts.reshape(-1, 1)
        is_1d_pt = False
    elif is_1d_pt:
        pts = pts.reshape(1, -1)

    res = _eval_cubic_batch_jit(coeffs_jax, pts, a, b, n, extrap_mode=extrap_mode)

    if is_1d_pt:
        return res[0]
    return res


def eval_linear_point_jax(
    values: jnp.ndarray,
    x: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    n: jnp.ndarray,
    extrap_mode: str = "linear",
) -> jnp.ndarray:
    """
    Evaluates an N-dimensional multilinear interpolation at a single coordinate vector x in JAX.

    Parameters
    ----------
    values : jnp.ndarray
        Array of values on the grid of shape (n_0, ..., n_{d-1}, ...).
    x : jnp.ndarray of shape (d,)
        Point coordinates.
    a : jnp.ndarray of shape (d,)
        Lower bounds along each dimension.
    b : jnp.ndarray of shape (d,)
        Upper bounds along each dimension.
    n : jnp.ndarray of shape (d,)
        Number of nodes along each dimension.
    extrap_mode : str
        'linear' or 'nearest'.

    Returns
    -------
    val : jnp.ndarray
        Interpolated scalar or vector of values.
    """
    d = a.shape[0]
    delta = (n - 1.0) / (b - a)

    if extrap_mode == "nearest":
        x = jnp.clip(x, a, b)

    u = (x - a) * delta
    i = jax.lax.stop_gradient(jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n - 2))
    lam = u - i

    start_indices = tuple(jnp.int32(i[k]) for k in range(d)) + (jnp.int32(0),) * (
        values.ndim - d
    )
    slice_sizes = (2,) * d + values.shape[d:]
    patch = jax.lax.dynamic_slice(values, start_indices, slice_sizes)

    for k in range(d):
        lam_k = lam[k]
        phi_k = jnp.stack([1.0 - lam_k, lam_k])
        patch = jnp.tensordot(phi_k, patch, axes=([0], [0]))

    return patch


@functools.partial(jax.jit, static_argnames=("extrap_mode",))
def _eval_linear_batch_jit(
    values: jnp.ndarray,
    pts: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    n: jnp.ndarray,
    extrap_mode: str = "linear",
) -> jnp.ndarray:
    eval_pt = functools.partial(
        eval_linear_point_jax,
        values,
        a=a,
        b=b,
        n=n,
        extrap_mode=extrap_mode,
    )
    return jax.vmap(eval_pt)(pts)


def eval_linear_jax(
    grid: Any,
    values: Union[jnp.ndarray, np.ndarray],
    coords: Union[jnp.ndarray, np.ndarray],
    extrap_mode: str = "linear",
) -> jnp.ndarray:
    """
    Evaluates an N-dimensional multilinear interpolation on uniform Cartesian grid in JAX.

    Parameters
    ----------
    grid :
        Grid object (e.g. UniformCartesianGrid) or tuple of `(min, max, n)`.
    values : jnp.ndarray or np.ndarray
        Field values on grid of shape (n_0, ..., n_{d-1}, ...).
    coords : jnp.ndarray or np.ndarray
        Coordinate array of shape (d,) or (N, d).
    extrap_mode : str
        'linear' (default) or 'nearest'.

    Returns
    -------
    result : jnp.ndarray
        Interpolated values of shape (...) or (N, ...).
    """
    a, b, n = _extract_grid_bounds(grid)
    vals_jax = jnp.asarray(values, dtype=jnp.float64)
    pts = jnp.asarray(coords, dtype=jnp.float64)

    is_1d_pt = (pts.ndim == 1) and (len(a) > 1 or pts.shape[0] == 1)
    if pts.ndim == 0 or (pts.ndim == 1 and len(a) == 1 and pts.shape[0] != 1):
        pts = pts.reshape(-1, 1)
        is_1d_pt = False
    elif is_1d_pt:
        pts = pts.reshape(1, -1)

    res = _eval_linear_batch_jit(vals_jax, pts, a, b, n, extrap_mode=extrap_mode)

    if is_1d_pt:
        return res[0]
    return res


def _extract_grid_shape(grid_or_shape: Any) -> Tuple[int, ...]:
    """Helper to extract shape tuple (n_0, ..., n_{d-1}) from grid or sequence."""
    if hasattr(grid_or_shape, "n"):
        return tuple(int(x) for x in grid_or_shape.n)
    if isinstance(grid_or_shape, (list, tuple)):
        first = grid_or_shape[0]
        if isinstance(first, (int, np.integer)):
            return tuple(int(x) for x in grid_or_shape)
        if isinstance(first, (tuple, list)) and len(first) == 3:
            return tuple(int(dim[2]) for dim in grid_or_shape)
    raise ValueError(f"Cannot extract grid shape from: {grid_or_shape}")


def _extract_grid_bounds(grid: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Helper to extract (a, b, n) arrays from a Grid or tuple representation."""
    if hasattr(grid, "min") and hasattr(grid, "max") and hasattr(grid, "n"):
        a = jnp.asarray(grid.min, dtype=jnp.float64)
        b = jnp.asarray(grid.max, dtype=jnp.float64)
        n = jnp.asarray(grid.n, dtype=jnp.int32)
        return a, b, n
    if hasattr(grid, "__numba_repr__"):
        repr_tuple = grid.__numba_repr__()
        return _extract_grid_bounds(repr_tuple)
    if isinstance(grid, (tuple, list)):
        a = jnp.array([dim[0] for dim in grid], dtype=jnp.float64)
        b = jnp.array([dim[1] for dim in grid], dtype=jnp.float64)
        n = jnp.array([dim[2] for dim in grid], dtype=jnp.int32)
        return a, b, n
    raise ValueError(f"Unsupported grid representation: {grid}")
