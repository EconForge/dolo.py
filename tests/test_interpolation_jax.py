import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from interpolation.splines import eval_cubic
from interpolation.splines.prefilter_cubic import prefilter_cubic

from dolo.numeric.grids import UniformCartesianGrid
from dolo.numeric.approximators import CubicSplineApproximator
from dolo.numeric.interpolation_jax import (
    prefilter_cubic_jax,
    eval_cubic_jax,
    eval_cubic_point_jax,
    eval_linear_jax,
    eval_linear_point_jax,
)


def test_prefilter_1d_parity():
    grid = ((0.0, 1.0, 5),)
    vals = np.array([1.0, 2.5, 3.2, 4.0, 5.1])
    coeffs_ref = prefilter_cubic(grid, vals)
    coeffs_jax = prefilter_cubic_jax(grid, vals)
    assert np.allclose(coeffs_ref, np.array(coeffs_jax), atol=1e-13)


def test_prefilter_2d_parity():
    grid = ((0.0, 1.0, 4), (0.0, 2.0, 5))
    vals = np.random.RandomState(42).randn(4, 5)
    coeffs_ref = prefilter_cubic(grid, vals)
    coeffs_jax = prefilter_cubic_jax(grid, vals)
    assert np.allclose(coeffs_ref, np.array(coeffs_jax), atol=1e-13)


def test_prefilter_3d_parity():
    grid = ((0.0, 1.0, 4), (0.0, 2.0, 5), (-1.0, 1.0, 6))
    vals = np.random.RandomState(42).randn(4, 5, 6)
    coeffs_ref = prefilter_cubic(grid, vals)
    coeffs_jax = prefilter_cubic_jax(grid, vals)
    assert np.allclose(coeffs_ref, np.array(coeffs_jax), atol=1e-13)


def test_eval_1d_linear_extrap_parity():
    grid = ((0.0, 1.0, 6),)
    vals = np.array([1.0, 2.0, 1.5, 3.0, 2.5, 3.5])
    coeffs = prefilter_cubic(grid, vals)

    # Interior, boundary, and outside points
    pts = np.array([[-0.5], [-0.01], [0.0], [0.2], [0.5], [0.99], [1.0], [1.05], [1.8]])
    out_ref = eval_cubic(grid, coeffs, pts)
    out_jax = eval_cubic_jax(grid, coeffs, pts, extrap_mode="linear")

    assert np.allclose(out_ref, np.array(out_jax), atol=1e-13)


def test_eval_2d_linear_extrap_parity():
    grid = ((0.0, 1.0, 5), (0.0, 2.0, 6))
    vals = np.random.RandomState(42).randn(5, 6)
    coeffs = prefilter_cubic(grid, vals)

    pts = np.array(
        [
            [0.5, 1.0],  # interior
            [0.0, 0.0],  # lower boundary
            [1.0, 2.0],  # upper boundary
            [-0.2, 0.5],  # left extrap on dim 0
            [1.4, 1.5],  # right extrap on dim 0
            [0.5, -0.4],  # left extrap on dim 1
            [0.5, 2.8],  # right extrap on dim 1
            [-0.5, -0.6],  # left extrap on both
            [1.6, 2.9],  # right extrap on both
        ]
    )

    out_ref = eval_cubic(grid, coeffs, pts)
    out_jax = eval_cubic_jax(grid, coeffs, pts, extrap_mode="linear")
    assert np.allclose(out_ref, np.array(out_jax), atol=1e-13)


def test_eval_3d_linear_extrap_parity():
    grid = ((0.0, 1.0, 4), (0.0, 2.0, 5), (-1.0, 1.0, 6))
    vals = np.random.RandomState(42).randn(4, 5, 6)
    coeffs = prefilter_cubic(grid, vals)

    pts = np.array(
        [
            [0.5, 1.0, 0.0],  # interior
            [-0.3, 0.5, 0.2],  # extrap x
            [0.4, 2.5, -0.5],  # extrap y
            [0.5, 1.0, -1.8],  # extrap z left
            [1.5, 3.0, 2.0],  # extrap all right
            [-1.0, -1.0, -3.0],  # extrap all left
        ]
    )

    out_ref = eval_cubic(grid, coeffs, pts)
    out_jax = eval_cubic_jax(grid, coeffs, pts, extrap_mode="linear")
    assert np.allclose(out_ref, np.array(out_jax), atol=1e-13)


def test_vector_valued_evaluation():
    grid = ((0.0, 1.0, 4), (0.0, 2.0, 5))
    vals = np.random.RandomState(42).randn(4, 5, 3)
    coeffs = prefilter_cubic(grid, vals)

    pts = np.array([[0.3, 0.7], [-0.2, 0.5], [1.3, 2.5]])
    out_ref = eval_cubic(grid, coeffs, pts)
    out_jax = eval_cubic_jax(grid, coeffs, pts)

    assert out_jax.shape == (3, 3)
    assert np.allclose(out_ref, np.array(out_jax), atol=1e-13)


def test_autodiff_gradient():
    a = jnp.array([0.0, -1.0])
    b = jnp.array([1.0, 1.0])
    n = jnp.array([5, 6])
    grid = ((0.0, 1.0, 5), (-1.0, 1.0, 6))
    vals = np.random.RandomState(42).randn(5, 6)
    coeffs = jnp.array(prefilter_cubic(grid, vals))

    f = lambda x: eval_cubic_point_jax(coeffs, x, a, b, n)
    grad_f = jax.grad(f)

    # Test interior and extrapolated points
    test_points = [
        jnp.array([0.4, 0.2]),
        jnp.array([-0.3, 0.5]),
        jnp.array([1.4, -1.5]),
    ]
    eps = 1e-7
    for pt in test_points:
        g_ad = grad_f(pt)
        # Finite difference check
        g_num0 = (f(pt + jnp.array([eps, 0.0])) - f(pt - jnp.array([eps, 0.0]))) / (
            2 * eps
        )
        g_num1 = (f(pt + jnp.array([0.0, eps])) - f(pt - jnp.array([0.0, eps]))) / (
            2 * eps
        )
        g_num = jnp.array([g_num0, g_num1])
        assert np.allclose(g_ad, g_num, atol=1e-6)


def test_approximator_backend_parity():
    grid = UniformCartesianGrid(min=[0.0, -1.0], max=[1.0, 1.0], n=[5, 6])
    vals = np.random.RandomState(42).randn(grid.n_nodes, 2)

    app_numba = CubicSplineApproximator(grid, backend="numba")
    app_jax = CubicSplineApproximator(grid, backend="jax")

    c_numba = app_numba.fit(vals)
    c_jax = app_jax.fit(vals)
    assert np.allclose(c_numba, np.array(c_jax), atol=1e-13)

    query_pts = np.array([[0.3, 0.2], [-0.5, 0.5], [1.5, 1.5]])
    res_numba = app_numba.evaluate_coords(c_numba, query_pts)
    res_jax = app_jax.evaluate_coords(c_jax, query_pts)
    assert np.allclose(res_numba, np.array(res_jax), atol=1e-13)


def test_eval_4d_parity():
    grid = ((0.0, 1.0, 3), (0.0, 2.0, 4), (-1.0, 1.0, 3), (1.0, 5.0, 4))
    vals = np.random.RandomState(42).randn(3, 4, 3, 4)
    coeffs = prefilter_cubic(grid, vals)

    pts = np.array(
        [
            [0.5, 1.0, 0.0, 3.0],
            [-0.2, 0.5, 0.2, 2.0],
            [0.4, 2.5, -0.5, 5.5],
            [-0.5, -0.8, 1.2, 0.5],
        ]
    )

    out_ref = eval_cubic(grid, coeffs, pts)
    out_jax = eval_cubic_jax(grid, coeffs, pts, extrap_mode="linear")
    assert np.allclose(out_ref, np.array(out_jax), atol=1e-13)


def test_eval_nearest_extrapolation():
    grid = ((0.0, 1.0, 4), (0.0, 2.0, 5))
    vals = np.random.RandomState(42).randn(4, 5)
    coeffs = prefilter_cubic(grid, vals)

    # Outside points: with nearest extrapolation, evaluating at [-0.5, 2.5] should equal evaluating at clamped point [0.0, 2.0]
    pts_outside = np.array([[-0.5, 2.5], [1.5, -0.5]])
    pts_clamped = np.array([[0.0, 2.0], [1.0, 0.0]])

    out_outside = eval_cubic_jax(grid, coeffs, pts_outside, extrap_mode="nearest")
    out_clamped = eval_cubic_jax(grid, coeffs, pts_clamped, extrap_mode="nearest")
    assert np.allclose(out_outside, out_clamped, atol=1e-13)


def test_grad_wrt_coeffs():
    grid = ((0.0, 1.0, 4), (0.0, 2.0, 5))
    a = jnp.array([0.0, 0.0])
    b = jnp.array([1.0, 2.0])
    n = jnp.array([4, 5])
    coeffs = jnp.array(np.random.RandomState(42).randn(6, 7))
    pt = jnp.array([0.3, 0.7])

    loss = lambda c: eval_cubic_point_jax(c, pt, a, b, n)
    grad_loss = jax.grad(loss)(coeffs)

    # Gradient w.r.t linear coefficients should match finite differences
    eps = 1e-7
    diff_patch = np.zeros_like(coeffs)
    for i in range(coeffs.shape[0]):
        for j in range(coeffs.shape[1]):
            delta_c = jnp.zeros_like(coeffs).at[i, j].set(eps)
            diff_patch[i, j] = (loss(coeffs + delta_c) - loss(coeffs - delta_c)) / (
                2 * eps
            )

    assert np.allclose(grad_loss, diff_patch, atol=1e-7)


def test_jit_compilation():
    grid = ((0.0, 1.0, 5), (0.0, 2.0, 6))
    vals = np.random.RandomState(42).randn(5, 6)
    coeffs = prefilter_cubic_jax(grid, vals)

    @jax.jit
    def evaluate(pts):
        return eval_cubic_jax(grid, coeffs, pts)

    pts = jnp.array([[0.5, 1.0], [-0.2, 0.5], [1.4, 2.5]])
    out1 = evaluate(pts)
    out2 = evaluate(pts)
    assert np.allclose(out1, out2)


def test_eval_linear_1d_parity():
    from interpolation.splines import eval_linear

    grid = ((0.0, 1.0, 6),)
    vals = np.array([1.0, 2.0, 1.5, 3.0, 2.5, 3.5])
    pts = np.array([[-0.5], [-0.01], [0.0], [0.2], [0.5], [0.99], [1.0], [1.05], [1.8]])

    out_ref = eval_linear(grid, vals, pts)
    out_jax = eval_linear_jax(grid, vals, pts)
    assert np.allclose(out_ref, np.array(out_jax), atol=1e-13)


def test_eval_linear_2d_parity():
    from interpolation.splines import eval_linear

    grid = ((0.0, 1.0, 5), (0.0, 2.0, 6))
    vals = np.random.RandomState(42).randn(5, 6)
    pts = np.array(
        [
            [0.5, 1.0],
            [0.0, 0.0],
            [1.0, 2.0],
            [-0.2, 0.5],
            [1.4, 1.5],
            [0.5, -0.4],
            [0.5, 2.8],
            [-0.5, -0.6],
            [1.6, 2.9],
        ]
    )

    out_ref = eval_linear(grid, vals, pts)
    out_jax = eval_linear_jax(grid, vals, pts)
    assert np.allclose(out_ref, np.array(out_jax), atol=1e-13)


def test_eval_linear_3d_parity():
    from interpolation.splines import eval_linear

    grid = ((0.0, 1.0, 4), (0.0, 2.0, 5), (-1.0, 1.0, 6))
    vals = np.random.RandomState(42).randn(4, 5, 6)
    pts = np.array(
        [
            [0.5, 1.0, 0.0],
            [-0.3, 0.5, 0.2],
            [0.4, 2.5, -0.5],
            [0.5, 1.0, -1.8],
            [1.5, 3.0, 2.0],
        ]
    )

    out_ref = eval_linear(grid, vals, pts)
    out_jax = eval_linear_jax(grid, vals, pts)
    assert np.allclose(out_ref, np.array(out_jax), atol=1e-13)


def test_eval_linear_vector_valued():
    from interpolation.splines import eval_linear

    grid = ((0.0, 1.0, 5), (0.0, 2.0, 6))
    vals = np.random.RandomState(42).randn(5, 6, 3)
    pts = np.array([[0.3, 0.7], [-0.2, 0.5], [1.3, 2.5]])

    out_ref = eval_linear(grid, vals, pts)
    out_jax = eval_linear_jax(grid, vals, pts)
    assert out_jax.shape == (3, 3)
    assert np.allclose(out_ref, np.array(out_jax), atol=1e-13)


def test_linear_approximator_backend_parity():
    from dolo.numeric.approximators import LinearApproximator

    grid = UniformCartesianGrid(min=[0.0, -1.0], max=[1.0, 1.0], n=[5, 6])
    vals = np.random.RandomState(42).randn(grid.n_nodes, 2)

    app_numba = LinearApproximator(grid, backend="numba")
    app_jax = LinearApproximator(grid, backend="jax")

    c_numba = app_numba.fit(vals)
    c_jax = app_jax.fit(vals)

    query_pts = np.array([[0.3, 0.2], [-0.5, 0.5], [1.5, 1.5]])
    res_numba = app_numba.evaluate_coords(c_numba, query_pts)
    res_jax = app_jax.evaluate_coords(c_jax, query_pts)
    assert np.allclose(res_numba, np.array(res_jax), atol=1e-13)
