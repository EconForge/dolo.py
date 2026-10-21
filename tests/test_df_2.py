import pytest
import numpy as np
import dolo
from dolo import dF_1, dF_2, PolicyLinearOperator
from dolo.numeric.garray import GArray


@pytest.fixture(scope="module")
def rbc_model():
    return dolo.yaml_import("examples/models/rbc.yaml")


@pytest.fixture(scope="module")
def discretized_rbc(rbc_model):
    return dolo.numeric.DiscretizedModel(rbc_model, grid_options=dict(n=[10, 10]))


def test_df_2_point_level(discretized_rbc):
    dmodel = discretized_rbc
    x0 = dmodel.initial_guess()
    s0 = dmodel.endo_nodes[0]
    c0 = x0[0, 0]

    # JAX point evaluation
    branches_jax = dF_2(dmodel, s0, c0, x0, engine="jax")
    assert len(branches_jax) == dmodel.dprocess.n_inodes(0)
    for fx, s in branches_jax:
        assert fx.shape == (2, 2)
        assert s.shape == (2,)

    # Vectorized point evaluation
    branches_vec = dF_2(dmodel, s0, c0, x0, engine="vectorized")
    assert len(branches_vec) == len(branches_jax)

    # Parity check between JAX and vectorized point branches
    for (fx_j, s_j), (fx_v, s_v) in zip(branches_jax, branches_vec):
        np.testing.assert_allclose(s_j, s_v[-2:])
        np.testing.assert_allclose(fx_j, fx_v, rtol=1e-4, atol=1e-4)


def test_df_2_grid_level_operator(discretized_rbc):
    dmodel = discretized_rbc
    x0 = dmodel.initial_guess()

    L_jax = dF_2(dmodel, x0, x0, engine="jax")
    L_vec = dF_2(dmodel, x0, x0, engine="vectorized")
    L_auto = dF_2(dmodel, x0, x0, engine="auto")

    assert isinstance(L_jax, PolicyLinearOperator)
    assert isinstance(L_vec, PolicyLinearOperator)
    assert L_jax.shape == (200, 200)
    assert L_vec.shape == (200, 200)

    # Test operator application parity on random vector v
    np.random.seed(42)
    v = np.random.randn(100, 2)
    out_jax = L_jax @ v
    out_vec = L_vec @ v
    out_auto = L_auto @ v

    assert out_jax.shape == (100, 2)
    assert out_vec.shape == (100, 2)
    np.testing.assert_allclose(out_jax, out_vec, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_jax, out_auto, rtol=1e-12, atol=1e-12)


def test_df_2_operator_inputs(discretized_rbc):
    dmodel = discretized_rbc
    x0 = dmodel.initial_guess()
    L = dF_2(dmodel, x0, x0, engine="jax")

    # 1. 2D array input
    v_2d = np.ones((100, 2))
    out_2d = L @ v_2d
    assert out_2d.shape == (100, 2)

    # 2. 1D flat array input
    v_1d = np.ones(200)
    out_1d = L @ v_1d
    assert out_1d.shape == (200,)
    np.testing.assert_allclose(out_1d, out_2d.ravel())

    # 3. GArray input
    v_ga = GArray.zeros_like(x0)
    v_ga[0] = v_2d
    out_ga = L @ v_ga
    assert isinstance(out_ga, GArray)
    np.testing.assert_allclose(out_ga[0], out_2d)

    # 4. Scaling
    L_scaled = 2.5 * L
    out_scaled = L_scaled @ v_2d
    np.testing.assert_allclose(out_scaled, 2.5 * out_2d)


def test_df_2_as_matrix(discretized_rbc):
    dmodel = discretized_rbc
    x0 = dmodel.initial_guess()
    L = dF_2(dmodel, x0, x0, engine="jax")

    mat = L.as_matrix()
    assert mat.shape == (200, 200)

    np.random.seed(123)
    v = np.random.randn(200)
    out_op = L @ v
    out_mat = mat @ v
    np.testing.assert_allclose(out_op, out_mat, rtol=1e-10, atol=1e-10)


def test_df_2_preconditioned_and_solve(discretized_rbc):
    dmodel = discretized_rbc
    x0 = dmodel.initial_guess()

    L = dF_2(dmodel, x0, x0, engine="jax")
    J = dF_1(dmodel, x0, x0, engine="jax")

    # Preconditioned T' = - J^{-1} L
    L_prec = (-1.0) * L.preconditioned(J)
    assert L_prec.shape == (200, 200)

    # Solve (I - L_prec) x = b
    np.random.seed(7)
    b = np.random.randn(100, 2) * 0.05
    sol_neumann = L_prec.solve(b, linsolve="neumann", maxit=60, tol=1e-6)
    sol_gmres = L_prec.solve(b, linsolve="gmres", maxit=30, tol=1e-6)

    # Both linear solvers satisfy (I - L_prec) x = b
    res_gmres = b.ravel() - (sol_gmres.ravel() - (L_prec @ sol_gmres).ravel())
    np.testing.assert_allclose(res_gmres, 0.0, atol=1e-5)

    res_neumann = b.ravel() - (sol_neumann.ravel() - (L_prec @ sol_neumann).ravel())
    np.testing.assert_allclose(res_neumann, 0.0, atol=1e-2)

    # GMRES and Neumann agree within truncation error
    np.testing.assert_allclose(sol_neumann, sol_gmres, atol=1e-2)


def test_df_2_gpu_if_available(discretized_rbc):
    import jax

    gpus = jax.devices("gpu")
    if not gpus:
        pytest.skip("No CUDA GPU available")

    dmodel = discretized_rbc
    x0 = dmodel.initial_guess()

    L_gpu = dF_2(dmodel, x0, x0, engine="gpu")
    L_cpu = dF_2(dmodel, x0, x0, engine="cpu")

    np.random.seed(42)
    v = np.random.randn(100, 2)
    out_gpu = L_gpu @ v
    out_cpu = L_cpu @ v

    np.testing.assert_allclose(out_gpu, out_cpu, rtol=1e-12, atol=1e-12)


@pytest.fixture(scope="module")
def discretized_rbc_mc():
    model = dolo.yaml_import("examples/models/rbc_mc.yaml")
    return dolo.numeric.DiscretizedModel(model, grid_options=dict(n=[20]))


def _reference_apply(dmodel, F_x, S, v):
    """L.v from the dense blocks: out[i] = sum_k F_x[i, k] v_k(S[i, k]), v_k cubic spline of v[k]."""
    from dolo.numeric.interpolation_jax import eval_cubic_jax, prefilter_cubic_jax

    grid = dmodel.endo_grid
    gs = tuple(int(n) for n in grid.n)
    F_x, S = np.asarray(F_x), np.asarray(S)
    out = np.zeros_like(v)
    for i in range(F_x.shape[0]):
        for k in range(F_x.shape[1]):
            coeffs = prefilter_cubic_jax(grid, v[k].reshape(gs + (-1,)))
            v_S = np.asarray(eval_cubic_jax(grid, coeffs, S[i, k]))
            out[i] += np.einsum("nij,nj->ni", F_x[i, k], v_S)
    return out


def test_df_2_multi_regime_operator(discretized_rbc_mc):
    from dolo.numeric.residuals import _get_or_compile_jax_residual

    dmodel = discretized_rbc_mc
    assert dmodel.n_regimes > 1
    x0 = dmodel.initial_guess()
    n_m, N, n_x = x0.data.shape

    L_jax = dF_2(dmodel, x0, x0, engine="jax")
    L_vec = dF_2(dmodel, x0, x0, engine="vectorized")
    assert L_jax.shape == (n_m * N * n_x,) * 2

    cr = _get_or_compile_jax_residual(dmodel, interp_method="cubic")
    F_x, S = cr.dF_2(x0.data, cr.prefilter(x0))
    np.testing.assert_allclose(np.asarray(L_jax.F_x), np.asarray(F_x), rtol=0, atol=0)

    np.random.seed(3)
    v = np.random.randn(n_m, N, n_x)
    out_jax = L_jax @ v
    assert out_jax.shape == (n_m, N, n_x)
    np.testing.assert_allclose(out_jax, _reference_apply(dmodel, F_x, S, v), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out_jax, L_vec @ v, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose((-2.0 * L_jax) @ v, -2.0 * out_jax, rtol=1e-14, atol=1e-14)


def test_df_2_multi_regime_preconditioned_and_solve(discretized_rbc_mc):
    dmodel = discretized_rbc_mc
    x0 = dmodel.initial_guess()
    n_m, N, n_x = x0.data.shape

    L = dF_2(dmodel, x0, x0, engine="jax")
    J = np.asarray(dF_1(dmodel, x0, x0, engine="jax"))
    L_prec = (-1.0) * L.preconditioned(J)

    # preconditioned blocks: -J^{-1} F_x
    F_prec = -np.linalg.solve(J[:, None], np.asarray(L.F_x))
    np.testing.assert_allclose(np.asarray(L_prec.F_x), F_prec, rtol=1e-10, atol=1e-12)

    np.random.seed(11)
    b = np.random.randn(n_m, N, n_x) * 0.05
    sol_gmres = L_prec.solve(b, linsolve="gmres", maxit=50, tol=1e-10)
    sol_neumann = L_prec.solve(b, linsolve="neumann", maxit=200, tol=1e-12)
    assert sol_gmres.shape == b.shape
    res_gmres = b - (sol_gmres - L_prec @ sol_gmres)
    assert np.abs(res_gmres).max() < 1e-8
    np.testing.assert_allclose(sol_neumann, sol_gmres, atol=1e-6)
