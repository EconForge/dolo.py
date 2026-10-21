import pytest
import dolo


@pytest.fixture(scope="module")
def rbc_model():
    return dolo.yaml_import("examples/models/rbc.yaml")


def test_standard_time_iteration(rbc_model):
    sol = dolo.time_iteration(rbc_model, verbose=False, maxit=150)
    assert sol.converged
    assert sol.iterations < 150
    assert sol.dr is not None
    assert sol.error_step < 1e-5


def test_accelerated_time_iteration_neumann(rbc_model):
    sol = dolo.time_iteration(rbc_model, accelerate=True, verbose=False)
    assert sol.converged
    assert sol.iterations < 25  # Neumann acceleration should converge in ~15 iterations
    assert sol.error_step < 1e-5


def test_accelerated_time_iteration_gmres(rbc_model):
    sol = dolo.time_iteration(
        rbc_model, accelerate=dict(linsolve="gmres", wait=3), verbose=False
    )
    assert sol.converged
    assert sol.iterations < 15  # GMRES acceleration converges in ~8 iterations
    assert sol.error_step < 1e-5


def test_time_iteration_inner_options(rbc_model):
    sol = dolo.time_iteration(
        rbc_model,
        inner=dict(maxit=5, max_backsteps=5),
        maxit=50,
        verbose=False,
    )
    assert sol.dr is not None


def test_newton_gmres_backtracking(rbc_model):
    sol = dolo.newton(
        rbc_model, linsolve="gmres", linesearch="backtracking", verbose=False
    )
    assert sol.converged
    assert sol.iterations < 15
    assert sol.error_res < 1e-7
    assert sol.error_step < 1e-5


def test_newton_neumann_golden_section(rbc_model):
    sol = dolo.newton(
        rbc_model, linsolve="neumann", linesearch="golden_section", verbose=False
    )
    assert sol.converged
    assert sol.iterations < 20
    assert sol.error_res < 1e-7


def test_newton_no_linesearch(rbc_model):
    # Pure Newton without line search works reliably when warm-started near the solution
    sol_init = dolo.time_iteration(rbc_model, maxit=5, verbose=False)
    sol = dolo.newton(rbc_model, dr0=sol_init.dr, linesearch=False, verbose=False)
    assert sol.converged
    assert sol.iterations < 10


def test_unified_result_properties(rbc_model):
    sol_ti = dolo.time_iteration(rbc_model, verbose=False)
    assert isinstance(sol_ti.converged, bool)
    assert isinstance(sol_ti.iterations, int)
    assert sol_ti.error_step is not None

    sol_newton = dolo.newton(rbc_model, linesearch="backtracking", verbose=False)
    assert isinstance(sol_newton.converged, bool)
    assert isinstance(sol_newton.iterations, int)
    assert sol_newton.error_res is not None
    assert sol_newton.error_step is not None
    # Backward compatible fields
    assert sol_newton.x_converged == sol_newton.converged
    assert sol_newton.N == sol_newton.iterations
    assert sol_newton.f_x == sol_newton.error_res
    assert sol_newton.d_x == sol_newton.error_step


def test_improved_time_iteration_backward_compatibility(rbc_model):
    sol = dolo.improved_time_iteration(rbc_model, verbose=False)
    assert sol.x_converged
    assert sol.f_x < 1e-7
    assert sol.dr is not None


def test_newton_engines_parity(rbc_model):
    import numpy as np

    sol_jax = dolo.newton(
        rbc_model, engine="jax", linesearch="backtracking", verbose=False
    )
    assert sol_jax.converged
    assert sol_jax.error_res < 1e-7

    sol_vec = dolo.newton(
        rbc_model, engine="vectorized", linesearch="backtracking", verbose=False
    )
    assert sol_vec.converged

    s_test = rbc_model.discretize()[0]["endo"].nodes
    np.testing.assert_allclose(sol_jax.dr(s_test), sol_vec.dr(s_test), atol=1e-5)


def test_newton_gpu_if_available(rbc_model):
    import jax

    gpus = jax.devices("gpu")
    if not gpus:
        pytest.skip("No GPU available.")

    sol_gpu = dolo.newton(
        rbc_model, engine="gpu", linesearch="backtracking", verbose=False
    )
    assert sol_gpu.converged
    assert sol_gpu.error_res < 1e-7


def test_accelerated_time_iteration_jax(rbc_model):
    sol_jax = dolo.time_iteration(
        rbc_model, accelerate=True, engine="jax", verbose=False
    )
    assert sol_jax.converged
    assert sol_jax.iterations < 25
    assert sol_jax.error_step < 1e-5


def test_accelerated_time_iteration_gpu_if_available(rbc_model):
    import jax

    gpus = jax.devices("gpu")
    if not gpus:
        pytest.skip("No GPU available.")

    sol_gpu = dolo.time_iteration(
        rbc_model,
        accelerate=dict(linsolve="gmres", wait=3),
        engine="gpu",
        verbose=False,
    )
    assert sol_gpu.converged
    assert sol_gpu.iterations < 15
    assert sol_gpu.error_step < 1e-5
