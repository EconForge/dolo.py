import pytest
import numpy as np
import dolo


@pytest.fixture(scope="module")
def rbc_model():
    return dolo.yaml_import("examples/models/rbc.yaml")


def test_time_iteration_engines_parity(rbc_model):
    # Solve with vectorized
    sol_vec = dolo.time_iteration(
        rbc_model, engine="vectorized", maxit=25, verbose=False
    )

    # Solve with jax
    sol_jax = dolo.time_iteration(rbc_model, engine="jax", maxit=25, verbose=False)

    # Both should have converged or reached the same stage
    assert sol_vec.iterations == sol_jax.iterations

    # Check decision rules evaluate to identical values at steady state
    s_ss = rbc_model.calibration["states"]
    val_vec = sol_vec.dr(0, s_ss)
    val_jax = sol_jax.dr(0, s_ss)
    np.testing.assert_allclose(val_jax, val_vec, rtol=1e-6, atol=1e-6)


def test_time_iteration_gpu(rbc_model):
    import jax

    gpus = jax.devices("gpu")
    if not gpus:
        pytest.skip("No CUDA GPU available")

    sol_gpu = dolo.time_iteration(rbc_model, engine="gpu", maxit=15, verbose=False)
    sol_jax = dolo.time_iteration(rbc_model, engine="jax", maxit=15, verbose=False)

    s_ss = rbc_model.calibration["states"]
    np.testing.assert_allclose(
        sol_gpu.dr(0, s_ss), sol_jax.dr(0, s_ss), rtol=1e-10, atol=1e-10
    )


def test_time_iteration_backend_alias(rbc_model):
    sol_b = dolo.time_iteration(rbc_model, backend="jax", maxit=5, verbose=False)
    sol_e = dolo.time_iteration(rbc_model, engine="jax", maxit=5, verbose=False)
    assert sol_b.iterations == sol_e.iterations
