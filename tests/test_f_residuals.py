import pytest
import numpy as np
import dolo
from dolo import F, dF_1, F_and_dF_1


@pytest.fixture(scope="module")
def rbc_model():
    return dolo.yaml_import("examples/models/rbc.yaml")


@pytest.fixture(scope="module")
def discretized_rbc(rbc_model):
    return dolo.numeric.DiscretizedModel(rbc_model, grid_options=dict(n=[15, 15]))


def test_f_residual_signatures(discretized_rbc):
    dmodel = discretized_rbc
    x0 = dmodel.initial_guess()

    # 1. F(model, x0, x0)
    res_two_args = F(dmodel, x0, x0)
    assert res_two_args.shape == x0.shape

    # 2. F(model, x0) where dr defaults to x0
    res_one_arg = F(dmodel, x0)
    assert np.allclose(res_two_args.data, res_one_arg.data)

    # 3. Model passed directly as non-discretized model
    res_from_model = F(discretized_rbc.model, x0, x0, grid_options=dict(n=[15, 15]))
    assert np.allclose(res_two_args.data, res_from_model.data)


def test_f_engine_parity(discretized_rbc):
    dmodel = discretized_rbc
    x0 = dmodel.initial_guess()

    res_jax = F(dmodel, x0, x0, engine="jax")
    res_vec = F(dmodel, x0, x0, engine="vectorized")
    res_auto = F(dmodel, x0, x0, engine="auto")

    # JAX and Vectorized should match to near machine precision
    np.testing.assert_allclose(res_jax.data, res_vec.data, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(res_jax.data, res_auto.data, rtol=1e-12, atol=1e-12)


def test_df_1_and_joint_evaluation(discretized_rbc):
    dmodel = discretized_rbc
    x0 = dmodel.initial_guess()

    # 1. dF_1 analytical autodiff via JAX
    jac_jax = dF_1(dmodel, x0, x0, engine="jax")
    assert jac_jax.shape == (15 * 15, 2, 2)

    # 2. dF_1 finite-difference via vectorized
    jac_vec = dF_1(dmodel, x0, x0, engine="vectorized")
    assert jac_vec.shape == (15 * 15, 2, 2)

    # Autodiff and FD should match to ~1e-5
    np.testing.assert_allclose(np.asarray(jac_jax), jac_vec, rtol=1e-4, atol=1e-4)

    # 3. Joint evaluation F_and_dF_1
    res_joint, jac_joint = F_and_dF_1(dmodel, x0, x0, engine="jax")
    res_ref = F(dmodel, x0, x0, engine="jax")
    np.testing.assert_allclose(res_joint.data, res_ref.data)
    np.testing.assert_allclose(np.asarray(jac_joint), np.asarray(jac_jax))


def test_f_gpu_if_available(discretized_rbc):
    import jax

    gpus = jax.devices("gpu")
    if not gpus:
        pytest.skip("No CUDA GPU available on system")

    dmodel = discretized_rbc
    x0 = dmodel.initial_guess()

    res_gpu = F(dmodel, x0, x0, engine="gpu")
    res_cpu = F(dmodel, x0, x0, engine="cpu")
    np.testing.assert_allclose(res_gpu.data, res_cpu.data, rtol=1e-12, atol=1e-12)

    jac_gpu = dF_1(dmodel, x0, x0, engine="gpu")
    jac_cpu = dF_1(dmodel, x0, x0, engine="cpu")
    np.testing.assert_allclose(
        np.asarray(jac_gpu), np.asarray(jac_cpu), rtol=1e-12, atol=1e-12
    )
