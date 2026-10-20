import pytest
import numpy as np
import jax
import jax.numpy as jnp

import dolo
from dolo.compiler.model_v2 import ModelV2, yaml_compose
from dolo.numeric.residual_jax import compile_residual, CompiledResidual
from dolo.algos.time_iteration import euler_residuals_tau
from dolo.numeric.optimize.newton import SerialDifferentiableFunction


@pytest.fixture(scope="module")
def rbc_model():
    return dolo.yaml_import_v2("examples/models/rbc.yaml")


@pytest.fixture(scope="module")
def cs_model():
    return dolo.yaml_import_v2("examples/models/consumption_savings_iid.yaml")


def test_compiled_residual_initialization(rbc_model):
    cr = compile_residual(rbc_model)
    assert isinstance(cr, CompiledResidual)
    assert cr.n_regimes == 1
    assert cr.grid_shape == (100, 100)


def test_compiled_residual_from_filepath():
    cr = compile_residual("examples/models/rbc.yaml", grid_options=dict(n=[10, 10]))
    assert isinstance(cr, CompiledResidual)
    assert cr.grid_shape == (10, 10)


def test_compiled_residual_diff_mode_rev(rbc_model):
    dmodel = dolo.numeric.DiscretizedModel(rbc_model, grid_options=dict(n=[10, 10]))
    cr_fwd = compile_residual(dmodel, diff_mode="fwd")
    cr_rev = compile_residual(dmodel, diff_mode="rev")

    controls_ga = dmodel.initial_guess()
    dF_fwd = cr_fwd.dF_1(controls_ga)
    dF_rev = cr_rev.dF_1(controls_ga)

    assert np.allclose(np.array(dF_fwd), np.array(dF_rev), atol=1e-14)


def test_residual_vmap_parity_rbc(rbc_model):
    dmodel = dolo.numeric.DiscretizedModel(rbc_model)
    cr = compile_residual(dmodel, interp_method="cubic")

    controls_ga = dmodel.initial_guess()
    dr = dmodel.decision_rule(controls_ga)

    # 1. Reference legacy/current vectorized residuals
    res_ref = euler_residuals_tau(dmodel, controls_ga, dr)[0]

    # 2. JAX vmapped residuals
    res_jax = cr.F(controls_ga)[0]

    # Numerical difference should be at machine precision
    max_err = np.max(np.abs(res_ref - np.array(res_jax)))
    assert max_err < 1e-13


def test_derivative_vmap_parity_rbc(rbc_model):
    # Use smaller grid for fast finite-difference reference test
    dmodel = dolo.numeric.DiscretizedModel(rbc_model, grid_options=dict(n=[15, 15]))
    cr = compile_residual(dmodel, interp_method="cubic", diff_mode="fwd")

    controls_ga = dmodel.initial_guess()
    dr = dmodel.decision_rule(controls_ga)
    n_x = len(dmodel.model.calibration["controls"])

    # Reference: SerialDifferentiableFunction finite differences
    fn_cur = (
        lambda x_val: euler_residuals_tau(dmodel, controls_ga.unravel(x_val), dr)
        .ravel()
        .reshape((-1, n_x))
    )
    dfn_cur = SerialDifferentiableFunction(fn_cur, epsilon=1e-7)
    _, dval_ref = dfn_cur(np.array(controls_ga[0]))

    # JAX vmapped derivative
    dval_jax = np.array(cr.dF_1(controls_ga))

    # Match within finite difference truncation error
    rel_err = np.max(np.abs(dval_ref - dval_jax) / (np.abs(dval_ref) + 1e-8))
    assert rel_err < 1e-4


def test_joint_residual_and_derivative_parity(rbc_model):
    dmodel = dolo.numeric.DiscretizedModel(rbc_model, grid_options=dict(n=[10, 10]))
    cr = compile_residual(dmodel)
    controls_ga = dmodel.initial_guess()

    res_sep = cr.F(controls_ga)[0]
    dF_sep = cr.dF_1(controls_ga)

    res_joint, dF_joint = cr.F_and_dF_1(controls_ga)

    assert np.allclose(np.array(res_sep), np.array(res_joint[0]), atol=1e-14)
    assert np.allclose(np.array(dF_sep), np.array(dF_joint), atol=1e-14)


def test_residual_consumption_savings(cs_model):
    dmodel = dolo.numeric.DiscretizedModel(cs_model)
    cr = compile_residual(dmodel, interp_method="cubic")

    controls_ga = dmodel.initial_guess()
    dr = dmodel.decision_rule(controls_ga)

    res_ref = euler_residuals_tau(dmodel, controls_ga, dr)[0]
    res_jax = cr.F(controls_ga)[0]

    max_err = np.max(np.abs(res_ref - np.array(res_jax)))
    assert max_err < 1e-13

    # Derivative
    dF_jax = cr.dF_1(controls_ga)
    assert dF_jax.shape == (dmodel.endo_nodes.shape[0], 1, 1)
