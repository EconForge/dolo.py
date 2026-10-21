"""JAX solvers keep their data on the device; residual wrappers take the decision rule as values."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import dolo
from dolo.numeric.decision_rule import DecisionRule
from dolo.numeric.residuals import F, F_and_dF_1, _get_or_compile_jax_residual


@pytest.fixture(scope="module")
def dmodel():
    model = dolo.yaml_import("examples/models/rbc_mc.yaml")
    return dolo.numeric.DiscretizedModel(model, grid_options=dict(n=[20]))


def _two_policies(dmodel):
    x = dmodel.initial_guess()
    y = x.data * (1 + 0.02 * np.random.default_rng(1).standard_normal(x.data.shape))
    return x.data, y


def test_dr_keyword_is_used(dmodel):
    x, y = _two_policies(dmodel)
    by_position = np.asarray(F(dmodel, x, y, engine="jax"))
    by_keyword = np.asarray(F(dmodel, x, dr=y, engine="jax"))
    own = np.asarray(F(dmodel, x, engine="jax"))
    np.testing.assert_allclose(by_keyword, by_position, rtol=0, atol=0)
    assert np.abs(by_keyword - own).max() > 1e-6  # the decision rule matters
    r_k, J_k = F_and_dF_1(dmodel, x, dr=y, engine="jax")
    r_p, J_p = F_and_dF_1(dmodel, x, y, engine="jax")
    np.testing.assert_allclose(np.asarray(J_k), np.asarray(J_p), rtol=0, atol=0)


def test_decision_rule_object_rejected_by_jax(dmodel):
    x, _ = _two_policies(dmodel)
    dr = DecisionRule(dmodel.grid["exo"], dmodel.grid["endo"], dprocess=dmodel.dprocess)
    dr.set_values(x)
    with pytest.raises(TypeError, match="values on the grid"):
        F(dmodel, x, dr=dr, engine="jax")


def test_prefilter_device_and_host(dmodel):
    cr = _get_or_compile_jax_residual(dmodel)
    x, _ = _two_policies(dmodel)
    x_dev = jax.device_put(jnp.asarray(x))
    c_host, c_dev = cr.prefilter(x), cr.prefilter(x_dev)
    np.testing.assert_allclose(np.asarray(c_dev), np.asarray(c_host), rtol=0, atol=0)
    with jax.transfer_guard_device_to_host("disallow"):
        cr.prefilter(x_dev).block_until_ready()


@pytest.mark.parametrize("accelerate", [False, dict(linsolve="gmres", wait=3)], ids=["plain", "accelerated"])
def test_time_iteration_jax_matches_vectorized(dmodel, accelerate):
    sol_v = dolo.time_iteration(dmodel, engine="vectorized", accelerate=accelerate, verbose=False)
    sol_j = dolo.time_iteration(dmodel, engine="jax", accelerate=accelerate, verbose=False)
    assert sol_j.converged
    s = dmodel.endo_grid.nodes
    for i in range(dmodel.n_regimes):
        np.testing.assert_allclose(sol_j.dr(i, s), sol_v.dr(i, s), rtol=1e-5, atol=1e-6)


def test_newton_jax_matches_vectorized(dmodel):
    sol_v = dolo.newton(dmodel, engine="vectorized", verbose=False)
    sol_j = dolo.newton(dmodel, engine="jax", verbose=False)
    assert sol_j.converged
    s = dmodel.endo_grid.nodes
    for i in range(dmodel.n_regimes):
        np.testing.assert_allclose(sol_j.dr(i, s), sol_v.dr(i, s), rtol=1e-5, atol=1e-6)
