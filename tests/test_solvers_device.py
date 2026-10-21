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


# rbc_mc has complementarity conditions, which the device loops do not handle (they go through
# the previous loops): without them, the JAX engine runs the device loops


@pytest.mark.parametrize(
    "accelerate",
    [False, dict(linsolve="gmres", wait=3), dict(linsolve="neumann", wait=3)],
    ids=["plain", "accelerated_gmres", "accelerated_neumann"],
)
def test_time_iteration_jax_matches_vectorized(dmodel, accelerate):
    kw = dict(accelerate=accelerate, ignore_constraints=True, verbose=False)
    sol_v = dolo.time_iteration(dmodel, engine="vectorized", **kw)
    sol_j = dolo.time_iteration(dmodel, engine="jax", **kw)
    assert sol_j.converged and sol_v.converged
    s = dmodel.endo_grid.nodes
    for i in range(dmodel.n_regimes):
        np.testing.assert_allclose(sol_j.dr(i, s), sol_v.dr(i, s), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "options",
    [dict(), dict(linsolve="gmres", linesearch="backtracking"), dict(linesearch="golden_section")],
    ids=["defaults", "gmres_backtracking", "golden_section"],
)
def test_newton_jax_matches_vectorized(dmodel, options):
    kw = dict(ignore_constraints=True, verbose=False, **options)
    sol_v = dolo.newton(dmodel, engine="vectorized", **kw)
    sol_j = dolo.newton(dmodel, engine="jax", **kw)
    assert sol_j.converged and sol_v.converged
    s = dmodel.endo_grid.nodes
    for i in range(dmodel.n_regimes):
        np.testing.assert_allclose(sol_j.dr(i, s), sol_v.dr(i, s), rtol=1e-5, atol=1e-6)


def test_device_loops_are_used(dmodel, monkeypatch):
    """Without complementarities, the JAX engine runs the device loops."""
    import importlib

    ti = importlib.import_module("dolo.algos.time_iteration")
    nm = importlib.import_module("dolo.algos.newton")
    calls = []
    for mod, name in ((ti, "_time_iteration_device"), (nm, "_newton_device")):
        orig = getattr(mod, name)
        monkeypatch.setattr(
            mod, name, lambda *a, _o=orig, _n=name, **k: (calls.append(_n), _o(*a, **k))[1]
        )
    dolo.time_iteration(dmodel, engine="jax", ignore_constraints=True, maxit=3, verbose=False)
    dolo.newton(dmodel, engine="jax", ignore_constraints=True, maxit=2, verbose=False)
    assert calls == ["_time_iteration_device", "_newton_device"]
