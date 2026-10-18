"""
Tests for Space hierarchy and Approximators in dolo.numeric.
"""

import numpy as np
import pytest

from dolo.compiler.model_v2 import ModelV2, yaml_compose
from dolo.numeric.garray import GArray, QP, SliceQP
from dolo.numeric.spaces import (
    Space,
    CartesianSpace,
    DiscreteSpace,
    GridSpace,
    ProductSpace,
    EmptySpace,
)
from dolo.numeric.grids import (
    UniformCartesianGrid,
    UnstructuredGrid,
    ProductGrid,
    SmolyakGrid,
    EmptyGrid,
)
from dolo.numeric.approximators import (
    Approximator,
    LinearApproximator,
    CubicSplineApproximator,
    SmolyakApproximator,
    ProductApproximator,
    get_approximator,
)
from dolo.numeric.decision_rule import DecisionRule
from dolo.numeric.discretized_model import DiscretizedModel

# ---------------------------------------------------------------------------
# 1. Space Hierarchy Tests
# ---------------------------------------------------------------------------


def test_cartesian_space():
    cs = CartesianSpace(min=[0.1, 1.0], max=[0.5, 5.0], variables=["z", "k"])
    assert cs.d == 2
    assert cs.variables == ("z", "k")
    assert np.allclose(cs.min, [0.1, 1.0])
    assert np.allclose(cs.max, [0.5, 5.0])

    # Dict-like compatibility
    assert np.allclose(cs["z"], [0.1, 0.5])
    assert np.allclose(cs["k"], [1.0, 5.0])
    assert cs.states == ("z", "k")

    # Membership
    assert [0.2, 2.0] in cs
    assert [0.05, 2.0] not in cs
    assert [0.2, 5.5] not in cs

    # QualifiedPoint construction
    qp = cs.qp([0.3, 3.0])
    assert np.allclose(qp.loc, [0.3, 3.0])
    assert np.allclose(qp.val, [0.3, 3.0])

    # Random sampling
    draw = cs.draw()
    assert draw.val.shape == (2,)
    assert draw.val in cs

    # Discretization
    grid = cs.discretize(n=[10, 20])
    assert isinstance(grid, UniformCartesianGrid)
    assert grid.nodes.shape == (200, 2)

    sg = cs.discretize(type="smolyak", mu=2)
    assert isinstance(sg, SmolyakGrid)


def test_discrete_space():
    pts = [[-0.02], [0.0], [0.02]]
    ds = DiscreteSpace(pts, variables=["z"])
    assert ds.d == 1
    assert ds.n_points == 3
    assert GridSpace is DiscreteSpace

    # Membership
    assert 0 in ds
    assert 2 in ds
    assert 3 not in ds
    assert [-0.02] in ds
    assert [0.05] not in ds

    # QP construction
    qp0 = ds.qp(0)
    assert qp0.loc == 0
    assert np.allclose(qp0.val, [-0.02])

    qp_coord = ds.qp([-0.02])
    assert qp_coord.loc == 0
    assert np.allclose(qp_coord.val, [-0.02])

    # Discretization
    ug = ds.discretize()
    assert isinstance(ug, UnstructuredGrid)
    assert ug.n_nodes == 3


def test_empty_space():
    es = EmptySpace()
    assert es.d == 0
    assert es.variables == ()
    assert "anything" in es
    qp = es.qp()
    assert qp.loc is None
    assert qp.val.shape == (0,)
    g = es.discretize()
    assert isinstance(g, EmptyGrid)


def test_product_space():
    exo = DiscreteSpace([[-0.01], [0.01]], variables=["z"])
    endo = CartesianSpace(min=[1.0], max=[5.0], variables=["k"])
    ps = ProductSpace([exo, endo], names=["exo", "endo"])
    assert ps.d == 2
    assert ps.variables == ("z", "k")
    assert ps.exo is exo
    assert ps.endo is endo
    assert ps["exo"] is exo
    assert ps["endo"] is endo

    # Multiplication operator
    ps2 = exo * endo
    assert isinstance(ps2, ProductSpace)
    assert len(ps2.spaces) == 2

    # QP construction with composite tuple loc
    qp = ps.qp((1, [2.5]))
    assert qp.loc == (1, [2.5])
    assert qp.index == 1
    assert np.allclose(qp.val, [0.01, 2.5])

    # Discretization
    pgrid = ps.discretize(endo={"n": 50})
    assert isinstance(pgrid, ProductGrid)
    assert pgrid.grids[0].n_nodes == 2
    assert pgrid.grids[1].nodes.shape[0] == 50


# ---------------------------------------------------------------------------
# 2. Approximator Tests
# ---------------------------------------------------------------------------


def test_linear_and_cubic_approximators():
    grid = UniformCartesianGrid(min=[0.0, 0.0], max=[1.0, 1.0], n=[11, 11])
    lin_app = LinearApproximator(grid)
    cub_app = CubicSplineApproximator(grid)

    # Function f(x, y) = 2*x + 3*y
    nodes = grid.nodes
    f_vals = 2.0 * nodes[:, 0] + 3.0 * nodes[:, 1]
    f_vals = f_vals[:, None]

    # Fit
    coeffs_lin = lin_app.fit(f_vals)
    coeffs_cub = cub_app.fit(f_vals)

    # Evaluate at test point
    test_pts = np.array([[0.25, 0.75], [0.5, 0.5]])
    expected = np.array([[2.0 * 0.25 + 3.0 * 0.75], [2.0 * 0.5 + 3.0 * 0.5]])

    eval_lin = lin_app.evaluate_coords(coeffs_lin, test_pts)
    eval_cub = cub_app.evaluate_coords(coeffs_cub, test_pts)

    assert np.allclose(eval_lin, expected, atol=1e-12)
    assert np.allclose(eval_cub, expected, atol=1e-12)

    # Test with QP
    qp = QP(loc=test_pts[0], val=test_pts[0])
    assert np.allclose(lin_app.evaluate(coeffs_lin, qp), expected[0])
    assert np.allclose(cub_app.evaluate(coeffs_cub, qp), expected[0])


def test_product_approximator():
    exo_grid = UnstructuredGrid([[-0.05], [0.05]])
    endo_grid = UniformCartesianGrid(min=[1.0], max=[5.0], n=[21])
    pgrid = ProductGrid(exo_grid, endo_grid, names=["exo", "endo"])

    prod_app = ProductApproximator(pgrid, interp_method="cubic")
    ga = GArray.zeros(pgrid, n_vars=1)
    # Fill regime 0: 2*k, regime 1: 3*k
    k_nodes = endo_grid.nodes[:, 0]
    ga[0, :, 0] = 2.0 * k_nodes
    ga[1, :, 0] = 3.0 * k_nodes

    coeffs = prod_app.fit(ga)

    # Evaluate via QP
    qp0 = QP(loc=(0, [2.5]), val=np.array([-0.05, 2.5]))
    qp1 = QP(loc=(1, [2.5]), val=np.array([0.05, 2.5]))
    assert np.isclose(prod_app.evaluate(coeffs, qp0)[0], 5.0)
    assert np.isclose(prod_app.evaluate(coeffs, qp1)[0], 7.5)

    # Evaluate via QualifiedBatch / SliceQP
    batch0 = SliceQP(loc=0, exo=np.array([-0.05]), endo=np.array([[2.5], [3.5]]))
    res0 = prod_app.evaluate_batch(coeffs, batch0)
    assert np.allclose(res0[:, 0], [5.0, 7.0])


# ---------------------------------------------------------------------------
# 3. DecisionRule fit() and Chaining
# ---------------------------------------------------------------------------


def test_decision_rule_fit_api():
    exo_grid = UnstructuredGrid([[-0.05], [0.05]])
    endo_grid = UniformCartesianGrid(min=[1.0], max=[5.0], n=[21])
    ga = GArray.zeros(ProductGrid(exo_grid, endo_grid), n_vars=1)
    k_nodes = endo_grid.nodes[:, 0]
    ga[0, :, 0] = 2.0 * k_nodes
    ga[1, :, 0] = 3.0 * k_nodes

    # Test chaining with fit()
    dr = DecisionRule(exo_grid, endo_grid, interp_method="cubic").fit(ga)
    assert hasattr(dr, "coefficients")

    # Verify fit() and set_values are aliases
    assert dr.set_values == dr.fit

    # Evaluate
    val0 = dr(0, np.array([[2.5]]))
    assert np.isclose(val0[0, 0], 5.0)

    qp0 = QP(loc=(0, [2.5]), val=np.array([-0.05, 2.5]))
    assert np.isclose(dr(qp0)[0], 5.0)


# ---------------------------------------------------------------------------
# 4. ModelV2 and DiscretizedModel Integration
# ---------------------------------------------------------------------------


def _load_v2(path: str) -> ModelV2:
    with open(path, "rt", encoding="utf-8") as f:
        txt = f.read()
    return ModelV2.from_yaml(yaml_compose(txt), filename=path)


def test_model_v2_domain_space():
    model = _load_v2("examples/models/rbc_iid.yaml")
    assert isinstance(model.domain, CartesianSpace)
    assert isinstance(model.domain, Space)
    assert model.domain.d == 2
    assert "k" in model.domain
    assert "z" in model.domain
    assert model.domain.min.shape == (2,)
    assert model.domain.max.shape == (2,)


def test_discretized_model_space_and_dr():
    model = _load_v2("examples/models/rbc_iid.yaml")
    dmodel = DiscretizedModel(model)

    # Check space
    assert hasattr(dmodel, "space")
    assert isinstance(dmodel.space, ProductSpace)
    assert isinstance(dmodel.space.endo, CartesianSpace)

    # Check decision rule factory and fit
    x0 = dmodel.initial_guess()
    dr = dmodel.decision_rule(x0)
    assert hasattr(dr, "coefficients")

    # Evaluate at steady-state states
    s0 = dmodel.endo_nodes[0:2]
    eval_dr = dr(0, s0)
    assert eval_dr.shape == (2, len(model.calibration["controls"]))


def test_unified_decision_rule_and_approximator_evaluation():
    """Verify that dr(...) and app.evaluate(...) support all calling styles."""
    import warnings

    # 1. Pure continuous model (empty exogenous)
    exo_empty = EmptyGrid()
    endo_cart = UniformCartesianGrid(min=[1.0], max=[5.0], n=[21])
    grid_iid = ProductGrid(exo_empty, endo_cart)
    ga_iid = GArray.zeros(grid_iid, n_vars=1)
    k_nodes = endo_cart.nodes[:, 0]
    ga_iid.data[0, :, 0] = 2.0 * k_nodes

    dr_iid = DecisionRule(exo_empty, endo_cart).fit(ga_iid)

    # Calling styles for pure continuous:
    # a) dr(s) 1D
    assert np.isclose(dr_iid(np.array([2.5]))[0], 5.0)
    # b) dr(s) 2D
    assert np.allclose(dr_iid(np.array([[2.5], [3.5]]))[:, 0], [5.0, 7.0])
    # c) dr(qp)
    qp_iid = QP(loc=None, val=np.array([2.5]))
    assert np.isclose(dr_iid(qp_iid)[0], 5.0)
    # d) dr(slice_qp)
    slice_iid = SliceQP(loc=None, exo=np.array([]), endo=np.array([[2.5], [3.5]]))
    assert np.allclose(dr_iid(slice_iid)[:, 0], [5.0, 7.0])

    # 2. Discrete exogenous (Markov chain)
    exo_disc = UnstructuredGrid([[-0.05], [0.05]])
    grid_mc = ProductGrid(exo_disc, endo_cart)
    ga_mc = GArray.zeros(grid_mc, n_vars=1)
    ga_mc.data[0, :, 0] = 2.0 * k_nodes
    ga_mc.data[1, :, 0] = 3.0 * k_nodes

    dr_mc = DecisionRule(exo_disc, endo_cart).fit(ga_mc)

    # Calling styles for discrete exogenous:
    # a) dr(i, s) 1D and 2D
    assert np.isclose(dr_mc(0, np.array([2.5]))[0], 5.0)
    assert np.isclose(dr_mc(1, np.array([2.5]))[0], 7.5)
    assert np.allclose(dr_mc(0, np.array([[2.5], [3.5]]))[:, 0], [5.0, 7.0])
    # b) dr(qp)
    qp_mc = QP(loc=(0, [2.5]), val=np.array([-0.05, 2.5]))
    assert np.isclose(dr_mc(qp_mc)[0], 5.0)
    # c) dr(slice_qp)
    slice_mc = SliceQP(loc=1, exo=np.array([0.05]), endo=np.array([[2.5], [3.5]]))
    assert np.allclose(dr_mc(slice_mc)[:, 0], [7.5, 10.5])

    # 3. Continuous exogenous (VAR1 / joint Cartesian)
    exo_cart = UniformCartesianGrid(min=[-0.1], max=[0.1], n=[11])
    grid_var = ProductGrid(exo_cart, endo_cart)
    ga_var = GArray.zeros(grid_var, n_vars=1)
    for i, m_val in enumerate(exo_cart.nodes[:, 0]):
        ga_var.data[i, :, 0] = m_val + 2.0 * k_nodes

    dr_var = DecisionRule(exo_cart, endo_cart).fit(ga_var)

    # Calling styles for continuous exogenous:
    # a) dr(m, s) 1D and 2D
    assert np.isclose(dr_var(np.array([0.0]), np.array([2.5]))[0], 5.0)
    m_pts = np.array([[0.0], [0.0]])
    s_pts = np.array([[2.5], [3.5]])
    assert np.allclose(dr_var(m_pts, s_pts)[:, 0], [5.0, 7.0])
    # b) dr(qp)
    qp_var = QP(loc=None, val=np.array([0.0, 2.5]))
    assert np.isclose(dr_var(qp_var)[0], 5.0)

    # 4. Check deprecation warnings on legacy eval_* methods
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res_dep = dr_mc.eval_is(0, np.array([[2.5]]))
        assert np.isclose(res_dep[0, 0], 5.0)
        assert any(issubclass(item.category, DeprecationWarning) for item in w)
