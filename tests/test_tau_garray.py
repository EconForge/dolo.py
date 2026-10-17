"""
Tests for GArray, QP, Tau, and DiscretizedModel in dolo.numeric.
"""

import numpy as np
import pytest

from dolo.compiler.model_v2 import ModelV2, yaml_compose
from dolo.numeric.decision_rule import DecisionRule
from dolo.numeric.garray import GArray, QP, SliceQP
from dolo.numeric.grids import (
    UniformCartesianGrid,
    ProductGrid,
    UnstructuredGrid,
    EmptyGrid,
)
from dolo.numeric.tau import Tau, trembling_hand
from dolo.numeric.discretized_model import DiscretizedModel
from dolo.algos.time_iteration import residuals_simple, time_iteration
from dolo.numeric.optimize.newton import newton, SerialDifferentiableFunction


def _load_v2(path: str) -> ModelV2:
    with open(path, "rt", encoding="utf-8") as f:
        txt = f.read()
    return ModelV2.from_yaml(yaml_compose(txt), filename=path)


# ---------------------------------------------------------------------------
# 1. GArray and QP Unit Tests
# ---------------------------------------------------------------------------


def test_garray_cartesian():
    grid = UniformCartesianGrid(min=[0.1, 1.0], max=[0.5, 5.0], n=[10, 20])
    ga = GArray.zeros(grid, n_vars=3)
    assert ga.shape == (1, 200, 3)

    # Test ravel & unravel
    flat = ga.ravel()
    assert flat.shape == (600,)
    flat_mod = flat + 1.5
    ga2 = ga.unravel(flat_mod)
    assert np.allclose(ga2.data, 1.5)
    assert ga2.norm() == 1.5

    # Arithmetic
    diff = ga2 - ga
    assert np.allclose(diff.data, 1.5)
    assert (ga2 * 2.0).norm() == 3.0


def test_garray_product_grid():
    exo_grid = UnstructuredGrid([[-0.01], [0.01]])
    endo_grid = UniformCartesianGrid(min=[0.5], max=[2.0], n=[50])
    grid = ProductGrid(exo_grid, endo_grid, names=["exo", "endo"])

    ga = GArray.zeros(grid, n_vars=2)
    assert ga.shape == (2, 50, 2)

    ga[0] = 1.0
    ga[1] = 2.0
    assert np.allclose(ga[0], 1.0)
    assert np.allclose(ga[1], 2.0)

    # Test ravel/unravel round-trip
    flat = ga.ravel()
    assert flat.shape == (200,)
    ga_rec = ga.unravel(flat)
    assert np.allclose(ga_rec.data, ga.data)


def test_qualified_point_and_batch():
    from dolo.numeric import QualifiedPoint, QP, QualifiedBatch, SliceQP, QualifiedSlice

    # Verify alias identities
    assert QP is QualifiedPoint
    assert SliceQP is QualifiedBatch
    assert QualifiedSlice is QualifiedBatch

    # Construct QualifiedPoint
    qp = QualifiedPoint(loc=(0, 2), val=np.array([1.0, 2.5]))
    assert qp.index == 0
    assert np.allclose(qp.coords, [1.0, 2.5])
    assert "QualifiedPoint" in repr(qp)

    # Construct QualifiedBatch
    batch = QualifiedBatch(
        loc=1, exo=np.array([0.02]), endo=np.array([[1.0, 2.0], [1.5, 2.5]])
    )
    assert batch.val.shape == (2, 3)
    assert np.allclose(batch.val[0], [0.02, 1.0, 2.0])
    assert np.allclose(batch.val[1], [0.02, 1.5, 2.5])


def test_trembling_hand_1d():
    grid = UniformCartesianGrid(min=[0.0], max=[1.0], n=[11])  # step = 0.1
    # Test point at 0.25 (between index 2 (0.2) and index 3 (0.3), weights 0.5, 0.5)
    pts = np.array([[0.25]])
    projections = trembling_hand(grid, pts)
    # 1D has 2 vertices
    assert len(projections) == 2
    w_left, idx_left = projections[0]
    w_right, idx_right = projections[1]

    assert idx_left[0] == 2
    assert idx_right[0] == 3
    assert np.isclose(w_left[0], 0.5)
    assert np.isclose(w_right[0], 0.5)
    assert np.isclose(w_left[0] + w_right[0], 1.0)


# ---------------------------------------------------------------------------
# 2. Universal Euler Residuals & Parity
# ---------------------------------------------------------------------------


def euler_residuals(
    dmodel: DiscretizedModel, controls: GArray, dr: DecisionRule
) -> GArray:
    """Universal Euler residuals using Tau, GArray, and direct dr(SliceQP) calls."""
    res = GArray.zeros_like(controls)
    for i, m_val, s_endo in dmodel.regimes():
        xm = controls[i]
        for prob, S in dmodel.tau.slice_transitions(i, m_val, s_endo, xm):
            # dr(S) takes SliceQP directly!
            XM = dr(S)
            rr = dmodel.arbitrage(m_val, s_endo, xm, S.exo, S.endo, XM)
            res[i] += prob * rr
    return res


@pytest.mark.parametrize(
    "model_path",
    [
        "examples/models/rbc_mc.yaml",
        "examples/models/rbc_iid.yaml",
    ],
)
def test_tau_euler_parity(model_path: str):
    model = _load_v2(model_path)
    dmodel = DiscretizedModel(model)

    controls = dmodel.initial_guess()
    dr = DecisionRule(
        dmodel.grid.grids[0],
        dmodel.grid.grids[1],
        dprocess=dmodel.dprocess,
        interp_method="cubic",
    )
    # dr.set_values accepts GArray directly
    dr.set_values(controls)

    # Legacy residuals
    f = model.functions["arbitrage"]
    g = model.functions["transition"]
    s = dmodel.endo_nodes
    p = model.calibration["parameters"]
    res_legacy = residuals_simple(f, g, s, controls.data, dr, dmodel.dprocess, p)

    # Universal Tau residuals
    res_tau = euler_residuals(dmodel, controls, dr)

    # Exact bitwise parity check
    diff = np.max(np.abs(res_legacy - res_tau.data))
    assert diff < 1e-14, f"Parity mismatch on {model_path}: diff={diff}"


# ---------------------------------------------------------------------------
# 3. DecisionRule with QP and SliceQP
# ---------------------------------------------------------------------------


def test_decision_rule_qp_dispatch():
    model = _load_v2("examples/models/rbc_mc.yaml")
    dmodel = DiscretizedModel(model)
    controls = dmodel.initial_guess()
    dr = DecisionRule(
        dmodel.grid.grids[0],
        dmodel.grid.grids[1],
        dprocess=dmodel.dprocess,
        interp_method="cubic",
    )
    dr.set_values(controls)

    # Test SliceQP
    s_endo = dmodel.endo_nodes
    sqp = SliceQP(loc=1, exo=np.array([0.01, 0.1]), endo=s_endo)
    res_slice = dr(sqp)
    res_expected = dr.eval_is(1, s_endo)
    assert np.allclose(res_slice, res_expected)

    # Test single QP
    qp = QP(loc=(1, s_endo[5]), val=np.concatenate([[0.01, 0.1], s_endo[5]]))
    res_point = dr(qp)
    assert np.allclose(res_point, res_expected[5])


# ---------------------------------------------------------------------------
# 4. Transition Matrix & Ergodic Distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_path",
    [
        "examples/models/rbc_mc.yaml",
        "examples/models/rbc_iid.yaml",
    ],
)
def test_transition_matrix_and_ergodic(model_path: str):
    model = _load_v2(model_path)
    dmodel = DiscretizedModel(model)
    controls = dmodel.initial_guess()

    P = dmodel.transition_matrix(controls)

    # Properties of Markov transition matrix
    assert P.shape == (dmodel.N_total, dmodel.N_total)
    assert np.all(P >= 0.0), "Negative transition probabilities found!"
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Row sums differ from 1.0: {row_sums}"

    # Ergodic distribution
    mu = dmodel.ergodic_distribution(controls)
    assert isinstance(mu, GArray)
    assert np.isclose(
        mu.data.sum(), 1.0
    ), f"Ergodic distribution does not sum to 1: {mu.data.sum()}"
    assert np.all(mu.data >= 0.0), "Negative stationary probabilities!"


# ---------------------------------------------------------------------------
# 5. Full Time Iteration Solver Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_path",
    [
        "examples/models/rbc_mc.yaml",
        "examples/models/rbc_iid.yaml",
    ],
)
def test_full_solver_parity(model_path: str):
    model = _load_v2(model_path)
    dmodel = DiscretizedModel(model)

    # Legacy solver
    sol_legacy = time_iteration(model, verbose=False)

    # Universal Tau solver
    controls = dmodel.initial_guess()
    dr = DecisionRule(
        dmodel.grid.grids[0],
        dmodel.grid.grids[1],
        dprocess=dmodel.dprocess,
        interp_method="cubic",
    )

    err = 10.0
    it = 0
    maxit = 1000
    tol = 1e-6
    n_x = controls.data.shape[-1]

    while err > tol and it < maxit:
        it += 1
        dr.set_values(controls)

        fn = (
            lambda flat_x: euler_residuals(dmodel, controls.unravel(flat_x), dr)
            .ravel()
            .reshape((-1, n_x))
        )

        dfn = SerialDifferentiableFunction(fn)
        flat_controls, _ = newton(
            dfn, controls.ravel().reshape((-1, n_x)), verbose=False, maxit=10
        )
        new_controls = controls.unravel(flat_controls)

        err = (new_controls - controls).norm()
        controls = new_controls

    # Parity check on final decision rule coefficients
    if isinstance(sol_legacy.dr.coefficients, list):
        dr_diff = max(
            float(np.max(np.abs(c1 - c2)))
            for c1, c2 in zip(sol_legacy.dr.coefficients, dr.coefficients)
        )
    else:
        dr_diff = float(np.max(np.abs(sol_legacy.dr.coefficients - dr.coefficients)))

    assert (
        dr_diff < 1e-5
    ), f"Decision rule parity failed on {model_path}! diff={dr_diff}"
