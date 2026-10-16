"""
Tests for the dynspec-based model importer (:mod:`dolo.compiler.model_v2`).

These tests encode the migration contract:

* ``ModelV2`` (built on ``dyno.dynspec`` + JAX) reproduces the legacy
  importer's ``symbols`` / ``calibration`` / ``functions`` API, so that
  ``dolo.algos`` keep working unchanged;
* dolo YAML models and their ``.dyno`` counterparts load to equivalent
  models.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

import dolo

from dolo.compiler.model_v2 import ModelV2, yaml_compose

MODEL_FILES = [
    "examples/models/consumption_savings_iid.yaml",
    "examples/models/rbc.yaml",
    "examples/models/rbc_ar1.yaml",
    "examples/models/rbc_iid.yaml",
    "examples/models/rbc_mc.yaml",
    "examples/models/rmt3_ch11.yaml",
]

_RECIPES = yaml.safe_load(Path("src/dolo/compiler/recipes.yaml").read_text())["dtcc"]

_AUXILIARY_SPEC = {
    "eqs": [
        ["exogenous", 0, "m"],
        ["states", 0, "s"],
        ["controls", 0, "x"],
        ["parameters", 0, "p"],
    ]
}


def _group_args(calib, specs):
    """Build the positional arguments of a legacy function from its specs.

    The specs list every argument group in order, *including* parameters;
    the legacy functions take exactly one argument per group entry.
    ``specs`` is either a mapping with an ``eqs`` key (a recipe group) or a
    plain list of ``[group, shift, name]`` entries (complementarity specs).
    """
    eqs = specs["eqs"] if isinstance(specs, dict) else specs
    return [np.asarray(calib[sg[0]], dtype=float).copy() for sg in eqs]


def _bound_args(calib):
    specs = _RECIPES["specs"]["arbitrage"]["complementarities"]["left-right"]
    return _group_args(calib, specs)


def _load_v2(fname):
    with open(fname, "rt", encoding="utf-8") as f:
        txt = f.read()
    return ModelV2.from_yaml(yaml_compose(txt), filename=fname)


def _group_spec_for(functions, name):
    """Recipe spec (dict or plain list) for a group name, or None for
    complementarity-bound groups."""
    if name in ("auxiliary",):
        return _AUXILIARY_SPEC
    if name in (
        "arbitrage_lb",
        "arbitrage_ub",
        "controls_lb",
        "controls_ub",
    ):
        return _RECIPES["specs"]["arbitrage"]["complementarities"]["left-right"]
    assert name in _RECIPES["specs"], f"no recipe for {name}"
    return _RECIPES["specs"][name]


def _batched(group_args, N):
    """Tile the 1-d arguments to N rows; 2-d arguments pass through."""
    out = []
    for a in group_args:
        a = np.asarray(a, dtype=float)
        out.append(np.tile(a, (N, 1)) if a.ndim == 1 else a)
    return out


@pytest.mark.parametrize("fname", MODEL_FILES)
def test_parity_with_legacy_importer(fname):
    old = dolo.yaml_import(fname)
    new = _load_v2(fname)

    assert {k: v for k, v in old.symbols.items()} == {
        k: v for k, v in new.symbols.items()
    }

    oc, nc = old.calibration, new.calibration
    for group in ("exogenous", "states", "controls", "parameters"):
        assert np.allclose(oc[group], nc[group])

    f_old = old.functions["arbitrage"]
    f_new = new.functions["arbitrage"]
    g_old = old.functions["transition"]
    g_new = new.functions["transition"]

    m, s, x, p = oc["exogenous"], oc["states"], oc["controls"], oc["parameters"]

    assert np.allclose(f_old(m, s, x, m, s, x, p), f_new(m, s, x, m, s, x, p))
    assert np.allclose(g_old(m, s, x, m, p), g_new(m, s, x, m, p))

    if "auxiliary" in old.functions:
        aux_old = old.functions["auxiliary"]
        aux_new = new.functions["auxiliary"]
        assert np.allclose(aux_old(m, s, x, p), aux_new(m, s, x, p))

    # row-vectorized evaluation must agree as well
    N = 5
    ss = np.tile(s, (N, 1))
    xx = np.tile(x, (N, 1))
    ee = np.tile(m, (N, 1))
    assert np.allclose(
        f_old(ee, ss, xx, ee, ss, xx, p),
        f_new(ee, ss, xx, ee, ss, xx, p),
    )
    assert np.allclose(g_old(ee, ss, xx, ee, p), g_new(ee, ss, xx, ee, p))


def _assert_alike(v_old, v_new, name=""):
    """Compare values, treating ``+/-inf`` and ``nan`` at poles alike."""
    a = np.where(np.isinf(v_old), np.nan, np.asarray(v_old, dtype=float))
    b = np.where(np.isinf(v_new), np.nan, np.asarray(v_new, dtype=float))
    finite = np.isfinite(a) & np.isfinite(b)
    if not (np.isnan(a) == np.isnan(b)).all():
        raise AssertionError(f"{name}: NaN positions differ:\n{a}\nvs\n{b}")
    assert np.allclose(
        a[finite], b[finite]
    ), f"{name}: maxdiff {np.abs(a[finite] - b[finite]).max() if finite.any() else 0}"


@pytest.mark.parametrize("fname", MODEL_FILES)
def test_all_group_functions_parity(fname):
    """Every function the legacy importer builds must exist in ModelV2 with
    the same signature, and agree on the values it computes."""

    old = dolo.yaml_import(fname)
    new = _load_v2(fname)

    names = set(old.functions.keys()) | set(new.functions.keys())
    for name in sorted(names):
        assert name in old.functions, f"{name}: missing from legacy importer"
        assert name in new.functions, f"{name}: missing from ModelV2"

        if name in ("auxiliary",):
            spec = _AUXILIARY_SPEC
        elif name in (
            "arbitrage_lb",
            "arbitrage_ub",
            "controls_lb",
            "controls_ub",
        ):
            spec = None  # handled with the dedicated bound args below
        else:
            assert name in _RECIPES["specs"], f"no recipe for {name}"
            spec = _RECIPES["specs"][name]

        if spec is None:
            args_old = _bound_args(old.calibration)
            args_new = _bound_args(new.calibration)
        else:
            args_old = _group_args(old.calibration, spec)
            args_new = _group_args(new.calibration, spec)

        v_old = old.functions[name](*args_old)
        v_new = new.functions[name](*args_new)
        _assert_alike(v_old, v_new, name)


def test_diff_parity_rbc():
    old = dolo.yaml_import("examples/models/rbc.yaml")
    new = _load_v2("examples/models/rbc.yaml")

    oc, nc = old.calibration, new.calibration
    m, s, x, p = oc["exogenous"], oc["states"], oc["controls"], oc["parameters"]
    n_m, n_s, n_x, n_p = nc["exogenous"], nc["states"], nc["controls"], nc["parameters"]

    f_old = old.functions["arbitrage"]
    f_new = new.functions["arbitrage"]

    d_old = f_old(m, s, x, m, s, x, p, diff=True)
    d_new = f_new(n_m, n_s, n_x, n_m, n_s, n_x, n_p, diff=True)
    assert np.allclose(d_old[0], d_new[0])
    for jo, jn in zip(d_old[1:], d_new[1:]):
        # legacy jacobians are finite-difference based (epsilon=1e-8)
        assert np.allclose(jo, jn, atol=1e-5), np.abs(jo - jn).max()

    # jacobians have the expected shapes (f: n_eq x n_group)
    n_eq = len(old.symbols["controls"])
    shapes = [
        jn.shape
        for jn in [
            d_new[1],
            d_new[2],
            d_new[3],
            d_new[4],
            d_new[5],
            d_new[6],
        ]
    ]
    assert shapes == [
        (n_eq, 1),
        (n_eq, 2),
        (n_eq, 2),
        (n_eq, 1),
        (n_eq, 2),
        (n_eq, 2),
    ]


def test_dyno_import_rbc():
    fname = "examples/dyno/rbc_dolo.dyno"
    with open(fname, "rt", encoding="utf-8") as f:
        txt = f.read()
    model = ModelV2.from_dyno(txt, filename=fname)

    assert model.name == "RBC_Dolo"
    assert model.symbols["states"] == ["z", "k"]
    assert model.symbols["controls"] == ["n", "i"]
    assert model.symbols["exogenous"] == ["e_z"]
    assert model.symbols["auxiliaries"] == ["y", "c", "rk", "w"]

    c = model.calibration
    for group in ("states", "controls", "auxiliaries", "parameters"):
        assert c[group].shape[0] == len(model.symbols[group])

    m, s, x, p = c["exogenous"], c["states"], c["controls"], c["parameters"]
    f = model.functions["arbitrage"]
    g = model.functions["transition"]
    assert np.allclose(f(m, s, x, m, s, x, p), 0.0)
    assert np.allclose(g(m, s, x, m, p) - s, 0.0)

    # same calibration as the equivalent YAML model
    yaml_model = _load_v2("examples/models/rbc.yaml")
    yc = yaml_model.calibration
    for group, ynames in yaml_model.symbols.items():
        names = model.symbols[group]
        if group == "parameters":
            # .dyno models declare parameters through assignment order,
            # which need not match the YAML `symbols:` declaration order.
            assert set(names) == set(ynames)
            names = sorted(names)
        else:
            assert names == ynames
        assert np.allclose([c.flat[n] for n in names], [yc.flat[n] for n in names])


def test_import_entry_points():
    model = dolo.yaml_import_v2("examples/models/rbc.yaml")
    assert "arbitrage" in model.functions
    assert "transition" in model.functions


VALUE_MODEL = """
name: dummy_value
symbols:
  states: [k]
  controls: [c]
  exogenous: [e]
  values: [v]
  rewards: [r]
  parameters: [beta, delta]
equations:
  transition: |
    k[t] = (1-delta)*k[t-1] + c[t-1]
  arbitrage: |
    1 - beta*(c[t]/c[t+1])*(1-delta)
  felicity: |
    r[t] = log(c[t])
  value: |
    v[t] = log(c[t]) + beta*v[t+1]
calibration:
  beta: 0.96
  delta: 0.1
  k: 1.0
  c: 0.5
  v: 0.0
  r: 0.0
  e: 0.0
"""


def test_value_and_felicity_groups():
    """The dtcc `value` / `felicity` groups follow the legacy recipes."""
    from dolo.compiler.model import Model

    tree = yaml_compose(VALUE_MODEL)
    old = Model(tree)
    new = ModelV2.from_yaml(tree)

    for name in sorted(set(old.functions) | set(new.functions)):
        assert name in old.functions
        assert name in new.functions
        spec = _RECIPES["specs"][name]
        args_old = _group_args(old.calibration, spec)
        args_new = _group_args(new.calibration, spec)
        v_old = old.functions[name](*args_old)
        v_new = new.functions[name](*args_new)
        assert np.allclose(v_old, v_new), f"{name}: {v_old} vs {v_new}"


def test_dyno_options_and_domain():
    """@options: / @domain: one-liners are picked up from the .dyno metadata."""
    fname = "examples/dyno/rbc_dolo.dyno"
    with open(fname, "rt", encoding="utf-8") as f:
        txt = f.read()

    txt = txt.rsplit("@run: steady", 1)[0]
    txt += (
        "\n@options: {grid: {type: cartesian, n: [4, 4]}}\n"
        "@domain: {z: [-0.1, 0.1], k: [5.0, 12.0]}\n"
    )
    model = ModelV2.from_dyno(txt, filename=fname)

    assert model.options["grid"]["type"] == "cartesian"
    assert model.domain.min.tolist() == [-0.1, 5.0]
    assert model.domain.max.tolist() == [0.1, 12.0]
    g = model.endo_grid
    assert g.nodes.shape == (16, 2)

    # `discretize()` on .dyno models: exogenous shocks provide the markov
    # part, `@domain` + `@options.grid` the endogenous part.
    from dolo.numeric.processes import DiscretizedIIDProcess

    grid, dprocess = model.discretize()
    assert isinstance(dprocess, DiscretizedIIDProcess)
    assert dprocess.n_nodes == 1
    assert grid["endo"].nodes.shape == (16, 2)
    assert grid["exo"].nodes is None  # iid processes carry an empty grid


# ---------------------------------------------------------------------------
# shape regression tests: batched calls must return exactly (N, n_out)
# (guards the jax.vmap p-broadcast class of bugs, DEV_NOTES §6.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fname", MODEL_FILES)
def test_batched_calls_return_exact_shapes(fname):
    new = _load_v2(fname)
    c = new.calibration
    N = 7

    for name, fn in new.functions.items():
        spec = _group_spec_for(new.functions, name)
        point_args = _group_args(c, spec)
        pval = np.asarray(fn(*point_args))
        assert pval.ndim == 1, f"{name}: point call should return a 1-d vector"

        # row-vectorized variable args, parameters kept 1-d (unmapped p):
        # the result must be (N, n_out) — not (N, n_p, n_out) or similar.
        batched = _batched(point_args, N)
        bval = np.asarray(fn(*batched))
        assert bval.shape == (N, pval.shape[0]), (
            f"{name}: batched call returned {bval.shape}, "
            f"expected {(N, pval.shape[0])}"
        )

        # parameters broadcast as well (legacy does this in `simulate`)
        bval2 = np.asarray(fn(*_batched(point_args[:-1], N), *point_args[-1:]))
        assert bval2.shape == (
            N,
            pval.shape[0],
        ), f"{name}: batched call with scalar params returned {bval2.shape}"
        assert np.allclose(bval, bval2, equal_nan=True)


def test_batched_with_inconsistent_shapes_raises():
    new = _load_v2("examples/models/rbc.yaml")
    c = new.calibration
    f = new.functions["transition"]
    m, s, x, p = c["exogenous"], c["states"], c["controls"], c["parameters"]
    with pytest.raises(ValueError):
        f(np.tile(m, (3, 1)), s, x, np.tile(m, (4, 1)), p)


# ---------------------------------------------------------------------------
# algorithm smoke tests on ModelV2 (DEV_NOTES §6.3): same results as legacy
# ---------------------------------------------------------------------------


def test_perturb_and_simulate_parity():
    from dolo.algos import perturb, simulate

    old = dolo.yaml_import("examples/models/rbc.yaml")
    new = _load_v2("examples/models/rbc.yaml")

    dr_o = perturb(old, verbose=False).dr
    dr_n = perturb(new, verbose=False).dr

    e = old.calibration["exogenous"][None, :]
    s0 = old.calibration["states"][None, :]
    rng = np.random.RandomState(0)
    pts = s0 + 0.05 * rng.randn(10, 2)  # includes the steady state
    pts_m = np.tile(e, (10, 1))
    assert np.allclose(
        np.asarray(dr_o.eval_ms(pts_m, pts)),
        np.asarray(dr_n.eval_ms(pts_m, pts)),
        atol=1e-7,
    )

    # identical shocks -> identical paths (1e-7-scale, near-bit-equal)
    T, N = 40, 3
    driving = 0.01 * rng.randn(T, N, 1)
    sim_o = simulate(old, dr_o, driving_process=driving, N=N)
    sim_n = simulate(new, dr_n, driving_process=driving, N=N)
    assert np.allclose(sim_o.values, sim_n.values, rtol=1e-9, atol=1e-5)


def test_time_iteration_parity(tmp_path):
    from dolo.algos import time_iteration

    # small grid keeps the smoke test fast (~ a few seconds per importer)
    src = Path("examples/models/rbc.yaml").read_text()
    src = src.replace("n: [100, 100]", "n: [5, 5]")
    fname = tmp_path / "rbc_small.yaml"
    fname.write_text(src)

    old = dolo.yaml_import(str(fname))
    new = dolo.yaml_import_v2(str(fname))

    dr_o = time_iteration(old, verbose=False).dr
    dr_n = time_iteration(new, verbose=False).dr

    e = new.calibration["exogenous"]
    s = new.calibration["states"]
    rng = np.random.RandomState(2)
    pts = s[None, :] + 0.15 * rng.randn(6, 2)
    assert np.allclose(np.asarray(dr_o(e, pts)), np.asarray(dr_n(e, pts)), atol=1e-8)


def test_egm_parity():
    from dolo.algos.egm import egm

    fname = "examples/models/consumption_savings_iid.yaml"
    old = dolo.yaml_import(fname)
    new = _load_v2(fname)

    a_grid = np.linspace(0.1, 10, 10) ** 2
    sol_o = egm(old, a_grid=a_grid, verbose=False)
    sol_n = egm(new, a_grid=a_grid, verbose=False)

    e = new.calibration["exogenous"]
    rng = np.random.RandomState(3)
    for N in (1, 5):
        mm = np.tile(e, (N, 1))
        ss = a_grid[None, :] * (1 + 0.05 * rng.randn(N, 10))
        xo = np.asarray(sol_o.dr(mm, ss))
        xn = np.asarray(sol_n.dr(mm, ss))
        assert np.array_equal(xo, xn)


# ---------------------------------------------------------------------------
# standalone bounds groups (`controls_lb`/`controls_ub`) — DEV_NOTES §6.1:
# the `controls_lb -> arbitrage_lb` alias must not hijack real keys.
# ---------------------------------------------------------------------------

GROWTH_MODEL = """
name: growth_vi
symbols:
  exogenous: [e]
  states: [k]
  controls: [c]
  values: [V]
  rewards: [u]
  parameters: [beta, alpha, delta, sig_e]
equations:
  transition:
    - k[t] = exp(e[t])*((1-delta)*k[t-1] + k[t-1]^alpha - c[t-1])
  felicity:
    - u[t] = log(c[t])
  value:
    - V[t] = log(c[t]) + beta*V[t+1]
  controls_lb:
    - "c = 1e-4"
  controls_ub:
    - "c = (1-delta)*k + k^alpha"
calibration:
  beta: 0.95
  alpha: 0.3
  delta: 0.1
  sig_e: 0.01
  k: 0.9
  c: 0.3
  V: log(0.3)/(1-beta)
  u: log(0.3)
  e: 0.0
exogenous: !Normal
  Σ: [[sig_e**2]]
domain:
  k: [0.5, 3.0]
options:
  grid: {type: cartesian, n: [12]}
"""


def test_eval_formula_parity():
    """`eval_formula` (dynspec-backed) agrees with the legacy evaluator."""
    import pandas as pd

    old = dolo.yaml_import("examples/models/rbc.yaml")
    new = _load_v2("examples/models/rbc.yaml")

    rng = np.random.RandomState(7)
    df = pd.DataFrame(index=range(6))
    for name, val in old.calibration.flat.items():
        if name in old.symbols["parameters"]:
            df[name] = val
        else:
            df[name] = val * (1 + 0.02 * rng.randn(6))

    exprs = [
        "delta*k(0)-i(0)",
        "1+1",
        "beta/(1+delta/(1-beta))",
    ]
    for expr in exprs:
        v_old = old.eval_formula(expr, df)
        v_new = new.eval_formula(expr, df)
        np.testing.assert_allclose(v_old, v_new, equal_nan=True)

    # dynspec-style timing is accepted by ModelV2
    assert new.eval_formula("rho*z[t-1] + e_z", df)[2] == pytest.approx(
        old.calibration["rho"] * df["z"].shift(1).iloc[2] + df["e_z"].iloc[2]
    )


def test_standalone_bounds_groups():
    m = ModelV2.from_yaml(yaml_compose(GROWTH_MODEL), filename="growth_vi")

    assert set(("controls_lb", "controls_ub")) <= set(m.functions)
    lb = m.functions["controls_lb"]
    ub = m.functions["controls_ub"]
    assert lb is not None and "arbitrage" not in lb.__doc__

    p = m.calibration["parameters"]
    e = m.calibration["exogenous"]
    s = m.calibration["states"]
    assert np.allclose(lb(e, s, p), [1e-4])
    assert np.allclose(ub(e, s, p), [0.9 * 0.9 + 0.9**0.3])

    assert m.x_bounds == [lb, ub]

    # when no standalone bounds exist, the legacy aliases still work
    rbc_iid = _load_v2("examples/models/rbc_iid.yaml")
    assert rbc_iid.functions["controls_lb"] is rbc_iid.functions["arbitrage_lb"]
    assert rbc_iid.x_bounds == [
        rbc_iid.functions["arbitrage_lb"],
        rbc_iid.functions["arbitrage_ub"],
    ]
