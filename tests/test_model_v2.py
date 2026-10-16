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
