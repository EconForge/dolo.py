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

import numpy as np
import pytest

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
