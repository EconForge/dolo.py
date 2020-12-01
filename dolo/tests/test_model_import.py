import unittest
import numpy


# def test_web_import():
#
#     from dolo import yaml_import
#
#     model = yaml_import("https://raw.githubusercontent.com/EconForge/dolo/master/examples/models/rbc.yaml")
#     assert(len(model.symbols['states'])==2)


def model_evaluation(compiler="numpy", data_layout="columns"):

    from dolo import yaml_import

    model = yaml_import("examples/models/rbc_iid.yaml")

    s0 = model.calibration["states"]
    x0 = model.calibration["controls"]
    e0 = model.calibration["exogenous"]
    p = model.calibration["parameters"]

    assert s0.ndim == 1
    assert x0.ndim == 1
    assert p.ndim == 1

    N = 10
    f = model.functions["arbitrage"]

    if data_layout == "rows":
        ss = numpy.ascontiguousarray(numpy.tile(s0, (N, 1)).T)
        xx = numpy.ascontiguousarray(numpy.tile(x0, (N, 1)).T)
        ee = e0[:, None].repeat(N, axis=1)
    else:
        ss = numpy.tile(s0, (N, 1))
        xx = numpy.tile(x0, (N, 1))
        ee = e0[None, :].repeat(N, axis=0)

    ss = s0[None, :].repeat(N, axis=0)
    xx = x0[None, :].repeat(N, axis=0)
    ee = e0[None, :].repeat(N, axis=0)

    vec_res = f(ee, ss, xx, ee, ss, xx, p)

    res = f(e0, s0, x0, e0, s0, x0, p)

    assert res.ndim == 1

    d = 0
    for i in range(N):
        d += abs(vec_res[i, :] - res).max()
    assert d == 0


if __name__ == "__main__":
    model_evaluation()
