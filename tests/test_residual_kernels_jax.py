import numpy as np
import pytest
import jax.numpy as jnp

import dolo
from dolo.numeric.residual_jax import CompiledResidual
from dolo.numeric.residual_kernels_jax import ColoredJacobian, coloring, jacobian_pattern


def test_jacobian_pattern_structural():
    def f(x):
        return jnp.stack(
            [
                x[0] * x[1],
                jnp.maximum(x[2], 0.0),  # depends on x[2] even where the branch is inactive
                x[3] ** 0.0 + 0.0 * x[0],  # literal zeros: no dependency
                jnp.exp(x[1]) / (1.0 + x[3] ** 2),
            ]
        )

    x = np.array([1.0, 2.0, -3.0, 4.0])  # x[2] < 0: d/dx2 max(x2, 0) = 0 here
    pattern = jacobian_pattern(f, (x,), 0)
    expected = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 1],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(pattern, expected)

    groups = coloring(pattern)
    assert sorted(j for g in groups for j in g) == [0, 1, 2, 3]
    for g in groups:
        for i in g:
            for j in g:
                assert i == j or not (pattern[:, i] & pattern[:, j]).any()

    cj = ColoredJacobian(pattern)
    val, Y = cj.value_and_compressed(f, jnp.asarray([1.0, 2.0, 3.0, 4.0]))
    J = cj.expand(Y)
    import jax

    np.testing.assert_allclose(J, jax.jacfwd(f)(jnp.asarray([1.0, 2.0, 3.0, 4.0])), rtol=1e-14)
    np.testing.assert_allclose(val, f(jnp.asarray([1.0, 2.0, 3.0, 4.0])), rtol=1e-14)


def _models():
    rbc = dolo.yaml_import("examples/models/rbc.yaml")
    rbc_mc = dolo.yaml_import("examples/models/rbc_mc.yaml")
    return [
        dolo.numeric.DiscretizedModel(rbc, grid_options=dict(n=[10, 10])),
        dolo.numeric.DiscretizedModel(rbc_mc, grid_options=dict(n=[20])),
    ]


@pytest.mark.parametrize("interp_method", ["cubic", "linear"])
@pytest.mark.parametrize("i_model", [0, 1], ids=["rbc", "rbc_mc"])
def test_fast_kernels_match_point_kernels(i_model, interp_method):
    dmodel = _models()[i_model]
    fast = CompiledResidual(dmodel, interp_method=interp_method, fast=True)
    ref = CompiledResidual(dmodel, interp_method=interp_method, fast=False)
    assert fast._fast is not None and ref._fast is None

    # controls away from the steady state, so that no derivative vanishes by accident
    x0 = dmodel.initial_guess()
    rng = np.random.default_rng(0)
    x = x0.data * (1 + 0.05 * rng.standard_normal(x0.data.shape)) + 0.01
    x = x if dmodel.n_regimes > 1 else x[0]
    c = ref.prefilter(x0.data * (1 + 0.05 * rng.standard_normal(x0.data.shape)))

    def close(a, b):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11 * np.abs(b).max())

    close(fast.F(x, c), ref.F(x, c))
    close(fast.dF_1(x, c), ref.dF_1(x, c))
    r1, J1 = fast.F_and_dF_1(x, c)
    r0, J0 = ref.F_and_dF_1(x, c)
    close(r1, r0)
    close(J1, J0)
    Fx1, S1 = fast.dF_2(x, c)
    Fx0, S0 = ref.dF_2(x, c)
    close(S1, S0)
    close(Fx1, Fx0)
