"""
JAX-compiled Euler residual function and automatic differentiation.

Follows the Dolo.jl architecture:
- Dolo.F(model, s, x, φ): Point Euler equation residuals
- Dolo.dF_1(model, s, x, φ): Point Jacobian with respect to controls x
- Dolo.F(model, controls, φ): Vectorized across the grid (via jax.vmap)
- Dolo.dF_1(model, controls, φ): Vectorized Jacobians across the grid (via jax.vmap)
- Joint evaluation F_and_dF: Evaluates both residuals and Jacobians in a single fused XLA kernel.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from dolo.numeric.discretized_model import DiscretizedModel
from dolo.numeric.garray import GArray
from dolo.numeric.interpolation_jax import (
    eval_cubic_point_jax,
    eval_linear_point_jax,
    prefilter_cubic_jax,
)

jax.config.update("jax_enable_x64", True)


def _get_raw_jnp_point_fn(
    fun: Any, model: Any = None, fun_name: str = None
) -> Callable:
    """Extract or construct a pure JAX point function from a dolo function."""
    if hasattr(fun, "point_jnp"):
        return fun.point_jnp
    # Inspect closure for point_fn in ModelV2 functions
    if hasattr(fun, "__closure__") and fun.__closure__:
        for cell in fun.__closure__:
            obj = cell.cell_contents
            if hasattr(obj, "_fun"):
                orig = obj._fun
                if hasattr(orig, "__closure__") and orig.__closure__:
                    vmap_f = orig.__closure__[0].cell_contents
                    if hasattr(vmap_f, "__closure__") and vmap_f.__closure__:
                        for c in vmap_f.__closure__:
                            cand = c.cell_contents
                            if callable(cand) and getattr(cand, "__name__", "") == "f":
                                return cand
    # If it's a legacy dolang guvectorize object, try to construct ModelV2
    if model is not None and hasattr(model, "data") and fun_name is not None:
        try:
            from dolo.compiler.model_v2 import ModelV2

            v2 = ModelV2.from_yaml(model.data)
            target = "arbitrage" if fun_name in ("f", "arbitrage") else "transition"
            if target in v2.functions:
                return v2.functions[target].point_jnp
        except Exception:
            pass
    # Fallback to fun directly
    return fun


class CompiledResidual:
    """Compiled JAX residual operator mirroring Dolo.jl Dolo.F / Dolo.dF_1."""

    def __init__(
        self,
        dmodel: DiscretizedModel,
        interp_method: str = "cubic",
        diff_mode: str = "fwd",
        fast: bool = True,
    ):
        self.dmodel = dmodel
        self.interp_method = interp_method
        self.diff_mode = diff_mode

        # Model functions
        self.f_pt = _get_raw_jnp_point_fn(dmodel.f, dmodel.model, "f")
        self.g_pt = _get_raw_jnp_point_fn(dmodel.g, dmodel.model, "g")
        self.parms = jnp.asarray(dmodel.parms, dtype=jnp.float64)

        # Discretization details
        self.grid = dmodel.endo_grid
        self.a = jnp.asarray(self.grid.min, dtype=jnp.float64)
        self.b = jnp.asarray(self.grid.max, dtype=jnp.float64)
        self.n = jnp.asarray(self.grid.n, dtype=jnp.int32)
        self.grid_shape = tuple(int(x) for x in self.grid.n)

        # Exogenous shock quadrature
        self.dprocess = dmodel.dprocess
        self.n_regimes = dmodel.n_regimes

        # Precompute regime transition structures
        self.regime_data = []
        for i_m in range(self.n_regimes):
            m_val = jnp.asarray(self.dprocess.node(i_m), dtype=jnp.float64)
            n_inodes = self.dprocess.n_inodes(i_m)
            inodes = jnp.asarray(
                [self.dprocess.inode(i_m, k) for k in range(n_inodes)],
                dtype=jnp.float64,
            )
            weights = jnp.asarray(
                [self.dprocess.iweight(i_m, k) for k in range(n_inodes)],
                dtype=jnp.float64,
            )
            self.regime_data.append((m_val, inodes, weights))

        self.same_inodes = all(
            len(w) == len(self.regime_data[0][2]) for _, _, w in self.regime_data
        )
        if self.same_inodes and self.n_regimes > 1:
            self.m_vals = jnp.stack([m for m, _, _ in self.regime_data], axis=0)
            self.all_inodes = jnp.stack(
                [inodes for _, inodes, _ in self.regime_data], axis=0
            )
            self.all_weights = jnp.stack([w for _, _, w in self.regime_data], axis=0)
        else:
            self.m_vals = None
            self.all_inodes = None
            self.all_weights = None

        # Build JAX kernels
        self._build_kernels()
        self._fast = self._build_fast() if fast else None

    def _build_kernels(self):
        f_pt = self.f_pt
        g_pt = self.g_pt
        parms = self.parms
        a, b, n = self.a, self.b, self.n
        interp_method = self.interp_method

        def eval_interp_pt(coeffs, S):
            if interp_method == "cubic":
                return eval_cubic_point_jax(coeffs, S, a, b, n)
            else:
                return eval_linear_point_jax(coeffs, S, a, b, n)

        # Single regime point residual function
        if self.n_regimes == 1:
            m_val, inodes, weights = self.regime_data[0]
            n_inodes = len(weights)

            def F_pt(s, x, coeffs):
                res = jnp.zeros_like(x)
                for k in range(n_inodes):
                    M = inodes[k]
                    w = weights[k]
                    S = g_pt(m_val, s, x, M, parms)
                    X = eval_interp_pt(coeffs, S)
                    r = f_pt(m_val, s, x, M, S, X, parms)
                    res = res + w * r
                return res

        else:
            # Multi-regime point residual function
            def F_pt(s, x, coeffs, i_m: int):
                m_val, inodes, weights = self.regime_data[i_m]
                n_inodes = len(weights)
                res = jnp.zeros_like(x)
                for k in range(n_inodes):
                    M = inodes[k]
                    w = weights[k]
                    S = g_pt(m_val, s, x, M, parms)
                    # Next regime target: in dprocess, inode target is k
                    target_regime = 0 if self.n_regimes == 1 else k
                    X = eval_interp_pt(coeffs[target_regime], S)
                    r = f_pt(m_val, s, x, M, S, X, parms)
                    res = res + w * r
                return res

        self._F_pt_raw = F_pt

        # Point derivative: forward-mode (jacfwd) or reverse-mode (jacrev)
        diff_fn = jax.jacfwd if self.diff_mode == "fwd" else jax.jacrev

        def value_and_jac(f, x):
            """Value and Jacobian of f at x in one pass (the value is the auxiliary output),
            instead of evaluating f once more for the value."""

            def f_aux(u):
                r = f(u)
                return r, r

            jac, val = diff_fn(f_aux, has_aux=True)(x)
            return val, jac

        if self.n_regimes == 1:

            def dF_pt(s, x, coeffs):
                return diff_fn(lambda u: F_pt(s, u, coeffs))(x)

            def F_and_dF_pt(s, x, coeffs):
                return value_and_jac(lambda u: F_pt(s, u, coeffs), x)

            self.F_pt_jit = jax.jit(F_pt)
            self.dF_pt_jit = jax.jit(dF_pt)
            self.F_and_dF_pt_jit = jax.jit(F_and_dF_pt)

            # Vectorized across continuous state grid using jax.vmap
            self.F_vmap = jax.jit(jax.vmap(F_pt, in_axes=(0, 0, None)))
            self.dF_vmap = jax.jit(jax.vmap(dF_pt, in_axes=(0, 0, None)))
            self.F_and_dF_vmap = jax.jit(jax.vmap(F_and_dF_pt, in_axes=(0, 0, None)))

            def branch_eval(s, x, coeffs, M, w):
                S = g_pt(m_val, s, x, M, parms)
                X = eval_interp_pt(coeffs, S)
                df_dX = diff_fn(lambda u: f_pt(m_val, s, x, M, S, u, parms))(X)
                return w * df_dX, S

            self.dF_2_branch_vmap = jax.jit(
                jax.vmap(branch_eval, in_axes=(0, 0, None, None, None))
            )

        else:
            # Multi-regime vectorized kernels
            self.regime_vmap = []
            for i_m in range(self.n_regimes):
                F_regime = functools.partial(F_pt, i_m=i_m)
                dF_regime = lambda s, x, coeffs, F_regime=F_regime: diff_fn(
                    lambda u: F_regime(s, u, coeffs)
                )(x)
                F_and_dF_regime = (
                    lambda s, x, coeffs, F_regime=F_regime: value_and_jac(
                        lambda u: F_regime(s, u, coeffs), x
                    )
                )

                self.regime_vmap.append(
                    {
                        "F": jax.jit(jax.vmap(F_regime, in_axes=(0, 0, None))),
                        "dF": jax.jit(jax.vmap(dF_regime, in_axes=(0, 0, None))),
                        "F_and_dF": jax.jit(
                            jax.vmap(F_and_dF_regime, in_axes=(0, 0, None))
                        ),
                    }
                )

            if self.same_inodes:

                def F_pt_all(s, x, coeffs, m_val, inodes, weights):
                    res = jnp.zeros_like(x)
                    n_inodes = inodes.shape[0]
                    for k in range(n_inodes):
                        M = inodes[k]
                        w = weights[k]
                        S = g_pt(m_val, s, x, M, parms)
                        target_regime = k
                        X = eval_interp_pt(coeffs[target_regime], S)
                        r = f_pt(m_val, s, x, M, S, X, parms)
                        res = res + w * r
                    return res

                def dF_pt_all(s, x, coeffs, m_val, inodes, weights):
                    return diff_fn(
                        lambda u: F_pt_all(s, u, coeffs, m_val, inodes, weights)
                    )(x)

                def F_and_dF_pt_all(s, x, coeffs, m_val, inodes, weights):
                    return value_and_jac(
                        lambda u: F_pt_all(s, u, coeffs, m_val, inodes, weights), x
                    )

                F_regime_v = jax.vmap(F_pt_all, in_axes=(0, 0, None, None, None, None))
                self.F_all_vmap = jax.jit(
                    jax.vmap(F_regime_v, in_axes=(None, 0, None, 0, 0, 0))
                )

                dF_regime_v = jax.vmap(
                    dF_pt_all, in_axes=(0, 0, None, None, None, None)
                )
                self.dF_all_vmap = jax.jit(
                    jax.vmap(dF_regime_v, in_axes=(None, 0, None, 0, 0, 0))
                )

                F_and_dF_regime_v = jax.vmap(
                    F_and_dF_pt_all, in_axes=(0, 0, None, None, None, None)
                )
                self.F_and_dF_all_vmap = jax.jit(
                    jax.vmap(F_and_dF_regime_v, in_axes=(None, 0, None, 0, 0, 0))
                )
                n_inodes_const = len(self.regime_data[0][2])

                def dF_2_pt_all(s, x, coeffs, m_val, inodes, weights):
                    Fx_list = []
                    S_list = []
                    for k in range(n_inodes_const):
                        M = inodes[k]
                        w = weights[k]
                        S = g_pt(m_val, s, x, M, parms)
                        target_regime = k
                        X = eval_interp_pt(coeffs[target_regime], S)
                        df_dX = diff_fn(lambda u: f_pt(m_val, s, x, M, S, u, parms))(X)
                        Fx_list.append(w * df_dX)
                        S_list.append(S)
                    return jnp.stack(Fx_list, axis=0), jnp.stack(S_list, axis=0)

                dF_2_regime_v = jax.vmap(
                    dF_2_pt_all, in_axes=(0, 0, None, None, None, None)
                )
                self.dF_2_all_vmap = jax.jit(
                    jax.vmap(dF_2_regime_v, in_axes=(None, 0, None, 0, 0, 0))
                )
            else:
                self.F_all_vmap = None
                self.dF_all_vmap = None
                self.F_and_dF_all_vmap = None
                self.dF_2_all_vmap = None

            self.dF_2_regimes = []
            for i_m in range(self.n_regimes):
                m_val_i, inodes_i, weights_i = self.regime_data[i_m]

                def make_branch_eval_i(m_v):
                    def branch_eval_i(s, x, coeffs_target, M, w):
                        S = g_pt(m_v, s, x, M, parms)
                        X = eval_interp_pt(coeffs_target, S)
                        df_dX = diff_fn(lambda u: f_pt(m_v, s, x, M, S, u, parms))(X)
                        return w * df_dX, S

                    return branch_eval_i

                bv_i = jax.jit(
                    jax.vmap(
                        make_branch_eval_i(m_val_i), in_axes=(0, 0, None, None, None)
                    )
                )
                self.dF_2_regimes.append((m_val_i, inodes_i, weights_i, bv_i))

    def _build_fast(self):
        """Staged grid kernels (`residual_kernels_jax.GridKernels`) replacing the vmapped
        point kernels F_vmap, dF_vmap, F_and_dF_vmap (one regime) and F_all_vmap, dF_all_vmap,
        F_and_dF_all_vmap (several regimes with the same number of nodes), and used by dF_2.
        Their derivatives are forward mode whatever `diff_mode`."""
        from dolo.numeric.residual_kernels_jax import GridKernels

        if self.n_regimes > 1 and not self.same_inodes:
            return None
        if self.n_regimes == 1:
            m_val, inodes, weights = self.regime_data[0]
            mv, IN, W = m_val[None], inodes[None], weights[None]
            regimes = np.zeros(inodes.shape[0], dtype=np.int32)
        else:
            mv, IN, W = self.m_vals, self.all_inodes, self.all_weights
            regimes = np.arange(IN.shape[1], dtype=np.int32)
        fast = GridKernels(
            self.f_pt,
            self.g_pt,
            tuple(zip(self.grid.min, self.grid.max, self.grid.n)),
            3 if self.interp_method == "cubic" else 1,
            regimes,
            mv,
            IN,
            W,
            self.parms,
            jnp.asarray(self.dmodel.endo_nodes, dtype=jnp.float64),
            len(self.dmodel.model.symbols["controls"]),
        )

        if self.n_regimes == 1:
            self.F_vmap = lambda s, x, c: fast.F(x[None], c[None])[0]
            self.dF_vmap = lambda s, x, c: fast.F_and_dF_1(x[None], c[None])[1][0]

            def F_and_dF_vmap(s, x, c):
                r, J = fast.F_and_dF_1(x[None], c[None])
                return r[0], J[0]

            self.F_and_dF_vmap = F_and_dF_vmap
        else:
            self.F_all_vmap = lambda s, x, c, *_: fast.F(x, c)
            self.dF_all_vmap = lambda s, x, c, *_: fast.F_and_dF_1(x, c)[1]
            self.F_and_dF_all_vmap = lambda s, x, c, *_: fast.F_and_dF_1(x, c)
        return fast

    def prefilter(
        self, controls: Union[GArray, np.ndarray, jnp.ndarray]
    ) -> jnp.ndarray:
        """Precomputes interpolation coefficients from grid values."""
        if isinstance(controls, GArray):
            c_data = controls.data
        else:
            c_data = np.asarray(controls)

        if self.n_regimes == 1:
            vals = c_data[0] if c_data.ndim == 3 else c_data
            shaped_vals = vals.reshape(self.grid_shape + (-1,))
            if self.interp_method == "cubic":
                return prefilter_cubic_jax(self.grid, shaped_vals)
            return jnp.asarray(shaped_vals, dtype=jnp.float64)
        else:
            # Multi-regime coefficients
            all_coeffs = []
            for i in range(self.n_regimes):
                vals = c_data[i]
                shaped_vals = vals.reshape(self.grid_shape + (-1,))
                if self.interp_method == "cubic":
                    coeffs = prefilter_cubic_jax(self.grid, shaped_vals)
                else:
                    coeffs = jnp.asarray(shaped_vals, dtype=jnp.float64)
                all_coeffs.append(coeffs)
            return jnp.stack(all_coeffs, axis=0)

    def F_point(
        self,
        s: Union[np.ndarray, jnp.ndarray],
        x: Union[np.ndarray, jnp.ndarray],
        coeffs: jnp.ndarray,
        i_m: int = 0,
    ) -> jnp.ndarray:
        """Evaluate Euler residual vector at a single point (s, x)."""
        s_jax = jnp.asarray(s, dtype=jnp.float64)
        x_jax = jnp.asarray(x, dtype=jnp.float64)
        if self.n_regimes == 1:
            return self.F_pt_jit(s_jax, x_jax, coeffs)
        return self.regime_vmap[i_m]["F"](s_jax[None, :], x_jax[None, :], coeffs)[0]

    def dF_point(
        self,
        s: Union[np.ndarray, jnp.ndarray],
        x: Union[np.ndarray, jnp.ndarray],
        coeffs: jnp.ndarray,
        i_m: int = 0,
    ) -> jnp.ndarray:
        """Evaluate Jacobian dF/dx matrix at a single point (s, x)."""
        s_jax = jnp.asarray(s, dtype=jnp.float64)
        x_jax = jnp.asarray(x, dtype=jnp.float64)
        if self.n_regimes == 1:
            return self.dF_pt_jit(s_jax, x_jax, coeffs)
        return self.regime_vmap[i_m]["dF"](s_jax[None, :], x_jax[None, :], coeffs)[0]

    def F(
        self,
        controls: Union[GArray, np.ndarray, jnp.ndarray],
        coeffs: Optional[jnp.ndarray] = None,
    ) -> Union[GArray, jnp.ndarray]:
        """Vectorized Euler residuals across the grid using jax.vmap.

        Mirrors Dolo.jl: Dolo.F(model, controls, φ)
        """
        if coeffs is None:
            coeffs = self.prefilter(controls)

        s_nodes = jnp.asarray(self.dmodel.endo_nodes, dtype=jnp.float64)

        if self.n_regimes == 1:
            if isinstance(controls, GArray):
                x_vals = jnp.asarray(controls[0], dtype=jnp.float64)
            else:
                c_arr = jnp.asarray(controls, dtype=jnp.float64)
                x_vals = c_arr[0] if c_arr.ndim == 3 else c_arr
            res = self.F_vmap(s_nodes, x_vals, coeffs)
            if isinstance(controls, GArray):
                out = GArray.zeros_like(controls)
                out[0] = np.asarray(res)
                return out
            return res

        # Multi-regime
        if self.same_inodes and self.F_all_vmap is not None:
            x_vals = jnp.asarray(
                controls.data if isinstance(controls, GArray) else controls,
                dtype=jnp.float64,
            )
            res = self.F_all_vmap(
                s_nodes, x_vals, coeffs, self.m_vals, self.all_inodes, self.all_weights
            )
            if isinstance(controls, GArray):
                out = GArray.zeros_like(controls)
                out.data[...] = np.asarray(res)
                return out
            return res

        out_list = []
        for i in range(self.n_regimes):
            x_i = jnp.asarray(
                controls[i] if isinstance(controls, GArray) else controls[i],
                dtype=jnp.float64,
            )
            res_i = self.regime_vmap[i]["F"](s_nodes, x_i, coeffs)
            out_list.append(res_i)

        if isinstance(controls, GArray):
            out = GArray.zeros_like(controls)
            for i in range(self.n_regimes):
                out[i] = np.asarray(out_list[i])
            return out
        return jnp.stack(out_list, axis=0)

    def dF_1(
        self,
        controls: Union[GArray, np.ndarray, jnp.ndarray],
        coeffs: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Vectorized Jacobian tensor (N, n_x, n_x) across the grid using jax.vmap.

        Mirrors Dolo.jl: Dolo.dF_1(model, controls, φ)
        """
        if coeffs is None:
            coeffs = self.prefilter(controls)

        s_nodes = jnp.asarray(self.dmodel.endo_nodes, dtype=jnp.float64)

        if self.n_regimes == 1:
            if isinstance(controls, GArray):
                x_vals = jnp.asarray(controls[0], dtype=jnp.float64)
            else:
                c_arr = jnp.asarray(controls, dtype=jnp.float64)
                x_vals = c_arr[0] if c_arr.ndim == 3 else c_arr
            return self.dF_vmap(s_nodes, x_vals, coeffs)

        if self.same_inodes and self.dF_all_vmap is not None:
            x_vals = jnp.asarray(
                controls.data if isinstance(controls, GArray) else controls,
                dtype=jnp.float64,
            )
            return self.dF_all_vmap(
                s_nodes, x_vals, coeffs, self.m_vals, self.all_inodes, self.all_weights
            )

        out_list = []
        for i in range(self.n_regimes):
            x_i = jnp.asarray(
                controls[i] if isinstance(controls, GArray) else controls[i],
                dtype=jnp.float64,
            )
            jac_i = self.regime_vmap[i]["dF"](s_nodes, x_i, coeffs)
            out_list.append(jac_i)
        return jnp.stack(out_list, axis=0)

    def F_and_dF_1(
        self,
        controls: Union[GArray, np.ndarray, jnp.ndarray],
        coeffs: Optional[jnp.ndarray] = None,
    ) -> Tuple[Any, Any]:
        """Jointly evaluates residual and Jacobian in a single fused XLA pass."""
        if coeffs is None:
            coeffs = self.prefilter(controls)

        s_nodes = jnp.asarray(self.dmodel.endo_nodes, dtype=jnp.float64)

        if self.n_regimes == 1:
            if isinstance(controls, GArray):
                x_vals = jnp.asarray(controls[0], dtype=jnp.float64)
            else:
                c_arr = jnp.asarray(controls, dtype=jnp.float64)
                x_vals = c_arr[0] if c_arr.ndim == 3 else c_arr
            res, jac = self.F_and_dF_vmap(s_nodes, x_vals, coeffs)
            if isinstance(controls, GArray):
                out_res = GArray.zeros_like(controls)
                out_res[0] = np.asarray(res)
                return out_res, jac
            return res, jac

        if self.same_inodes and self.F_and_dF_all_vmap is not None:
            x_vals = jnp.asarray(
                controls.data if isinstance(controls, GArray) else controls,
                dtype=jnp.float64,
            )
            res, jac = self.F_and_dF_all_vmap(
                s_nodes, x_vals, coeffs, self.m_vals, self.all_inodes, self.all_weights
            )
            if isinstance(controls, GArray):
                out_res = GArray.zeros_like(controls)
                out_res.data[...] = np.asarray(res)
                return out_res, jac
            return res, jac

        res_list = []
        jac_list = []
        for i in range(self.n_regimes):
            x_i = jnp.asarray(
                controls[i] if isinstance(controls, GArray) else controls[i],
                dtype=jnp.float64,
            )
            r_i, j_i = self.regime_vmap[i]["F_and_dF"](s_nodes, x_i, coeffs)
            res_list.append(r_i)
            jac_list.append(j_i)

        if isinstance(controls, GArray):
            out_res = GArray.zeros_like(controls)
            for i in range(self.n_regimes):
                out_res[i] = np.asarray(res_list[i])
            return out_res, jnp.stack(jac_list, axis=0)
        return jnp.stack(res_list, axis=0), jnp.stack(jac_list, axis=0)

    def dF_2(
        self,
        controls: Union[GArray, np.ndarray, jnp.ndarray],
        coeffs: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate derivatives w.r.t future decision rule across the grid.

        Returns
        -------
        F_x : jnp.ndarray
            Shape (K, N, n_x, n_x) if 1 regime, or (n_regimes, K, N, n_x, n_x) if multi-regime.
        S : jnp.ndarray
            Shape (K, N, n_s) if 1 regime, or (n_regimes, K, N, n_s) if multi-regime.
        """
        if coeffs is None:
            coeffs = self.prefilter(controls)

        s_nodes = jnp.asarray(self.dmodel.endo_nodes, dtype=jnp.float64)

        if self._fast is not None:
            if self.n_regimes == 1:
                if isinstance(controls, GArray):
                    x_vals = jnp.asarray(controls[0], dtype=jnp.float64)
                else:
                    c_arr = jnp.asarray(controls, dtype=jnp.float64)
                    x_vals = c_arr[0] if c_arr.ndim == 3 else c_arr
                F_x, S = self._fast.dF_2(x_vals[None], coeffs[None])
                return F_x[0], S[0]
            x_vals = jnp.asarray(
                controls.data if isinstance(controls, GArray) else controls,
                dtype=jnp.float64,
            )
            return self._fast.dF_2(x_vals, coeffs)

        if self.n_regimes == 1:
            if isinstance(controls, GArray):
                x_vals = jnp.asarray(controls[0], dtype=jnp.float64)
            else:
                c_arr = jnp.asarray(controls, dtype=jnp.float64)
                x_vals = c_arr[0] if c_arr.ndim == 3 else c_arr
            m_val, inodes, weights = self.regime_data[0]
            n_inodes = len(weights)
            F_x_list = []
            S_list = []
            for k in range(n_inodes):
                M = inodes[k]
                w = weights[k]
                fx_k, s_k = self.dF_2_branch_vmap(s_nodes, x_vals, coeffs, M, w)
                F_x_list.append(fx_k)
                S_list.append(s_k)
            return jnp.stack(F_x_list, axis=0), jnp.stack(S_list, axis=0)

        if self.same_inodes and self.dF_2_all_vmap is not None:
            x_vals = jnp.asarray(
                controls.data if isinstance(controls, GArray) else controls,
                dtype=jnp.float64,
            )
            fx, s = self.dF_2_all_vmap(
                s_nodes, x_vals, coeffs, self.m_vals, self.all_inodes, self.all_weights
            )
            return jnp.swapaxes(fx, 1, 2), jnp.swapaxes(s, 1, 2)

        # General multi-regime fallback
        Fx_regimes = []
        S_regimes = []
        for i in range(self.n_regimes):
            m_val_i, inodes_i, weights_i, bv_i = self.dF_2_regimes[i]
            x_i = jnp.asarray(
                controls[i] if isinstance(controls, GArray) else controls[i],
                dtype=jnp.float64,
            )
            fx_list = []
            s_list = []
            for k in range(len(weights_i)):
                M = inodes_i[k]
                w = weights_i[k]
                fx_k, s_k = bv_i(s_nodes, x_i, coeffs[k], M, w)
                fx_list.append(fx_k)
                s_list.append(s_k)
            Fx_regimes.append(jnp.stack(fx_list, axis=0))
            S_regimes.append(jnp.stack(s_list, axis=0))
        return jnp.stack(Fx_regimes, axis=0), jnp.stack(S_regimes, axis=0)


def compile_residual(
    model: Any,
    interp_method: str = "cubic",
    diff_mode: str = "fwd",
    grid_options: Optional[Dict[str, Any]] = None,
    dprocess_options: Optional[Dict[str, Any]] = None,
) -> CompiledResidual:
    """Compile model residual function with JAX (Dolo.jl Dolo.F style).

    Parameters
    ----------
    model : Model, ModelV2, or DiscretizedModel
        The economic model to compile.
    interp_method : str
        Interpolation method for policy evaluation ('cubic' or 'linear').
    diff_mode : str
        Autodiff mode for Jacobian: 'fwd' (jax.jacfwd) or 'rev' (jax.jacrev).
    grid_options : dict, optional
        Grid discretization options (if model is not already discretized).
    dprocess_options : dict, optional
        Process discretization options (if model is not already discretized).

    Returns
    -------
    CompiledResidual
        Object containing point and vmapped residual and Jacobian functions.
    """
    if isinstance(model, str):
        from dolo.compiler.model_import import yaml_import_v2

        m_v2 = yaml_import_v2(model)
        dmodel = DiscretizedModel(
            m_v2, grid_options=grid_options, dprocess_options=dprocess_options
        )
    elif isinstance(model, DiscretizedModel):
        if hasattr(model.f, "point_jnp"):
            dmodel = model
        else:
            # Legacy DiscretizedModel: try to upgrade to ModelV2 if filename is known
            upgraded = False
            if hasattr(model.model, "filename") and model.model.filename not in (
                None,
                "<string>",
            ):
                try:
                    from dolo.compiler.model_import import yaml_import_v2

                    m_v2 = yaml_import_v2(model.model.filename)
                    dmodel = DiscretizedModel(
                        m_v2,
                        grid_options=grid_options,
                        dprocess_options=dprocess_options,
                    )
                    upgraded = True
                except Exception:
                    pass
            if not upgraded:
                dmodel = model
    else:
        from dolo.compiler.model_v2 import ModelV2

        if (
            not isinstance(model, ModelV2)
            and hasattr(model, "filename")
            and model.filename not in (None, "<string>")
        ):
            try:
                from dolo.compiler.model_import import yaml_import_v2

                m_v2 = yaml_import_v2(model.filename)
                dmodel = DiscretizedModel(
                    m_v2, grid_options=grid_options, dprocess_options=dprocess_options
                )
            except Exception:
                dmodel = DiscretizedModel(
                    model, grid_options=grid_options, dprocess_options=dprocess_options
                )
        else:
            dmodel = DiscretizedModel(
                model, grid_options=grid_options, dprocess_options=dprocess_options
            )

    return CompiledResidual(dmodel, interp_method=interp_method, diff_mode=diff_mode)
