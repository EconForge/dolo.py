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


def _get_raw_jnp_point_fn(fun: Any) -> Callable:
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
    # Fallback to fun directly
    return fun


class CompiledResidual:
    """Compiled JAX residual operator mirroring Dolo.jl Dolo.F / Dolo.dF_1."""

    def __init__(
        self,
        dmodel: DiscretizedModel,
        interp_method: str = "cubic",
        diff_mode: str = "fwd",
    ):
        self.dmodel = dmodel
        self.interp_method = interp_method
        self.diff_mode = diff_mode

        # Model functions
        self.f_pt = _get_raw_jnp_point_fn(dmodel.f)
        self.g_pt = _get_raw_jnp_point_fn(dmodel.g)
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

        # Build JAX kernels
        self._build_kernels()

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

        if self.n_regimes == 1:
            def dF_pt(s, x, coeffs):
                return diff_fn(lambda u: F_pt(s, u, coeffs))(x)

            def F_and_dF_pt(s, x, coeffs):
                return F_pt(s, x, coeffs), dF_pt(s, x, coeffs)

            self.F_pt_jit = jax.jit(F_pt)
            self.dF_pt_jit = jax.jit(dF_pt)
            self.F_and_dF_pt_jit = jax.jit(F_and_dF_pt)

            # Vectorized across continuous state grid using jax.vmap
            self.F_vmap = jax.jit(jax.vmap(F_pt, in_axes=(0, 0, None)))
            self.dF_vmap = jax.jit(jax.vmap(dF_pt, in_axes=(0, 0, None)))
            self.F_and_dF_vmap = jax.jit(jax.vmap(F_and_dF_pt, in_axes=(0, 0, None)))

        else:
            # Multi-regime vectorized kernels
            self.regime_vmap = []
            for i_m in range(self.n_regimes):
                F_regime = functools.partial(F_pt, i_m=i_m)
                dF_regime = lambda s, x, coeffs: diff_fn(lambda u: F_regime(s, u, coeffs))(x)
                F_and_dF_regime = lambda s, x, coeffs: (F_regime(s, x, coeffs), dF_regime(s, x, coeffs))

                self.regime_vmap.append({
                    "F": jax.jit(jax.vmap(F_regime, in_axes=(0, 0, None))),
                    "dF": jax.jit(jax.vmap(dF_regime, in_axes=(0, 0, None))),
                    "F_and_dF": jax.jit(jax.vmap(F_and_dF_regime, in_axes=(0, 0, None))),
                })

    def prefilter(self, controls: Union[GArray, np.ndarray, jnp.ndarray]) -> jnp.ndarray:
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
        out_list = []
        for i in range(self.n_regimes):
            x_i = jnp.asarray(controls[i], dtype=jnp.float64)
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

        out_list = []
        for i in range(self.n_regimes):
            x_i = jnp.asarray(controls[i], dtype=jnp.float64)
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

        res_list = []
        jac_list = []
        for i in range(self.n_regimes):
            x_i = jnp.asarray(controls[i], dtype=jnp.float64)
            r_i, j_i = self.regime_vmap[i]["F_and_dF"](s_nodes, x_i, coeffs)
            res_list.append(r_i)
            jac_list.append(j_i)

        if isinstance(controls, GArray):
            out_res = GArray.zeros_like(controls)
            for i in range(self.n_regimes):
                out_res[i] = np.asarray(res_list[i])
            return out_res, jnp.stack(jac_list, axis=0)
        return jnp.stack(res_list, axis=0), jnp.stack(jac_list, axis=0)


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
        dmodel = DiscretizedModel(m_v2, grid_options=grid_options, dprocess_options=dprocess_options)
    elif isinstance(model, DiscretizedModel):
        if hasattr(model.f, "point_jnp"):
            dmodel = model
        else:
            # Legacy DiscretizedModel: try to upgrade to ModelV2 if filename is known
            upgraded = False
            if hasattr(model.model, "filename") and model.model.filename not in (None, "<string>"):
                try:
                    from dolo.compiler.model_import import yaml_import_v2
                    m_v2 = yaml_import_v2(model.model.filename)
                    dmodel = DiscretizedModel(m_v2, grid_options=grid_options, dprocess_options=dprocess_options)
                    upgraded = True
                except Exception:
                    pass
            if not upgraded:
                dmodel = model
    else:
        from dolo.compiler.model_v2 import ModelV2
        if not isinstance(model, ModelV2) and hasattr(model, "filename") and model.filename not in (None, "<string>"):
            try:
                from dolo.compiler.model_import import yaml_import_v2
                m_v2 = yaml_import_v2(model.filename)
                dmodel = DiscretizedModel(m_v2, grid_options=grid_options, dprocess_options=dprocess_options)
            except Exception:
                dmodel = DiscretizedModel(model, grid_options=grid_options, dprocess_options=dprocess_options)
        else:
            dmodel = DiscretizedModel(model, grid_options=grid_options, dprocess_options=dprocess_options)

    return CompiledResidual(dmodel, interp_method=interp_method, diff_mode=diff_mode)
