"""Residual and Jacobian evaluation API for Dolo economic models.

Mirrors the Julia Dolo.jl interface:
- F(model, controls, φ=None, *, engine="auto", ...)
- dF_1(model, controls, φ=None, *, engine="auto", ...)
- F_and_dF_1(model, controls, φ=None, *, engine="auto", ...)

Supports engines:
- "auto": chooses JAX when supported, falls back to vectorized.
- "jax": compiled with JAX, vectorized with vmap (default device).
- "gpu" / "cuda": JAX compiled and executed on NVIDIA GPU via CUDA.
- "cpu": JAX compiled and executed explicitly on CPU.
- "vectorized" / "numpy": pure vectorized NumPy evaluation via Tau.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np

from dolo.numeric.garray import GArray, QP
from dolo.numeric.decision_rule import DecisionRule
from dolo.numeric.discretized_model import DiscretizedModel
from dolo.numeric.residual_jax import compile_residual, CompiledResidual
from dolo.numeric.tau import euler_residuals_tau
from dolo.numeric.optimize.newton import SerialDifferentiableFunction
from scipy.sparse.linalg import LinearOperator


def _resolve_engine(
    dmodel: DiscretizedModel,
    engine: str = "auto",
    backend: Optional[str] = None,
) -> str:
    """Resolve and validate the execution engine."""
    eng = (backend or engine).lower()
    if eng == "auto":
        # Check if JAX is available and model equations are supported
        try:
            import jax

            eng = "jax"
        except ImportError:
            eng = "vectorized"
    elif eng in ("cuda",):
        eng = "gpu"
    elif eng in ("numpy",):
        eng = "vectorized"

    valid = ("auto", "jax", "gpu", "cpu", "vectorized")
    if eng not in valid:
        raise ValueError(f"Unknown engine '{eng}'. Valid options are: {valid}")
    return eng


def _get_or_compile_jax_residual(
    dmodel: DiscretizedModel,
    interp_method: str = "cubic",
    diff_mode: str = "fwd",
) -> CompiledResidual:
    """Retrieve cached CompiledResidual or compile a new one."""
    cache_key = f"_compiled_residual_jax_{interp_method}_{diff_mode}"
    if not hasattr(dmodel, cache_key):
        cr = compile_residual(dmodel, interp_method=interp_method, diff_mode=diff_mode)
        setattr(dmodel, cache_key, cr)
    return getattr(dmodel, cache_key)


def F(
    model: Any,
    *args,
    engine: str = "auto",
    backend: Optional[str] = None,
    interp_method: str = "cubic",
    device: Optional[Any] = None,
    **kwargs,
) -> Union[GArray, np.ndarray, Any]:
    """Evaluate dynamic Euler equation residuals across grid points or at a single point.

    Usage
    -----
    1. Grid-level evaluation (mirroring Julia Dolo.jl):
       F(model, controls, dr=None, engine="auto")
       F(model, x0, x0)  # dr initialized to initial guess x0
       F(model, x0)      # dr defaults to x0

    2. Point-level evaluation:
       F(model, s, x, dr)

    Parameters
    ----------
    model : Model, ModelV2, or DiscretizedModel
        The economic model.
    controls : GArray, np.ndarray, or jnp.ndarray
        Control policy values across the grid (or point state s if point-call).
    dr : DecisionRule, GArray, or array, optional
        Decision rule (policy) to evaluate next-period controls φ(S).
        If omitted or equal to controls, fits φ on `controls`.
    engine / backend : str ("auto", "jax", "gpu", "cpu", "vectorized")
        Computational backend to execute the residual evaluation.
    interp_method : str ("cubic" or "linear")
        Interpolation method for decision rule φ.
    device : jax.Device, optional
        Target JAX device (e.g. jax.devices('gpu')[0]).
    """
    if len(args) == 0:
        raise ValueError(
            "F requires at least 2 arguments: (model, controls) or (model, s, x, dr)"
        )

    # 1. Point-level dispatch: F(model, s, x, dr)
    if len(args) >= 2 and (
        isinstance(args[0], (QP, tuple))
        or (isinstance(args[0], np.ndarray) and args[0].ndim == 1)
    ):
        s, x = args[0], args[1]
        dr = args[2] if len(args) > 2 else None
        dmodel = (
            model
            if isinstance(model, DiscretizedModel)
            else DiscretizedModel(model, **kwargs)
        )
        eng = _resolve_engine(dmodel, engine, backend)
        if eng in ("jax", "gpu", "cpu"):
            cr = _get_or_compile_jax_residual(dmodel, interp_method=interp_method)
            coeffs = cr.prefilter(dr if dr is not None else x)
            return cr.F_point(s if not isinstance(s, QP) else s.val, x, coeffs)
        else:
            # Vectorized / point transition
            dr_obj = (
                dr
                if isinstance(dr, DecisionRule)
                else dmodel.decision_rule(dr, interp_method=interp_method)
            )
            s_qp = s if isinstance(s, QP) else QP(0, s)
            res = np.zeros_like(x)
            n_m = (
                len(dmodel.model.symbols["exogenous"])
                if "exogenous" in dmodel.model.symbols
                else len(dmodel.model.calibration.get("exogenous", []))
            )
            for prob, S in dmodel.tau.point_transitions(s_qp, x):
                rr = dmodel.arbitrage(
                    s_qp.val[:n_m],
                    s_qp.val[n_m:],
                    x,
                    S.val[:n_m],
                    S.val[n_m:],
                    dr_obj(S),
                )
                res += prob * rr
            return res

    # 2. Grid-level dispatch: F(model, controls, dr=None)
    controls = args[0]
    dr = args[1] if len(args) > 1 else None

    # Resolve discretized model
    if isinstance(model, DiscretizedModel):
        dmodel = model
    else:
        grid_options = kwargs.pop("grid_options", None)
        dprocess_options = kwargs.pop("dprocess_options", {})
        dmodel = DiscretizedModel(
            model,
            grid_options=grid_options,
            dprocess_options=dprocess_options,
            **kwargs,
        )

    eng = _resolve_engine(dmodel, engine, backend)

    if eng in ("jax", "gpu", "cpu"):
        import jax
        import jax.numpy as jnp

        target_dev = device
        if target_dev is None:
            if eng == "gpu":
                gpus = jax.devices("gpu")
                if not gpus:
                    raise RuntimeError(
                        "GPU requested (engine='gpu'), but no CUDA device was found."
                    )
                target_dev = gpus[0]
            elif eng == "cpu":
                target_dev = jax.devices("cpu")[0]

        cr = _get_or_compile_jax_residual(dmodel, interp_method=interp_method)
        policy_for_coeffs = dr if dr is not None else controls
        coeffs = cr.prefilter(policy_for_coeffs)

        s_nodes = jnp.asarray(dmodel.endo_nodes, dtype=jnp.float64)
        if isinstance(controls, GArray):
            x_vals = jnp.asarray(
                controls[0] if dmodel.n_regimes == 1 else controls.data,
                dtype=jnp.float64,
            )
        else:
            c_arr = jnp.asarray(controls, dtype=jnp.float64)
            x_vals = c_arr[0] if (dmodel.n_regimes == 1 and c_arr.ndim == 3) else c_arr

        if target_dev is not None:
            s_nodes = jax.device_put(s_nodes, target_dev)
            x_vals = jax.device_put(x_vals, target_dev)
            coeffs = jax.device_put(coeffs, target_dev)

        if dmodel.n_regimes == 1:
            res_jax = cr.F_vmap(s_nodes, x_vals, coeffs)
            if target_dev is not None:
                res_jax.block_until_ready()
            if isinstance(controls, GArray):
                out = GArray.zeros_like(controls)
                out[0] = np.asarray(res_jax)
                return out
            return res_jax
        else:
            return cr.F(controls, coeffs)

    elif eng == "vectorized":
        if isinstance(dr, DecisionRule):
            dr_obj = dr
        elif dr is None or dr is controls or isinstance(dr, (GArray, np.ndarray)):
            policy_val = controls if dr is None or dr is controls else dr
            if not isinstance(policy_val, GArray):
                policy_ga = dmodel.initial_guess()
                if policy_ga.data.shape == policy_val.shape:
                    policy_ga.data[...] = policy_val
                elif policy_ga[0].shape == policy_val.shape:
                    policy_ga[0] = policy_val
                policy_val = policy_ga
            dr_obj = dmodel.decision_rule(policy_val, interp_method=interp_method)
        else:
            dr_obj = dr

        controls_ga = (
            controls if isinstance(controls, GArray) else dmodel.initial_guess()
        )
        if not isinstance(controls, GArray):
            if controls_ga.data.shape == controls.shape:
                controls_ga.data[...] = controls
            elif controls_ga[0].shape == controls.shape:
                controls_ga[0] = controls

        return euler_residuals_tau(dmodel, controls_ga, dr_obj)

    raise ValueError(f"Unsupported engine: {eng}")


def dF_1(
    model: Any,
    *args,
    engine: str = "auto",
    backend: Optional[str] = None,
    interp_method: str = "cubic",
    diff_mode: str = "fwd",
    device: Optional[Any] = None,
    **kwargs,
) -> Any:
    """Evaluate Jacobian of dynamic Euler residuals w.r.t controls across grid points.

    Mirrors Julia Dolo.jl: dF_1(model, controls, φ=None)

    Parameters
    ----------
    model : Model, ModelV2, or DiscretizedModel
        The economic model.
    controls : GArray, np.ndarray, or jnp.ndarray
        Control policy values across the grid.
    dr : DecisionRule, GArray, or array, optional
        Decision rule (policy). Defaults to `controls` if omitted.
    engine / backend : str ("auto", "jax", "gpu", "cpu", "vectorized")
        Backend engine. JAX computes exact analytical autodiff derivatives.
    interp_method : str ("cubic" or "linear")
        Interpolation method.
    diff_mode : str ("fwd" or "rev")
        Autodiff mode for JAX ('fwd' for jacfwd, 'rev' for jacrev).
    """
    if len(args) == 0:
        raise ValueError("dF_1 requires at least 1 positional argument: controls")

    controls = args[0]
    dr = args[1] if len(args) > 1 else None

    if isinstance(model, DiscretizedModel):
        dmodel = model
    else:
        grid_options = kwargs.pop("grid_options", None)
        dprocess_options = kwargs.pop("dprocess_options", {})
        dmodel = DiscretizedModel(
            model,
            grid_options=grid_options,
            dprocess_options=dprocess_options,
            **kwargs,
        )

    eng = _resolve_engine(dmodel, engine, backend)

    if eng in ("jax", "gpu", "cpu"):
        import jax
        import jax.numpy as jnp

        target_dev = device
        if target_dev is None:
            if eng == "gpu":
                gpus = jax.devices("gpu")
                if not gpus:
                    raise RuntimeError(
                        "GPU requested (engine='gpu'), but no CUDA device was found."
                    )
                target_dev = gpus[0]
            elif eng == "cpu":
                target_dev = jax.devices("cpu")[0]

        cr = _get_or_compile_jax_residual(
            dmodel, interp_method=interp_method, diff_mode=diff_mode
        )
        policy_for_coeffs = dr if dr is not None else controls
        coeffs = cr.prefilter(policy_for_coeffs)

        s_nodes = jnp.asarray(dmodel.endo_nodes, dtype=jnp.float64)
        if isinstance(controls, GArray):
            x_vals = jnp.asarray(
                controls[0] if dmodel.n_regimes == 1 else controls.data,
                dtype=jnp.float64,
            )
        else:
            c_arr = jnp.asarray(controls, dtype=jnp.float64)
            x_vals = c_arr[0] if (dmodel.n_regimes == 1 and c_arr.ndim == 3) else c_arr

        if target_dev is not None:
            s_nodes = jax.device_put(s_nodes, target_dev)
            x_vals = jax.device_put(x_vals, target_dev)
            coeffs = jax.device_put(coeffs, target_dev)

        if dmodel.n_regimes == 1:
            jac = cr.dF_vmap(s_nodes, x_vals, coeffs)
        else:
            jac = cr.dF_1(x_vals, coeffs)
        if target_dev is not None:
            jac.block_until_ready()
        return jac

    elif eng == "vectorized":
        # Finite-difference derivative using SerialDifferentiableFunction
        controls_ga = (
            controls if isinstance(controls, GArray) else dmodel.initial_guess()
        )
        if not isinstance(controls, GArray):
            if controls_ga.data.shape == controls.shape:
                controls_ga.data[...] = controls
            elif controls_ga[0].shape == controls.shape:
                controls_ga[0] = controls

        if isinstance(dr, DecisionRule):
            dr_obj = dr
        else:
            dr_obj = dmodel.decision_rule(
                controls_ga if dr is None else dr, interp_method=interp_method
            )

        n_x = len(dmodel.model.calibration["controls"])
        fn = (
            lambda x_val: euler_residuals_tau(
                dmodel, controls_ga.unravel(x_val), dr_obj
            )
            .ravel()
            .reshape((-1, n_x))
        )
        dfn = SerialDifferentiableFunction(fn)
        _, jacs = dfn(controls_ga.ravel().reshape((-1, n_x)))
        return jacs

    raise ValueError(f"Unsupported engine: {eng}")


def F_and_dF_1(
    model: Any,
    *args,
    engine: str = "auto",
    backend: Optional[str] = None,
    interp_method: str = "cubic",
    diff_mode: str = "fwd",
    device: Optional[Any] = None,
    **kwargs,
) -> Tuple[Any, Any]:
    """Simultaneously evaluate residuals and Jacobian w.r.t controls in a single pass.

    Returns
    -------
    (res, dres) : Tuple of residuals and Jacobian tensor
    """
    if len(args) == 0:
        raise ValueError("F_and_dF_1 requires at least 1 positional argument: controls")

    controls = args[0]
    dr = args[1] if len(args) > 1 else None

    if isinstance(model, DiscretizedModel):
        dmodel = model
    else:
        grid_options = kwargs.pop("grid_options", None)
        dprocess_options = kwargs.pop("dprocess_options", {})
        dmodel = DiscretizedModel(
            model,
            grid_options=grid_options,
            dprocess_options=dprocess_options,
            **kwargs,
        )

    eng = _resolve_engine(dmodel, engine, backend)

    if eng in ("jax", "gpu", "cpu"):
        import jax
        import jax.numpy as jnp

        target_dev = device
        if target_dev is None:
            if eng == "gpu":
                gpus = jax.devices("gpu")
                if not gpus:
                    raise RuntimeError(
                        "GPU requested (engine='gpu'), but no CUDA device was found."
                    )
                target_dev = gpus[0]
            elif eng == "cpu":
                target_dev = jax.devices("cpu")[0]

        cr = _get_or_compile_jax_residual(
            dmodel, interp_method=interp_method, diff_mode=diff_mode
        )
        policy_for_coeffs = dr if dr is not None else controls
        coeffs = cr.prefilter(policy_for_coeffs)

        s_nodes = jnp.asarray(dmodel.endo_nodes, dtype=jnp.float64)
        if isinstance(controls, GArray):
            x_vals = jnp.asarray(
                controls[0] if dmodel.n_regimes == 1 else controls.data,
                dtype=jnp.float64,
            )
        else:
            c_arr = jnp.asarray(controls, dtype=jnp.float64)
            x_vals = c_arr[0] if (dmodel.n_regimes == 1 and c_arr.ndim == 3) else c_arr

        if target_dev is not None:
            s_nodes = jax.device_put(s_nodes, target_dev)
            x_vals = jax.device_put(x_vals, target_dev)
            coeffs = jax.device_put(coeffs, target_dev)

        if dmodel.n_regimes == 1:
            res, dres = cr.F_and_dF_vmap(s_nodes, x_vals, coeffs)
            if target_dev is not None:
                res.block_until_ready()
                dres.block_until_ready()
            if isinstance(controls, GArray):
                out = GArray.zeros_like(controls)
                out[0] = np.asarray(res)
                return out, dres
            return res, dres
        else:
            res, dres = cr.F_and_dF_1(x_vals, coeffs)
            if target_dev is not None:
                res.block_until_ready()
                dres.block_until_ready()
            if isinstance(controls, GArray):
                out = GArray.zeros_like(controls)
                out.data[...] = np.asarray(res)
                return out, dres
            return res, dres

    elif eng == "vectorized":
        res = F(
            dmodel,
            controls,
            dr,
            engine="vectorized",
            interp_method=interp_method,
            **kwargs,
        )
        dres = dF_1(
            dmodel,
            controls,
            dr,
            engine="vectorized",
            interp_method=interp_method,
            **kwargs,
        )
        return res, dres

    raise ValueError(f"Unsupported engine: {eng}")


class PolicyLinearOperator(LinearOperator):
    r"""Linear operator representing dF_2 = \partial F / \partial \varphi.

    Acts on policy variations v defined on the continuous state grid:
        (L @ v)[n] = \sum_k F_{x, k}[n] \cdot v(S_k[n])
    where v(S_k[n]) is the interpolation of v at the next-period state S_k[n].

    Mirrors Dolo.jl's LL / LinearOperator struct.
    """

    def __init__(
        self,
        dmodel: DiscretizedModel,
        F_x: Any,
        S: Any,
        engine: str = "jax",
        interp_method: str = "cubic",
        device: Optional[Any] = None,
        cr: Optional[CompiledResidual] = None,
    ):
        self.dmodel = dmodel
        self.F_x = F_x
        self.S = S
        self.engine = engine
        self.interp_method = interp_method
        self.device = device
        self.cr = cr

        self.n_regimes = dmodel.n_regimes
        self.endo_grid = dmodel.endo_grid
        self.N = self.endo_grid.nodes.shape[0]
        self.n_x = len(dmodel.model.calibration["controls"])
        self.n_s = len(dmodel.model.symbols["states"])
        self.grid_shape = tuple(int(x) for x in self.endo_grid.n)

        self.dim = self.n_regimes * self.N * self.n_x
        super().__init__(dtype=np.float64, shape=(self.dim, self.dim))

        if self.engine in ("jax", "gpu", "cpu") and self.cr is not None:
            import jax
            import jax.numpy as jnp
            import functools
            from dolo.numeric.interpolation_jax import (
                prefilter_cubic_jax,
                eval_linear_point_jax,
                _eval_cubic_batch_jit,
            )

            a, b, n = self.cr.a, self.cr.b, self.cr.n
            grid_shape = self.grid_shape
            interp_method = self.interp_method
            n_regimes = self.n_regimes

            if n_regimes == 1:

                def _apply_kernel(v_2d, F_x, S):
                    v_reshaped = v_2d.reshape(grid_shape + (-1,))
                    if interp_method == "cubic":
                        v_coeffs = prefilter_cubic_jax(grid_shape, v_reshaped)

                        def eval_branch(fx_k, s_k):
                            v_S_k = _eval_cubic_batch_jit(v_coeffs, s_k, a, b, n)
                            return jnp.einsum("nij,nj->ni", fx_k, v_S_k)

                    else:
                        v_coeffs = v_reshaped

                        def eval_branch(fx_k, s_k):
                            v_S_k = jax.vmap(
                                eval_linear_point_jax,
                                in_axes=(None, 0, None, None, None),
                            )(v_coeffs, s_k, a, b, n)
                            return jnp.einsum("nij,nj->ni", fx_k, v_S_k)

                    branch_results = jax.vmap(eval_branch)(F_x, S)
                    return jnp.sum(branch_results, axis=0)

                self._apply_jit = jax.jit(_apply_kernel)
            else:

                def _apply_kernel_multi(v_3d, F_x, S):
                    v_reshaped = v_3d.reshape((n_regimes,) + grid_shape + (-1,))
                    if interp_method == "cubic":
                        v_coeffs = jax.vmap(
                            functools.partial(prefilter_cubic_jax, grid_shape)
                        )(v_reshaped)
                    else:
                        v_coeffs = v_reshaped

                    n_inodes = F_x.shape[1]
                    out = jnp.zeros_like(v_3d)
                    for k in range(n_inodes):
                        target_regime = k
                        if interp_method == "cubic":
                            v_S_k = jax.vmap(
                                lambda s_ik: _eval_cubic_batch_jit(
                                    v_coeffs[target_regime], s_ik, a, b, n
                                )
                            )(S[:, k])
                        else:
                            v_S_k = jax.vmap(
                                jax.vmap(
                                    eval_linear_point_jax,
                                    in_axes=(None, 0, None, None, None),
                                ),
                                in_axes=(None, 0, None, None, None),
                            )(v_coeffs[target_regime], S[:, k], a, b, n)
                        out = out + jnp.einsum("rnij,rnj->rni", F_x[:, k], v_S_k)
                    return out

                self._apply_jit = jax.jit(_apply_kernel_multi)
        else:
            self._apply_jit = None

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        if self.n_regimes == 1:
            x_reshaped = x.reshape((self.N, self.n_x))
        else:
            x_reshaped = x.reshape((self.n_regimes, self.N, self.n_x))
        out = self.apply(x_reshaped)
        return np.asarray(out).ravel()

    def apply(
        self, v: Union[GArray, np.ndarray, Any]
    ) -> Union[GArray, np.ndarray, Any]:
        """Apply linear operator to a policy variation v on the grid."""
        if self.engine in ("jax", "gpu", "cpu"):
            import jax
            import jax.numpy as jnp
            from dolo.numeric.interpolation_jax import (
                prefilter_cubic_jax,
                eval_cubic_jax,
                eval_linear_jax,
                _eval_cubic_batch_jit,
            )

            is_garray = isinstance(v, GArray)
            v_data = v.data if is_garray else v
            v_arr = jnp.asarray(v_data, dtype=jnp.float64)
            orig_ndim = v_arr.ndim

            if self.n_regimes == 1:
                if v_arr.ndim == 1:
                    v_arr = v_arr.reshape((self.N, self.n_x))
                elif v_arr.ndim == 3 and v_arr.shape[0] == 1:
                    v_arr = v_arr[0]
            else:
                if v_arr.ndim == 1:
                    v_arr = v_arr.reshape((self.n_regimes, self.N, self.n_x))

            if self.device is not None:
                v_arr = jax.device_put(v_arr, self.device)

            if self._apply_jit is not None:
                out = self._apply_jit(v_arr, self.F_x, self.S)
            else:
                if self.n_regimes == 1:
                    v_reshaped = v_arr.reshape(self.grid_shape + (-1,))
                    if self.interp_method == "cubic":
                        v_coeffs = prefilter_cubic_jax(self.endo_grid, v_reshaped)
                        eval_fn = eval_cubic_jax
                    else:
                        v_coeffs = v_reshaped
                        eval_fn = eval_linear_jax

                    out = jnp.zeros_like(v_arr)
                    n_branches = self.F_x.shape[0]
                    for k in range(n_branches):
                        fx_k = self.F_x[k]
                        s_k = self.S[k]
                        v_S_k = eval_fn(self.endo_grid, v_coeffs, s_k)
                        out = out + jnp.einsum("nij,nj->ni", fx_k, v_S_k)
                else:
                    v_reshaped = v_arr.reshape(
                        (self.n_regimes,) + self.grid_shape + (-1,)
                    )
                    if self.interp_method == "cubic":
                        v_coeffs = jnp.stack(
                            [
                                prefilter_cubic_jax(self.grid_shape, v_reshaped[i])
                                for i in range(self.n_regimes)
                            ],
                            axis=0,
                        )
                    else:
                        v_coeffs = v_reshaped
                    out = jnp.zeros_like(v_arr)
                    n_branches = self.F_x.shape[1]
                    for k in range(n_branches):
                        for i in range(self.n_regimes):
                            v_S_ik = _eval_cubic_batch_jit(
                                v_coeffs[k],
                                self.S[i, k],
                                self.cr.a,
                                self.cr.b,
                                self.cr.n,
                            )
                            out = out.at[i].add(
                                jnp.einsum("nij,nj->ni", self.F_x[i, k], v_S_ik)
                            )

            if self.device is not None:
                out.block_until_ready()

            if is_garray:
                res = GArray.zeros_like(v)
                res.data[...] = np.asarray(out)
                return res
            elif isinstance(v, np.ndarray):
                if orig_ndim == 1:
                    return np.asarray(out).ravel()
                return np.asarray(out)
            return out

        else:
            # Vectorized NumPy evaluation
            is_garray = isinstance(v, GArray)
            v_arr = np.asarray(v.data if is_garray else v, dtype=np.float64)
            orig_ndim = v_arr.ndim

            if self.n_regimes == 1:
                if v_arr.ndim == 3 and v_arr.shape[0] == 1:
                    v_2d = v_arr[0]
                elif v_arr.ndim == 1:
                    v_2d = v_arr.reshape((self.N, self.n_x))
                else:
                    v_2d = v_arr

                policy_ga = self.dmodel.initial_guess()
                if policy_ga.data.shape == v_2d.shape:
                    policy_ga.data[...] = v_2d
                else:
                    policy_ga[0] = v_2d
                dr_v = self.dmodel.decision_rule(
                    policy_ga, interp_method=self.interp_method
                )

                out = np.zeros_like(v_2d)
                n_branches = self.F_x.shape[0]
                for k in range(n_branches):
                    fx_k = np.asarray(self.F_x[k])
                    s_k = np.asarray(self.S[k])
                    v_S_k = dr_v.eval_ijs(0, k, s_k)
                    out += np.einsum("nij,nj->ni", fx_k, v_S_k)

                if is_garray:
                    res = GArray.zeros_like(v)
                    res[0] = out
                    return res
                elif orig_ndim == 1:
                    return out.ravel()
                return out
            else:
                if v_arr.ndim == 1:
                    v_3d = v_arr.reshape((self.n_regimes, self.N, self.n_x))
                else:
                    v_3d = v_arr

                policy_ga = self.dmodel.initial_guess()
                policy_ga.data[...] = v_3d
                dr_v = self.dmodel.decision_rule(
                    policy_ga, interp_method=self.interp_method
                )

                out = np.zeros_like(v_3d)
                n_branches = self.F_x.shape[1]
                for i in range(self.n_regimes):
                    for k in range(n_branches):
                        fx_k = np.asarray(self.F_x[i, k])
                        s_k = np.asarray(self.S[i, k])
                        v_S_k = dr_v.eval_ijs(i, k, s_k)
                        out[i] += np.einsum("nij,nj->ni", fx_k, v_S_k)

                if is_garray:
                    res = GArray.zeros_like(v)
                    res.data[...] = out
                    return res
                elif orig_ndim == 1:
                    return out.ravel()
                return out

    def __call__(self, v: Any) -> Any:
        return self.apply(v)

    def __matmul__(self, v: Any) -> Any:
        return self.apply(v)

    def __mul__(self, alpha: Union[float, int]) -> PolicyLinearOperator:
        return PolicyLinearOperator(
            self.dmodel,
            self.F_x * alpha,
            self.S,
            engine=self.engine,
            interp_method=self.interp_method,
            device=self.device,
            cr=self.cr,
        )

    def __rmul__(self, alpha: Union[float, int]) -> PolicyLinearOperator:
        return self.__mul__(alpha)

    def preconditioned(self, J: Any) -> PolicyLinearOperator:
        """Return preconditioned operator J^{-1} L, where J = dF_1 is block-diagonal."""
        if self.engine in ("jax", "gpu", "cpu"):
            import jax
            import jax.numpy as jnp

            J_arr = jnp.asarray(
                J.data if isinstance(J, GArray) else J, dtype=jnp.float64
            )
            if self.device is not None:
                J_arr = jax.device_put(J_arr, self.device)
            if self.n_regimes == 1:
                if J_arr.ndim == 4 and J_arr.shape[0] == 1:
                    J_arr = J_arr[0]
                new_Fx = jnp.stack(
                    [
                        jnp.linalg.solve(J_arr, self.F_x[k])
                        for k in range(self.F_x.shape[0])
                    ],
                    axis=0,
                )
            else:
                if J_arr.ndim == 3:
                    J_arr = J_arr.reshape((self.n_regimes, self.N, self.n_x, self.n_x))
                new_Fx = jnp.linalg.solve(J_arr[:, None, :, :, :], self.F_x)
        else:
            J_arr = np.asarray(J.data if isinstance(J, GArray) else J, dtype=np.float64)
            if self.n_regimes == 1:
                if J_arr.ndim == 4 and J_arr.shape[0] == 1:
                    J_arr = J_arr[0]
                new_Fx = np.stack(
                    [
                        np.linalg.solve(J_arr, np.asarray(self.F_x[k]))
                        for k in range(self.F_x.shape[0])
                    ],
                    axis=0,
                )
            else:
                if J_arr.ndim == 3:
                    J_arr = J_arr.reshape((self.n_regimes, self.N, self.n_x, self.n_x))
                new_Fx = np.linalg.solve(J_arr[:, None, :, :, :], np.asarray(self.F_x))

        return PolicyLinearOperator(
            self.dmodel,
            new_Fx,
            self.S,
            engine=self.engine,
            interp_method=self.interp_method,
            device=self.device,
            cr=self.cr,
        )

    def as_matrix(self) -> np.ndarray:
        """Compute explicit (dim, dim) dense matrix representation."""
        dim = self.dim
        mat = np.zeros((dim, dim))
        e = np.zeros(dim)
        for i in range(dim):
            if i > 0:
                e[i - 1] = 0.0
            e[i] = 1.0
            mat[:, i] = self._matvec(e)
        return mat

    def solve(
        self,
        b: Any,
        linsolve: str = "neumann",
        tol: float = 1e-8,
        maxit: int = 50,
        verbose: bool = False,
    ) -> Any:
        """Solve (I - L) x = b using Neumann series or GMRES."""
        if linsolve == "neumann":
            x = b.copy()
            du = b.copy()
            for it in range(maxit):
                du = self.apply(du)
                x = x + du
                err = np.abs(np.asarray(du)).max()
                if err < tol:
                    break
            return x
        elif linsolve == "gmres":
            from scipy.sparse.linalg import gmres as scipy_gmres

            I_minus_L = LinearOperator(
                dtype=np.float64,
                shape=self.shape,
                matvec=lambda v: v - self._matvec(v),
            )
            b_np = np.asarray(b.data if isinstance(b, GArray) else b)
            b_flat = b_np.ravel()
            try:
                sol_flat, info = scipy_gmres(
                    I_minus_L,
                    b_flat,
                    x0=b_flat,
                    rtol=tol,
                    atol=0.0,
                    restart=maxit,
                    maxiter=1,
                )
            except TypeError:
                sol_flat, info = scipy_gmres(
                    I_minus_L, b_flat, x0=b_flat, tol=tol, restart=maxit, maxiter=1
                )
            sol_reshaped = sol_flat.reshape(b_np.shape)
            if isinstance(b, GArray):
                out = GArray.zeros_like(b)
                out.data[...] = sol_reshaped
                return out
            return sol_reshaped
        else:
            raise ValueError(
                f"Unknown linear solver: '{linsolve}'. Use 'neumann' or 'gmres'."
            )


def dF_2(
    model: Any,
    *args,
    engine: str = "auto",
    backend: Optional[str] = None,
    interp_method: str = "cubic",
    device: Optional[Any] = None,
    **kwargs,
) -> Union[Tuple[Tuple[Any, Any], ...], PolicyLinearOperator]:
    r"""Evaluate derivative of dynamic Euler residuals w.r.t the decision rule φ.

    Mirrors Julia Dolo.jl: dF_2(model, s, x, φ) and dF_2(model, controls, φ=None)

    Usage
    -----
    1. Grid-level evaluation:
       L = dF_2(model, controls, dr=None, engine="auto")
       L @ v        # apply operator to variation v
       L.as_matrix() # explicit matrix

    2. Point-level evaluation:
       branches = dF_2(model, s, x, dr)
       # returns tuple of (F_x, S) for each shock branch k

    Parameters
    ----------
    model : Model, ModelV2, or DiscretizedModel
        The economic model.
    controls : GArray, np.ndarray, or jnp.ndarray
        Control policy values across the grid (or point state s if point call).
    dr : DecisionRule, GArray, or array, optional
        Decision rule (policy). Defaults to `controls` if omitted.
    engine / backend : str ("auto", "jax", "gpu", "cpu", "vectorized")
        Backend engine.
    interp_method : str ("cubic" or "linear")
        Interpolation method for decision rule φ.
    device : jax.Device, optional
        Target JAX device (e.g. jax.devices('gpu')[0]).
    """
    if len(args) == 0:
        raise ValueError(
            "dF_2 requires at least 1 argument: (model, controls) or (model, s, x, dr)"
        )

    # 1. Point-level dispatch: dF_2(model, s, x, dr)
    if len(args) >= 2 and (
        isinstance(args[0], (QP, tuple))
        or (isinstance(args[0], np.ndarray) and args[0].ndim == 1)
    ):
        s, x = args[0], args[1]
        dr = args[2] if len(args) > 2 else None
        dmodel = (
            model
            if isinstance(model, DiscretizedModel)
            else DiscretizedModel(model, **kwargs)
        )
        eng = _resolve_engine(dmodel, engine, backend)

        if eng in ("jax", "gpu", "cpu"):
            import jax
            import jax.numpy as jnp
            from dolo.numeric.interpolation_jax import (
                eval_cubic_point_jax,
                eval_linear_point_jax,
            )

            cr = _get_or_compile_jax_residual(dmodel, interp_method=interp_method)
            policy_for_coeffs = dr if dr is not None else x
            coeffs = cr.prefilter(policy_for_coeffs)

            s_pt = jnp.asarray(s if not isinstance(s, QP) else s.val, dtype=jnp.float64)
            x_pt = jnp.asarray(x, dtype=jnp.float64)
            m_val, inodes, weights = cr.regime_data[0]
            n_inodes = len(weights)

            branches = []
            for k in range(n_inodes):
                M = inodes[k]
                w = weights[k]
                S = cr.g_pt(m_val, s_pt, x_pt, M, cr.parms)
                if interp_method == "cubic":
                    X = eval_cubic_point_jax(coeffs, S, cr.a, cr.b, cr.n)
                else:
                    X = eval_linear_point_jax(coeffs, S, cr.a, cr.b, cr.n)
                df_dX = jax.jacobian(
                    lambda u: cr.f_pt(m_val, s_pt, x_pt, M, S, u, cr.parms)
                )(X)
                branches.append((np.asarray(w * df_dX), np.asarray(S)))
            return tuple(branches)
        else:
            # Vectorized point
            dr_obj = (
                dr
                if isinstance(dr, DecisionRule)
                else dmodel.decision_rule(
                    dr if dr is not None else x, interp_method=interp_method
                )
            )
            s_val = s.val if isinstance(s, QP) else s
            n_m = (
                len(dmodel.model.symbols["exogenous"])
                if "exogenous" in dmodel.model.symbols
                else len(dmodel.model.calibration.get("exogenous", []))
            )
            if isinstance(s, QP):
                m_val = s_val[:n_m]
                s_endo = s_val[n_m:]
            else:
                m_val = np.asarray(dmodel.dprocess.node(0))
                s_endo = s_val

            n_inodes = dmodel.dprocess.n_inodes(0)

            branches = []
            for k in range(n_inodes):
                M_val = dmodel.dprocess.inode(0, k)
                w = float(dmodel.dprocess.iweight(0, k))
                S_endo = dmodel.g(
                    m_val[None, :],
                    s_endo[None, :],
                    x[None, :],
                    M_val[None, :],
                    dmodel.parms,
                )[0]
                val_next = np.concatenate([M_val, S_endo])
                target_regime = 0 if dmodel.dprocess.n_nodes == 1 else k
                S_qp = QP(loc=(target_regime, S_endo), val=val_next)
                X_next = dr_obj(S_qp)

                ff = SerialDifferentiableFunction(
                    lambda u: dmodel.arbitrage(
                        m_val[None, :],
                        s_endo[None, :],
                        x[None, :],
                        M_val[None, :],
                        S_endo[None, :],
                        u,
                    )
                )
                _, rr_X = ff(X_next[None, :])
                branches.append((w * rr_X[0], S_endo))
            return tuple(branches)

    # 2. Grid-level dispatch: dF_2(model, controls, dr=None)
    controls = args[0]
    dr = args[1] if len(args) > 1 else None

    if isinstance(model, DiscretizedModel):
        dmodel = model
    else:
        grid_options = kwargs.pop("grid_options", None)
        dprocess_options = kwargs.pop("dprocess_options", {})
        dmodel = DiscretizedModel(
            model,
            grid_options=grid_options,
            dprocess_options=dprocess_options,
            **kwargs,
        )

    eng = _resolve_engine(dmodel, engine, backend)

    if eng in ("jax", "gpu", "cpu"):
        import jax
        import jax.numpy as jnp
        from dolo.numeric.interpolation_jax import (
            eval_cubic_point_jax,
            eval_linear_point_jax,
        )

        target_dev = device
        if target_dev is None:
            if eng == "gpu":
                gpus = jax.devices("gpu")
                if not gpus:
                    raise RuntimeError(
                        "GPU requested (engine='gpu'), but no CUDA device was found."
                    )
                target_dev = gpus[0]
            elif eng == "cpu":
                target_dev = jax.devices("cpu")[0]

        cr = _get_or_compile_jax_residual(dmodel, interp_method=interp_method)
        policy_for_coeffs = dr if dr is not None else controls
        coeffs = cr.prefilter(policy_for_coeffs)

        s_nodes = jnp.asarray(dmodel.endo_nodes, dtype=jnp.float64)
        if isinstance(controls, GArray):
            x_vals = jnp.asarray(
                controls[0] if dmodel.n_regimes == 1 else controls.data,
                dtype=jnp.float64,
            )
        else:
            c_arr = jnp.asarray(controls, dtype=jnp.float64)
            x_vals = c_arr[0] if (dmodel.n_regimes == 1 and c_arr.ndim == 3) else c_arr

        if target_dev is not None:
            x_vals = jax.device_put(x_vals, target_dev)
            coeffs = jax.device_put(coeffs, target_dev)

        F_x_all, S_all = cr.dF_2(x_vals, coeffs)

        if target_dev is not None:
            F_x_all.block_until_ready()
            S_all.block_until_ready()

        return PolicyLinearOperator(
            dmodel,
            F_x_all,
            S_all,
            engine=eng,
            interp_method=interp_method,
            device=target_dev,
            cr=cr,
        )

    elif eng == "vectorized":
        # Pure NumPy evaluation
        controls_ga = (
            controls if isinstance(controls, GArray) else dmodel.initial_guess()
        )
        if not isinstance(controls, GArray):
            if controls_ga.data.shape == controls.shape:
                controls_ga.data[...] = controls
            elif controls_ga[0].shape == controls.shape:
                controls_ga[0] = controls

        if isinstance(dr, DecisionRule):
            dr_obj = dr
        else:
            dr_obj = dmodel.decision_rule(
                controls_ga if dr is None else dr, interp_method=interp_method
            )

        s_nodes = dmodel.endo_nodes
        N = s_nodes.shape[0]

        if dmodel.n_regimes == 1:
            x_vals = (
                controls_ga[0]
                if isinstance(controls_ga, GArray)
                else (controls_ga[0] if controls_ga.ndim == 3 else controls_ga)
            )
            m_val = dmodel.dprocess.node(0)
            n_inodes = dmodel.dprocess.n_inodes(0)
            m_tiled = np.tile(m_val, (N, 1))

            F_x_list = []
            S_list = []
            for k in range(n_inodes):
                M_val = dmodel.dprocess.inode(0, k)
                M_tiled = np.tile(M_val, (N, 1))
                w = float(dmodel.dprocess.iweight(0, k))

                S = dmodel.g(m_tiled, s_nodes, x_vals, M_tiled, dmodel.parms)
                X = dr_obj.eval_ijs(0, k, S)

                ff = SerialDifferentiableFunction(
                    lambda u: dmodel.arbitrage(m_val, s_nodes, x_vals, M_val, S, u)
                )
                _, rr_X = ff(X)
                F_x_list.append(w * rr_X)
                S_list.append(S)

            F_x_all = np.stack(F_x_list, axis=0)  # (K, N, n_x, n_x)
            S_all = np.stack(S_list, axis=0)  # (K, N, n_s)

        else:
            Fx_regimes = []
            S_regimes = []
            for i in range(dmodel.n_regimes):
                m_val = dmodel.dprocess.node(i)
                n_inodes = dmodel.dprocess.n_inodes(i)
                m_tiled = np.tile(m_val, (N, 1))
                x_i = (
                    controls_ga.data[i]
                    if isinstance(controls_ga, GArray)
                    else controls_ga[i]
                )

                fx_list = []
                s_list = []
                for k in range(n_inodes):
                    M_val = dmodel.dprocess.inode(i, k)
                    M_tiled = np.tile(M_val, (N, 1))
                    w = float(dmodel.dprocess.iweight(i, k))

                    S = dmodel.g(m_tiled, s_nodes, x_i, M_tiled, dmodel.parms)
                    X = dr_obj.eval_ijs(i, k, S)

                    ff = SerialDifferentiableFunction(
                        lambda u: dmodel.arbitrage(m_val, s_nodes, x_i, M_val, S, u)
                    )
                    _, rr_X = ff(X)
                    fx_list.append(w * rr_X)
                    s_list.append(S)
                Fx_regimes.append(np.stack(fx_list, axis=0))
                S_regimes.append(np.stack(s_list, axis=0))

            F_x_all = np.stack(Fx_regimes, axis=0)  # (n_m, K, N, n_x, n_x)
            S_all = np.stack(S_regimes, axis=0)  # (n_m, K, N, n_s)

        return PolicyLinearOperator(
            dmodel,
            F_x_all,
            S_all,
            engine="vectorized",
            interp_method=interp_method,
            device=None,
            cr=None,
        )

    raise ValueError(f"Unsupported engine: {eng}")
