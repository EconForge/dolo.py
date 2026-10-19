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
        raise ValueError("F requires at least 2 arguments: (model, controls) or (model, s, x, dr)")

    # 1. Point-level dispatch: F(model, s, x, dr)
    if len(args) >= 2 and (
        isinstance(args[0], (QP, tuple))
        or (isinstance(args[0], np.ndarray) and args[0].ndim == 1)
    ):
        s, x = args[0], args[1]
        dr = args[2] if len(args) > 2 else None
        dmodel = model if isinstance(model, DiscretizedModel) else DiscretizedModel(model, **kwargs)
        eng = _resolve_engine(dmodel, engine, backend)
        if eng in ("jax", "gpu", "cpu"):
            cr = _get_or_compile_jax_residual(dmodel, interp_method=interp_method)
            coeffs = cr.prefilter(dr if dr is not None else x)
            return cr.F_point(s if not isinstance(s, QP) else s.val, x, coeffs)
        else:
            # Vectorized / point transition
            dr_obj = dr if isinstance(dr, DecisionRule) else dmodel.decision_rule(dr, interp_method=interp_method)
            s_qp = s if isinstance(s, QP) else QP(0, s)
            res = np.zeros_like(x)
            for prob, S in dmodel.tau.point_transitions(s_qp, x):
                rr = dmodel.arbitrage(s_qp.val[:dmodel.n_m], s_qp.val[dmodel.n_m:], x, S.val[:dmodel.n_m], S.val[dmodel.n_m:], dr_obj(S))
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
        dmodel = DiscretizedModel(model, grid_options=grid_options, dprocess_options=dprocess_options, **kwargs)

    eng = _resolve_engine(dmodel, engine, backend)

    if eng in ("jax", "gpu", "cpu"):
        import jax
        import jax.numpy as jnp

        target_dev = device
        if target_dev is None:
            if eng == "gpu":
                gpus = jax.devices("gpu")
                if not gpus:
                    raise RuntimeError("GPU requested (engine='gpu'), but no CUDA device was found.")
                target_dev = gpus[0]
            elif eng == "cpu":
                target_dev = jax.devices("cpu")[0]

        cr = _get_or_compile_jax_residual(dmodel, interp_method=interp_method)
        policy_for_coeffs = dr if dr is not None else controls
        coeffs = cr.prefilter(policy_for_coeffs)

        s_nodes = jnp.asarray(dmodel.endo_nodes, dtype=jnp.float64)
        if isinstance(controls, GArray):
            x_vals = jnp.asarray(controls[0] if dmodel.n_regimes == 1 else controls.data, dtype=jnp.float64)
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

        controls_ga = controls if isinstance(controls, GArray) else dmodel.initial_guess()
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
        dmodel = DiscretizedModel(model, grid_options=grid_options, dprocess_options=dprocess_options, **kwargs)

    eng = _resolve_engine(dmodel, engine, backend)

    if eng in ("jax", "gpu", "cpu"):
        import jax
        import jax.numpy as jnp

        target_dev = device
        if target_dev is None:
            if eng == "gpu":
                gpus = jax.devices("gpu")
                if not gpus:
                    raise RuntimeError("GPU requested (engine='gpu'), but no CUDA device was found.")
                target_dev = gpus[0]
            elif eng == "cpu":
                target_dev = jax.devices("cpu")[0]

        cr = _get_or_compile_jax_residual(dmodel, interp_method=interp_method, diff_mode=diff_mode)
        policy_for_coeffs = dr if dr is not None else controls
        coeffs = cr.prefilter(policy_for_coeffs)

        s_nodes = jnp.asarray(dmodel.endo_nodes, dtype=jnp.float64)
        if isinstance(controls, GArray):
            x_vals = jnp.asarray(controls[0] if dmodel.n_regimes == 1 else controls.data, dtype=jnp.float64)
        else:
            c_arr = jnp.asarray(controls, dtype=jnp.float64)
            x_vals = c_arr[0] if (dmodel.n_regimes == 1 and c_arr.ndim == 3) else c_arr

        if target_dev is not None:
            s_nodes = jax.device_put(s_nodes, target_dev)
            x_vals = jax.device_put(x_vals, target_dev)
            coeffs = jax.device_put(coeffs, target_dev)

        if dmodel.n_regimes == 1:
            jac = cr.dF_vmap(s_nodes, x_vals, coeffs)
            if target_dev is not None:
                jac.block_until_ready()
            return jac
        else:
            return cr.dF(controls, coeffs)

    elif eng == "vectorized":
        # Finite-difference derivative using SerialDifferentiableFunction
        controls_ga = controls if isinstance(controls, GArray) else dmodel.initial_guess()
        if not isinstance(controls, GArray):
            if controls_ga.data.shape == controls.shape:
                controls_ga.data[...] = controls
            elif controls_ga[0].shape == controls.shape:
                controls_ga[0] = controls

        if isinstance(dr, DecisionRule):
            dr_obj = dr
        else:
            dr_obj = dmodel.decision_rule(controls_ga if dr is None else dr, interp_method=interp_method)

        n_x = len(dmodel.model.calibration["controls"])
        fn = lambda x_val: euler_residuals_tau(dmodel, controls_ga.unravel(x_val), dr_obj).ravel().reshape((-1, n_x))
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
        dmodel = DiscretizedModel(model, grid_options=grid_options, dprocess_options=dprocess_options, **kwargs)

    eng = _resolve_engine(dmodel, engine, backend)

    if eng in ("jax", "gpu", "cpu"):
        import jax
        import jax.numpy as jnp

        target_dev = device
        if target_dev is None:
            if eng == "gpu":
                gpus = jax.devices("gpu")
                if not gpus:
                    raise RuntimeError("GPU requested (engine='gpu'), but no CUDA device was found.")
                target_dev = gpus[0]
            elif eng == "cpu":
                target_dev = jax.devices("cpu")[0]

        cr = _get_or_compile_jax_residual(dmodel, interp_method=interp_method, diff_mode=diff_mode)
        policy_for_coeffs = dr if dr is not None else controls
        coeffs = cr.prefilter(policy_for_coeffs)

        s_nodes = jnp.asarray(dmodel.endo_nodes, dtype=jnp.float64)
        if isinstance(controls, GArray):
            x_vals = jnp.asarray(controls[0] if dmodel.n_regimes == 1 else controls.data, dtype=jnp.float64)
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
            return cr.F(controls, coeffs), cr.dF(controls, coeffs)

    elif eng == "vectorized":
        res = F(dmodel, controls, dr, engine="vectorized", interp_method=interp_method, **kwargs)
        dres = dF_1(dmodel, controls, dr, engine="vectorized", interp_method=interp_method, **kwargs)
        return res, dres

    raise ValueError(f"Unsupported engine: {eng}")
