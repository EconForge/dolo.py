"""Time Iteration Algorithm"""

import copy
import time
import numpy
import scipy.sparse.linalg

from dolo import dprint
from dolo.compiler.model import Model
from dolo.misc.itprinter import IterationsPrinter
from dolo.numeric.decision_rule import DecisionRule
from dolo.numeric.grids import CartesianGrid
from dolo.numeric.optimize.ncpsolve import ncpsolve
from dolo.numeric.optimize.newton import (
    SerialDifferentiableFunction,
    newton as serial_newton,
)
from dolo.numeric.processes import DiscretizedIIDProcess

from dolo.numeric import DiscretizedModel, GArray
from .invert import solve_tensor
from .newton import (
    Operator,
    d_filt_dx,
    euler_residuals as euler_residuals_acc,
    neumann_solve,
    smooth,
    smooth_nodiff,
)
from .results import AlgoResult, TimeIterationResult
from dolo.numeric.residuals import _resolve_engine, _get_or_compile_jax_residual


def euler_residuals_tau(
    dmodel: DiscretizedModel, controls: GArray, dr: DecisionRule
) -> GArray:
    """Universal Euler residuals written entirely in terms of GArray and Tau."""
    res = GArray.zeros_like(controls)
    for i_m, m_val, s_endo in dmodel.regimes():
        xm = controls[i_m]
        for prob, S in dmodel.tau.slice_transitions(i_m, m_val, s_endo, xm):
            XM = dr(S)
            rr = dmodel.arbitrage(m_val, s_endo, xm, S.exo, S.endo, XM)
            res[i_m] += prob * rr
    return res


def residuals_simple(f, g, s, x, dr, dprocess, parms):
    """Legacy residuals computation retained for backward compatibility."""
    N = s.shape[0]
    res = numpy.zeros_like(x)

    for i_ms in range(dprocess.n_nodes):
        m = numpy.tile(dprocess.node(i_ms), (N, 1))
        xm = x[i_ms, :, :]

        for I_ms in range(dprocess.n_inodes(i_ms)):
            M = numpy.tile(dprocess.inode(i_ms, I_ms), (N, 1))
            prob = dprocess.iweight(i_ms, I_ms)
            S = g(m, s, xm, M, parms)
            XM = dr.eval_ijs(i_ms, I_ms, S)
            rr = f(m, s, xm, M, S, XM, parms)
            res[i_ms, :, :] += prob * rr

    return res



def _time_iteration_device(
    cr,
    dmodel,
    x_init,
    *,
    eng,
    device,
    interpolation,
    inner_cfg,
    acc_cfg,
    tol_eta,
    tol_eps,
    maxit,
    itprint,
    on_iteration=None,
):
    """Time iteration with the controls, the coefficients of the decision rule and all the
    derivatives kept on the JAX device: only scalars (errors) go to the host during the
    iterations. Same steps as the loop of `time_iteration`, as in Dolo.jl: per iteration, the
    decision rule phi_0 interpolates x_0; x_1 solves F(x_1, phi_0) = 0 by Newton steps with
    backtracking; with acceleration, x_0 + (I - L)^{-1}(x_1 - x_0) with L = -dF_1^{-1} dF_2 at
    (x_1, phi_0).

    Returns the final controls (device array (n_m, N, n_x)), the number of iterations, the last
    step and the residual of the final controls (with their own decision rule)."""
    import jax
    import jax.numpy as jnp
    from dolo.numeric.residuals import PolicyLinearOperator
    from dolo.numeric.residual_kernels_jax import axpy, block_solve, max_abs as _max_abs

    fk = cr._fast
    n_m = dmodel.n_regimes

    def prefilter(x):
        C = cr.prefilter(x)
        return C[None] if n_m == 1 else C

    def max_abs(a):
        return float(_max_abs(a))

    x0 = jnp.asarray(x_init, dtype=jnp.float64)
    if device is not None:
        x0 = jax.device_put(x0, device)

    inner_maxit = inner_cfg.get("maxit", 20)
    inner_tol = inner_cfg.get("tol", 1e-10)
    err_step = numpy.inf
    err_res = numpy.inf
    err_0 = numpy.nan
    it = 0
    while (err_step > tol_eta or err_res > tol_eps) and it < maxit:
        it += 1
        t_start = time.time()
        coeffs = prefilter(x0)
        r_cur, J_cur = fk.F_and_dF_1(x0, coeffs)
        err_res = max_abs(r_cur)
        if on_iteration is not None:
            on_iteration(x0)

        # inner Newton solver: x1 such that F(x1, phi_0) = 0
        x_cur = x0
        nit = 0
        for i_inner in range(inner_maxit):
            if i_inner > 0:
                r_cur, J_cur = fk.F_and_dF_1(x_cur, coeffs)
            err_cur = max_abs(r_cur)
            if err_cur < inner_tol:
                break
            dx = block_solve(J_cur, r_cur)
            best_x = None
            for bck in range(25):
                cand_x = axpy(x_cur, -(0.5**bck), dx)
                cand_err = max_abs(fk.F(cand_x, coeffs))
                if not numpy.isnan(cand_err) and cand_err < err_cur:
                    best_x = cand_x
                    break
            if best_x is None:
                break
            x_cur = best_x
            nit += 1
        x1 = x_cur
        r_cur = J_cur = None

        err_step = max_abs(axpy(x1, -1.0, x0))
        err_SA = err_step / err_0 if not numpy.isnan(err_0) and err_0 > 0 else numpy.nan
        err_0 = err_step

        if acc_cfg is not None and it > acc_cfg["wait"]:
            # x = T(x), x1 = T(x0): x = x0 + (I - T')^{-1} (x1 - x0), T' = -F_1^{-1} F_2
            _, J_1 = fk.F_and_dF_1(x1, coeffs)
            F_c, S = fk.dF_2_compressed(x1, coeffs)
            if n_m == 1:
                F_c, S = F_c[0], S[0]
            L_op = PolicyLinearOperator(
                dmodel,
                F_c,
                S,
                engine=eng,
                interp_method=interpolation,
                device=device,
                cr=cr,
                cols=fk.xcols,
            )
            del F_c
            L_prec = (-1.0) * L_op.preconditioned(J_1)
            del L_op, J_1
            tot = L_prec.solve(
                axpy(x1, -1.0, x0),
                linsolve=acc_cfg.get("linsolve", "neumann"),
                tol=acc_cfg.get("tol", 1e-8),
                maxit=acc_cfg.get("maxit", 50),
            )
            del L_prec
            if acc_cfg.get("safeguarded", False):
                c_1, alpha = 1e-4, 0.5
                lam = numpy.sqrt(1 - 2 * c_1 * alpha)
                x_new = x1
                for bck in range(6):
                    cand = axpy(x0, alpha**bck, tot)
                    cand_err = max_abs(fk.F(cand, coeffs))
                    if numpy.isfinite(cand_err) and cand_err < lam * err_res:
                        x_new = cand
                        break
            else:
                x_new = axpy(x0, 1.0, tot)
            x0 = x_new
            err_res = max_abs(fk.F(x0, coeffs))
        else:
            x0 = x1

        itprint.print_iteration(
            N=it, Error=err_step, Gain=err_SA, Time=time.time() - t_start, nit=nit
        )
        if err_step < tol_eta and it > 1:
            break

    err_res = max_abs(fk.F(x0, prefilter(x0)))
    return x0, it, err_step, err_res


def time_iteration(
    model: Model,
    *,
    engine: str = "auto",
    backend: str = None,
    dr0: DecisionRule = None,
    tol_eps: float = 1e-8,
    tol_eta: float = 1e-6,
    maxit: int = 500,
    inner: dict = None,
    accelerate=False,
    interpolation: str = "cubic",
    verbose: bool = True,
    details: bool = True,
    trace: bool = False,
    ignore_constraints: bool = False,
    hook=None,
    # Backward compatibility:
    dprocess=None,
    **kwargs,
) -> TimeIterationResult:
    """Finds a global solution for ``model`` using backward time-iteration.

    Parameters
    ----------
    model : Model, ModelV2, or DiscretizedModel
        model to be solved
    engine / backend : str ("auto", "jax", "gpu", "cpu", "vectorized")
        computational engine for residual and Jacobian evaluations
    dr0 : DecisionRule, optional
        initial guess for the decision rule
    tol_eps : float (1e-8)
        tolerance on Euler equation residuals ||F(x)||
    tol_eta : float (1e-6)
        tolerance on policy update step ||x_{k+1} - x_k||
    maxit : int (500)
        maximum number of outer time-iteration loops
    inner : dict, optional
        options for the inner pointwise Euler solver:
        dict(maxit=10, max_backsteps=10, tol=1e-8)
    accelerate : bool or dict (False)
        if True or a dict, accelerates policy iterations using a preconditioned
        Neumann or GMRES step on (I - L) dx = (x_1 - x_0) after a warm-up phase.
        e.g. dict(wait=10, linsolve="neumann", maxit=50, safeguarded=False)
    interpolation : str ("cubic")
        interpolation method ("cubic" or "linear")
    verbose : bool (True)
        display iteration progress
    trace : bool (False)
        record intermediate decision rules
    """
    # Absorb legacy aliases and flat arguments via kwargs
    if "ftol" in kwargs:
        tol_eps = kwargs.pop("ftol")
    if "xtol" in kwargs:
        tol_eta = kwargs.pop("xtol")
    if "tol" in kwargs:
        tol_eta = kwargs.pop("tol")
    if "interp_method" in kwargs:
        interpolation = kwargs.pop("interp_method")
    if "improve" in kwargs:
        accelerate = kwargs.pop("improve")

    with_complementarities = kwargs.pop("with_complementarities", None)
    if with_complementarities is not None:
        pass
    else:
        with_complementarities = not ignore_constraints

    inner_maxit = kwargs.pop("inner_maxit", None)
    max_backsteps = kwargs.pop("max_backsteps", None)

    # Inner solver configuration
    inner_cfg = {"maxit": 10, "max_backsteps": 10, "tol": 1e-6}
    if inner is not None:
        inner_cfg.update(inner)
    if inner_maxit is not None:
        inner_cfg["maxit"] = inner_maxit
    if max_backsteps is not None:
        inner_cfg["max_backsteps"] = max_backsteps

    # Acceleration configuration
    acc_flag = accelerate
    if acc_flag is False or acc_flag is None:
        acc_cfg = None
    elif acc_flag is True:
        acc_cfg = {
            "wait": 10,
            "linsolve": "neumann",
            "maxit": 50,
            "tol": 1e-8,
            "safeguarded": False,
        }
    elif isinstance(acc_flag, dict):
        acc_cfg = {
            "wait": 10,
            "linsolve": "neumann",
            "maxit": 50,
            "tol": 1e-8,
            "safeguarded": False,
            **acc_flag,
        }
        if acc_cfg.get("linsolve") == "iti":
            acc_cfg["linsolve"] = "neumann"
    else:
        acc_cfg = None

    if trace:
        trace_details = []
    else:
        trace_details = None

    # Construct or adopt DiscretizedModel
    if isinstance(model, DiscretizedModel):
        dmodel = model
    else:
        grid_options = kwargs.pop("grid_options", None)
        dprocess_options = kwargs.pop("dprocess_options", {})
        if dprocess is not None:
            dprocess_options["dprocess"] = dprocess
        dmodel = DiscretizedModel(
            model, grid_options=grid_options, dprocess_options=dprocess_options
        )

    grid = dmodel.grid
    dprocess = dmodel.dprocess

    n_ms = dmodel.n_regimes
    n_im = dprocess.n_inodes(0)

    x0 = dmodel.model.calibration["controls"]
    parms = dmodel.parms
    n_x = len(x0)
    n_s = len(dmodel.model.symbols["states"])

    endo_grid = grid["endo"]
    exo_grid = grid["exo"]

    mdr = DecisionRule(
        exo_grid, endo_grid, dprocess=dprocess, interp_method=interpolation
    )
    if acc_cfg is not None:
        mdr_filt = DecisionRule(
            exo_grid, endo_grid, dprocess=dprocess, interp_method=interpolation
        )
        jres_alloc = numpy.zeros((n_ms, n_im, endo_grid.nodes.shape[0], n_x, n_x))
        S_ij_alloc = numpy.zeros((n_ms, n_im, endo_grid.nodes.shape[0], n_s))
    else:
        mdr_filt = None

    s = mdr.endo_grid.nodes
    N = s.shape[0]

    controls_ga = dmodel.initial_guess()
    if dr0 is None:
        pass
    else:
        if isinstance(dr0, AlgoResult):
            dr0 = dr0.dr
        try:
            for i_m in range(n_ms):
                controls_ga[i_m] = dr0(i_m, s)
        except Exception:
            for i_m in range(n_ms):
                m = dprocess.node(i_m)
                controls_ga[i_m] = dr0(m, s)

    f = dmodel.f
    g = dmodel.g

    if with_complementarities:
        lb_ga = dmodel.lower_bounds()
        ub_ga = dmodel.upper_bounds()
        if lb_ga is None or ub_ga is None:
            with_complementarities = False
            lb_flat = None
            ub_flat = None
            lb = None
            ub = None
        else:
            lb_flat = lb_ga.ravel().reshape((-1, n_x))
            ub_flat = ub_ga.ravel().reshape((-1, n_x))
            lb = lb_ga.data
            ub = ub_ga.data
    else:
        with_complementarities = False
        lb = None
        ub = None
        lb_flat = None
        ub_flat = None

    sh_c = controls_ga.shape
    controls_0 = controls_ga.ravel().reshape((-1, n_x))

    # Engine setup
    eng = _resolve_engine(dmodel, engine, backend)
    use_jax = eng in ("jax", "gpu", "cpu")
    if use_jax:
        import jax
        import jax.numpy as jnp

        target_dev = None
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
            dmodel, interp_method=interpolation, diff_mode="fwd"
        )
        s_nodes_jax = jnp.asarray(dmodel.endo_nodes, dtype=jnp.float64)
        if target_dev is not None:
            s_nodes_jax = jax.device_put(s_nodes_jax, target_dev)
    else:
        cr = None
        target_dev = None

    itprint = IterationsPrinter(
        ("N", int),
        ("Error", float),
        ("Gain", float),
        ("Time", float),
        ("nit", int),
        verbose=verbose,
    )
    itprint.print_header("Start Time Iterations.")

    t1 = time.time()
    err_0 = numpy.nan
    err_step = 10.0
    err_res = 10.0
    it = 0
    verbit = verbose == "full"

    # JAX engines: everything on the device (one regime, or regimes with the same number of
    # nodes; complementarities still go through the loop below)
    use_device_loop = use_jax and not with_complementarities and cr._fast is not None
    if use_device_loop:

        def on_iteration(x):
            if trace or hook:
                mdr.set_values(numpy.asarray(x))
            if trace:
                trace_details.append({"dr": copy.deepcopy(mdr)})
            if hook:
                hook()

        x_final, it, err_step, err_res = _time_iteration_device(
            cr,
            dmodel,
            controls_ga.data,
            eng=eng,
            device=target_dev,
            interpolation=interpolation,
            inner_cfg=inner_cfg,
            acc_cfg=acc_cfg,
            tol_eta=tol_eta,
            tol_eps=tol_eps,
            maxit=maxit,
            itprint=itprint,
            on_iteration=on_iteration,
        )
        controls_0 = numpy.asarray(x_final).reshape((-1, n_x))
        controls_ga = controls_ga.unravel(controls_0)

    while not use_device_loop and (err_step > tol_eta or err_res > tol_eps) and it < maxit:
        it += 1
        t_start = time.time()

        mdr.set_values(controls_ga)

        if trace:
            trace_details.append({"dr": copy.deepcopy(mdr)})

        if use_jax:
            coeffs_jax = cr.prefilter(controls_ga)
            if target_dev is not None:
                coeffs_jax = jax.device_put(coeffs_jax, target_dev)

            def dfn_jax(x_val):
                x_reshaped = x_val.reshape(sh_c)
                x_jnp = jnp.asarray(x_reshaped, dtype=jnp.float64)
                if target_dev is not None:
                    x_jnp = jax.device_put(x_jnp, target_dev)
                r_jnp, dr_jnp = cr.F_and_dF_1(x_jnp, coeffs_jax)
                if target_dev is not None:
                    r_jnp.block_until_ready()
                    dr_jnp.block_until_ready()
                r_np = numpy.asarray(r_jnp).reshape((-1, n_x))
                dr_np = numpy.asarray(dr_jnp).reshape((-1, n_x, n_x))
                return [r_np, dr_np]

            def fn_jax(x_val):
                # residual only: no Jacobian (used for err_res and the safeguards)
                x_jnp = jnp.asarray(x_val.reshape(sh_c), dtype=jnp.float64)
                if target_dev is not None:
                    x_jnp = jax.device_put(x_jnp, target_dev)
                return numpy.asarray(cr.F(x_jnp, coeffs_jax)).reshape((-1, n_x))

            fn = fn_jax
            dfn = dfn_jax
            # residual and Jacobian at controls_0, kept on the device: the Jacobian is
            # reused by the first step of the inner Newton solver
            x0_jnp = jnp.asarray(controls_0.reshape(sh_c), dtype=jnp.float64)
            if target_dev is not None:
                x0_jnp = jax.device_put(x0_jnp, target_dev)
            r0_jnp, J0_jnp = cr.F_and_dF_1(x0_jnp, coeffs_jax)
            res_0 = numpy.asarray(r0_jnp).reshape((-1, n_x))
            err_res = abs(res_0).max()
        else:
            fn = (
                lambda x_val: euler_residuals_tau(
                    dmodel, controls_ga.unravel(x_val), mdr
                )
                .ravel()
                .reshape((-1, n_x))
            )
            dfn = SerialDifferentiableFunction(fn)

            res_0 = fn(controls_0)
            err_res = abs(res_0).max()

        if hook:
            hook()

        # Inner Euler solver
        if with_complementarities:
            [controls, nit] = ncpsolve(
                dfn,
                lb_flat,
                ub_flat,
                controls_0,
                verbose=verbit,
                maxit=inner_cfg["maxit"],
            )
        else:
            if use_jax:
                inner_maxit = inner_cfg.get("maxit", 20)
                inner_tol = inner_cfg.get("tol", 1e-10)
                x_cur = x0_jnp
                r_cur, J_cur = r0_jnp, J0_jnp
                nit = 0
                for i_inner in range(inner_maxit):
                    if i_inner > 0:
                        r_cur, J_cur = cr.F_and_dF_1(x_cur, coeffs_jax)
                    err_cur = float(jnp.abs(r_cur).max())
                    if err_cur < inner_tol:
                        break
                    dx = jnp.linalg.solve(J_cur, r_cur[..., None])[..., 0]
                    best_x = x_cur
                    for bck in range(25):
                        step = 0.5**bck
                        cand_x = x_cur - dx * step
                        cand_r = cr.F(cand_x, coeffs_jax)
                        cand_err = float(jnp.abs(cand_r).max())
                        if not numpy.isnan(cand_err) and cand_err < err_cur:
                            best_x = cand_x
                            break
                    if best_x is x_cur:
                        break
                    x_cur = best_x
                    nit += 1
                if target_dev is not None:
                    x_cur.block_until_ready()
                controls = numpy.asarray(x_cur).reshape((-1, n_x))
                # free the device memory of the Jacobians before the next kernels
                r0_jnp = J0_jnp = r_cur = J_cur = None
            else:
                [controls, nit] = serial_newton(
                    dfn, controls_0, verbose=verbit, maxit=inner_cfg["maxit"]
                )

        new_controls_ga = controls_ga.unravel(controls)
        err_step = (new_controls_ga - controls_ga).norm()
        controls_ga = new_controls_ga
        err_SA = err_step / err_0 if not numpy.isnan(err_0) and err_0 > 0 else numpy.nan
        err_0 = err_step

        # Policy acceleration step
        if acc_cfg is not None and it > acc_cfg["wait"]:
            d = (controls - controls_0).reshape(sh_c)
            acc_method = acc_cfg.get("linsolve", "neumann")
            acc_maxit = acc_cfg.get("maxit", 50)
            acc_tol = acc_cfg.get("tol", 1e-8)
            safeguarded = acc_cfg.get("safeguarded", False)

            if use_jax and not with_complementarities:
                from dolo.numeric.residuals import dF_2

                cand_ga = controls_ga.unravel(controls)
                _, dres_c = dfn(controls)
                if dmodel.n_regimes == 1:
                    J_c = dres_c.reshape((N, n_x, n_x))
                else:
                    J_c = dres_c.reshape((dmodel.n_regimes, N, n_x, n_x))

                # Evaluate dF_2 at candidate controls x1 with policy φ0 interpolating controls_0
                L_op = dF_2(
                    dmodel,
                    cand_ga,
                    dr=controls_0.reshape(sh_c),
                    engine=eng,
                    interp_method=interpolation,
                    device=target_dev,
                )
                L_prec = (-1.0) * L_op.preconditioned(J_c)
                tot = L_prec.solve(d, linsolve=acc_method, tol=acc_tol, maxit=acc_maxit)
                cand_controls_0 = controls_0 + numpy.asarray(tot).reshape((-1, n_x))
                # free the device memory of the operators before the next kernels
                del L_op, L_prec
            else:
                # Evaluate derivatives at candidate controls
                mdr.set_values(controls.reshape(sh_c))
                res_c, dres_c = dfn(controls)
                res_c = res_c.reshape(sh_c)
                dres_c = dres_c.reshape((sh_c[0], N, n_x, n_x))

                jres_alloc[...] = 0.0
                S_ij_alloc[...] = 0.0
                junk, jres, fut_S = euler_residuals_acc(
                    f,
                    g,
                    s,
                    controls.reshape(sh_c),
                    mdr,
                    dprocess,
                    parms,
                    diff=False,
                    with_jres=True,
                    set_dr=False,
                    jres=jres_alloc,
                    S_ij=S_ij_alloc,
                )

                if with_complementarities:
                    res_c, dres_c, jres = smooth(
                        res_c, dres_c, jres, controls.reshape(sh_c) - lb
                    )
                    res_c[...] *= -1
                    dres_c[...] *= -1
                    jres[...] *= -1
                    res_c, dres_c, jres = smooth(
                        res_c, dres_c, jres, ub - controls.reshape(sh_c), pos=-1.0
                    )
                    res_c[...] *= -1
                    dres_c[...] *= -1
                    jres[...] *= -1

                jres[...] *= -1.0

                for i_m in range(n_ms):
                    for j_m in range(n_im):
                        M_mat = jres[i_m, j_m, :, :, :]
                        X_mat = dres_c[i_m, :, :, :].copy()
                        solve_tensor(X_mat, M_mat)

                if acc_method == "gmres":
                    L = Operator(jres, fut_S, mdr_filt)
                    L.addid = True
                    try:
                        sol_gmres, info = scipy.sparse.linalg.gmres(
                            L, d.ravel(), rtol=acc_tol, atol=0.0, maxiter=acc_maxit
                        )
                    except TypeError:
                        sol_gmres, info = scipy.sparse.linalg.gmres(
                            L, d.ravel(), tol=acc_tol, maxiter=acc_maxit
                        )
                    tot = sol_gmres.reshape(d.shape)
                else:
                    tot, *rest = neumann_solve(
                        d,
                        jres,
                        fut_S,
                        mdr_filt,
                        tol=acc_tol,
                        maxit=acc_maxit,
                        verbose=(verbose == "full"),
                    )

                cand_controls_0 = controls_0 + tot.reshape((-1, n_x))

            if safeguarded:
                c_1 = 1e-4
                alpha = 0.5
                lam = numpy.sqrt(1 - 2 * c_1 * alpha)
                accepted = False
                tot_flat = numpy.asarray(tot).reshape((-1, n_x))
                for bck in range(6):
                    step = alpha**bck
                    damp_cand = controls_0 + step * tot_flat
                    damp_res = fn(damp_cand)
                    damp_err = float(abs(damp_res).max())
                    if numpy.isfinite(damp_err) and damp_err < lam * err_res:
                        controls_0 = damp_cand
                        accepted = True
                        break
                if not accepted:
                    controls_0 = controls
            else:
                controls_0 = cand_controls_0
        else:
            controls_0 = controls

        controls_ga = controls_ga.unravel(controls_0)

        if acc_cfg is not None and it > acc_cfg["wait"]:
            res_0 = fn(controls_0)
            err_res = float(abs(res_0).max())

        t_finish = time.time()
        elapsed = t_finish - t_start

        itprint.print_iteration(
            N=it, Error=err_step, Gain=err_SA, Time=elapsed, nit=nit
        )

        if err_step < tol_eta and it > 1:
            break

    mdr.set_values(controls_ga)

    if use_device_loop:
        pass  # err_res: residual of the final controls, computed on the device
    elif use_jax:
        coeffs_final = cr.prefilter(controls_ga)
        if target_dev is not None:
            coeffs_final = jax.device_put(coeffs_final, target_dev)
        x_final = jnp.asarray(controls_0.reshape(sh_c), dtype=jnp.float64)
        if target_dev is not None:
            x_final = jax.device_put(x_final, target_dev)
        res_final = cr.F(x_final, coeffs_final)
        if target_dev is not None:
            res_final.block_until_ready()
        err_res = float(jnp.abs(res_final).max())
    else:
        res_0 = fn(controls_0)
        err_res = float(abs(res_0).max())

    if trace:
        trace_details.append({"dr": copy.deepcopy(mdr)})

    itprint.print_finished()

    if not details:
        return mdr

    converged = bool(err_step < tol_eta or err_res < tol_eps)

    return TimeIterationResult(
        mdr,
        it,
        with_complementarities,
        dprocess,
        converged,
        tol_eta,
        err_step,
        None,
        trace_details,
        err_res,
    )
