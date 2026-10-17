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
from dolo.numeric.optimize.newton import SerialDifferentiableFunction, newton as serial_newton
from dolo.numeric.processes import DiscretizedIIDProcess

from .invert import solve_tensor
from .newton import (
    Operator,
    d_filt_dx,
    euler_residuals,
    neumann_solve,
    smooth,
    smooth_nodiff,
)
from .results import AlgoResult, TimeIterationResult


def residuals_simple(f, g, s, x, dr, dprocess, parms):
    N = s.shape[0]
    n_s = s.shape[1]

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


def time_iteration(
    model: Model,
    *,
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
    dprocess=None,
    ignore_constraints: bool = False,
    hook=None,
    **kwargs,
) -> TimeIterationResult:
    """Finds a global solution for ``model`` using backward time-iteration.

    Parameters
    ----------
    model : Model
        model to be solved
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

    grid, dprocess_ = model.discretize()
    if dprocess is None:
        dprocess = dprocess_

    n_ms = dprocess.n_nodes
    n_im = dprocess.n_inodes(0)

    x0 = model.calibration["controls"]
    parms = model.calibration["parameters"]
    n_x = len(x0)
    n_s = len(model.symbols["states"])

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

    controls_0 = numpy.zeros((n_ms, N, n_x))
    if dr0 is None:
        controls_0[:, :, :] = x0[None, None, :]
    else:
        if isinstance(dr0, AlgoResult):
            dr0 = dr0.dr
        try:
            for i_m in range(n_ms):
                controls_0[i_m, :, :] = dr0(i_m, s)
        except Exception:
            for i_m in range(n_ms):
                m = dprocess.node(i_m)
                controls_0[i_m, :, :] = dr0(m, s)

    f = model.functions["arbitrage"]
    g = model.functions["transition"]

    if "arbitrage_lb" in model.functions and with_complementarities:
        lb_fun = model.functions["arbitrage_lb"]
        ub_fun = model.functions["arbitrage_ub"]
        lb = numpy.zeros_like(controls_0) * numpy.nan
        ub = numpy.zeros_like(controls_0) * numpy.nan
        for i_m in range(n_ms):
            m = dprocess.node(i_m)[None, :]
            p = parms[None, :]
            m = numpy.repeat(m, N, axis=0)
            p = numpy.repeat(p, N, axis=0)

            lb[i_m, :, :] = lb_fun(m, s, p)
            ub[i_m, :, :] = ub_fun(m, s, p)
        lb_flat = lb.reshape((-1, n_x))
        ub_flat = ub.reshape((-1, n_x))
    else:
        with_complementarities = False
        lb = None
        ub = None
        lb_flat = None
        ub_flat = None

    sh_c = controls_0.shape
    controls_0 = controls_0.reshape((-1, n_x))

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

    while (err_step > tol_eta or err_res > tol_eps) and it < maxit:
        it += 1
        t_start = time.time()

        mdr.set_values(controls_0.reshape(sh_c))

        if trace:
            trace_details.append({"dr": copy.deepcopy(mdr)})

        fn = lambda x_val: residuals_simple(
            f, g, s, x_val.reshape(sh_c), mdr, dprocess, parms
        ).reshape((-1, n_x))
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
            [controls, nit] = serial_newton(
                dfn, controls_0, verbose=verbit, maxit=inner_cfg["maxit"]
            )

        err_step = abs(controls - controls_0).max()
        err_SA = err_step / err_0 if not numpy.isnan(err_0) and err_0 > 0 else numpy.nan
        err_0 = err_step

        # Policy acceleration step
        if acc_cfg is not None and it > acc_cfg["wait"]:
            d = (controls - controls_0).reshape(sh_c)

            # Evaluate derivatives at candidate controls
            mdr.set_values(controls.reshape(sh_c))
            res_c, dres_c = dfn(controls)
            res_c = res_c.reshape(sh_c)
            dres_c = dres_c.reshape((sh_c[0], N, n_x, n_x))

            jres_alloc[...] = 0.0
            S_ij_alloc[...] = 0.0
            junk, jres, fut_S = euler_residuals(
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

            acc_method = acc_cfg.get("linsolve", "neumann")
            acc_maxit = acc_cfg.get("maxit", 50)
            acc_tol = acc_cfg.get("tol", 1e-8)

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

            if acc_cfg.get("safeguarded", False):
                cand_res = fn(cand_controls_0)
                if abs(cand_res).max() < err_res:
                    controls_0 = cand_controls_0
                else:
                    controls_0 = controls
            else:
                controls_0 = cand_controls_0
        else:
            controls_0 = controls

        t_finish = time.time()
        elapsed = t_finish - t_start

        itprint.print_iteration(
            N=it, Error=err_step, Gain=err_SA, Time=elapsed, nit=nit
        )

        if err_step < tol_eta and it > 1:
            break

    controls_0 = controls_0.reshape(sh_c)
    mdr.set_values(controls_0)

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
