"""Newton-Krylov Policy Solver"""

import copy
import time
from functools import reduce
from math import sqrt
from operator import mul
from typing import Optional, Any, Union, Tuple, Dict

from numba import jit
import numpy
import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator

from dolo import dprint
from dolo.compiler.model import Model
from dolo.misc.itprinter import IterationsPrinter
from dolo.numeric.decision_rule import DecisionRule
from dolo.numeric.optimize.newton import SerialDifferentiableFunction

from .bruteforce_lib import *
from .invert import *
from .results import AlgoResult, ImprovedTimeIterationResult, NewtonResult
from dolo.numeric import DiscretizedModel, GArray
from dolo.numeric.residuals import (
    _get_or_compile_jax_residual,
    _resolve_engine,
    F,
    dF_1,
    dF_2,
    F_and_dF_1,
    PolicyLinearOperator,
)


def prod(l):
    return reduce(mul, l)


@jit
def inplace(Phi, J):
    a, b, c, d, e = J.shape
    for i_a in range(a):
        for i_b in range(b):
            for i_c in range(c):
                for i_d in range(d):
                    for i_e in range(e):
                        J[i_a, i_b, i_c, i_d, i_e] *= Phi[i_a, i_c, i_d]


def smooth(res, dres, jres, dx, pos=1.0):
    from numpy import sqrt

    dinf = dx > 100000
    n_m, N, n_x = res.shape
    sq = sqrt(res**2 + (dx) ** 2)
    H = res + (dx) - sq
    Phi_a = 1 - res / sq
    Phi_b = 1 - (dx) / sq
    H[dinf] = res[dinf]
    Phi_a[dinf] = 1.0
    Phi_b[dinf] = 0.0
    H_x = Phi_a[:, :, :, None] * dres
    for i_x in range(n_x):
        H_x[:, :, i_x, i_x] += Phi_b[:, :, i_x] * pos
    inplace(Phi_a, jres)
    return H, H_x, jres


def smooth_nodiff(res, dx):
    from numpy import sqrt

    n_m, N, n_x = res.shape
    dinf = dx > 100000
    sq = sqrt(res**2 + (dx) ** 2)
    H = res + (dx) - sq
    H[dinf] = res[dinf]
    return H


@jit
def ssmul(A, B):
    N, a, b = A.shape
    NN, b = B.shape
    O = numpy.zeros((N, a))
    for n in range(N):
        for k in range(a):
            for l in range(b):
                O[n, k] += A[n, k, l] * B[n, l]
    return O


@jit
def ssmul_inplace(A, B, O):
    N, a, b = A.shape
    NN, b = B.shape
    for n in range(N):
        for k in range(a):
            for l in range(b):
                O[n, k] += A[n, k, l] * B[n, l]
    return O


def d_filt_dx(pi, M_ij, S_ij, n_m, N, n_x, dumdr):
    n_m, n_im = M_ij.shape[:2]
    dumdr.set_values(pi)
    for i in range(n_m):
        pi[i, :, :] = 0
        for j in range(n_im):
            A = M_ij[i, j, :, :, :]
            B = dumdr.eval_ijs(i, j, S_ij[i, j, :, :])
            pi[i, :, :] += ssmul(A, B)
    return pi


class Operator(LinearOperator):
    """Special Linear Operator representing the policy Jacobian L = -F_1^{-1} F_2"""

    def __init__(self, M_ij, S_ij, dumdr):
        self.M_ij = M_ij
        self.S_ij = S_ij
        self.n_m = M_ij.shape[0]
        self.N = M_ij.shape[2]
        self.n_x = M_ij.shape[3]
        self.dumdr = dumdr
        self.dtype = numpy.dtype("float64")
        self.counter = 0
        self.addid = False
        nn = self.n_m * self.N * self.n_x
        super().__init__(dtype=self.dtype, shape=(nn, nn))
        self.ndim = 2

    def _matvec(self, x):
        self.counter += 1
        xx = x.reshape((self.n_m, self.N, self.n_x))
        yy = self.apply(xx)
        if self.addid:
            yy = xx - yy  # (I - L) operator
        return yy.ravel()

    def apply(self, pi, inplace=False):
        M_ij = self.M_ij
        S_ij = self.S_ij
        n_m = self.n_m
        N = self.N
        n_x = self.n_x
        dumdr = self.dumdr
        if not inplace:
            pi = pi.copy()
        return d_filt_dx(pi, M_ij, S_ij, n_m, N, n_x, dumdr)

    def as_matrix(self):
        arg = numpy.zeros((self.n_m, self.N, self.n_x))
        larg = arg.ravel()
        N = len(larg)
        J = numpy.zeros((N, N))
        for i in range(N):
            if i > 0:
                larg[i - 1] = 0.0
            larg[i] = 1.0
            J[:, i] = self.apply(arg).ravel()
        return J


def neumann_solve(ddx, jres, fut_S, dumdr, tol=1e-8, maxit=50, verbose=False):
    """Solve (I - L) tot = ddx using Neumann series expansion: sum_{k=0}^maxit L^k ddx."""
    n_m = ddx.shape[0]
    N = ddx.shape[1]
    n_x = ddx.shape[2]

    tot = ddx.copy()
    du = ddx.copy()
    err_0 = abs(du).max()
    lam = -1.0
    lam_max = -1.0
    nn = 0

    if verbose:
        print("Starting Neumann series")
    for nn in range(maxit):
        du = d_filt_dx(du, jres, fut_S, n_m, N, n_x, dumdr)
        err = abs(du).max()
        lam = err / err_0 if err_0 > 0 else 0.0
        lam_max = max(lam_max, lam)
        if verbose:
            print(f"- {err} | {lam} | {lam_max}")
        tot += du
        err_0 = err
        if err < tol:
            break

    return tot, nn, lam


def invert_jac(res, dres, jres, fut_S, dumdr, tol=1e-10, maxit=1000, verbose=False):
    """Solve (F_1 + F_2) tot = res via preconditioned Neumann series (I - L) tot = F_1^{-1} res."""
    ddx = solve_gu(dres.copy(), res.copy())
    return neumann_solve(ddx, jres, fut_S, dumdr, tol=tol, maxit=maxit, verbose=verbose)


def radius_jac(res, dres, jres, fut_S, dumdr, tol=1e-10, maxit=1000, verbose=False):
    from numpy import sqrt

    n_m = res.shape[0]
    N = res.shape[1]
    n_x = res.shape[2]

    norm2 = lambda m: sqrt((m**2).sum())

    pi = (numpy.random.random(res.shape) * 2 - 1) * 1.0
    pi /= norm2(pi)

    lam = 1.0
    lam_max = 0.0
    lambdas = []

    if verbose:
        print("Starting radius computation")
    for nn in range(maxit):
        pi[:, :, :] /= lam
        pi = d_filt_dx(pi, jres, fut_S, n_m, N, n_x, dumdr)
        lam = norm2(pi)
        lam_max = max(lam_max, lam)
        if verbose:
            print(f"- {lam} | {lam_max}")
        lambdas.append(lam)
    return (lam, lam_max, lambdas)


def golden_section_search(f_eval, alpha_min=0.0, alpha_max=1.0, maxiter=8, tol=1e-4):
    """Golden-section line search for step size alpha in [alpha_min, alpha_max]."""
    gr = (sqrt(5.0) - 1.0) / 2.0  # ~0.6180339887
    a = alpha_min
    b = alpha_max
    c = b - gr * (b - a)
    d = a + gr * (b - a)

    fc = f_eval(c)
    fd = f_eval(d)

    best_alpha = c if fc < fd else d
    best_f = min(fc, fd)

    for _ in range(maxiter):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f_eval(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f_eval(d)

        if fc < best_f:
            best_f = fc
            best_alpha = c
        if fd < best_f:
            best_f = fd
            best_alpha = d
        if abs(b - a) < tol:
            break

    return best_alpha, best_f


def newton(
    model: Model,
    *,
    engine: str = "auto",
    backend: Optional[str] = None,
    device: Optional[Any] = None,
    dr0: DecisionRule = None,
    tol_eps: float = 1e-8,
    tol_eta: float = 1e-6,
    maxit: int = 30,
    linsolve="gmres",
    linesearch="backtracking",
    interpolation: str = "cubic",
    verbose: bool = True,
    details: bool = True,
    trace: bool = False,
    dprocess=None,
    ignore_constraints: bool = False,
    compute_radius: bool = False,
    **kwargs,
) -> NewtonResult:
    """Finds a global solution for ``model`` using full-system Newton-Krylov solver.

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
    maxit : int (30)
        maximum number of outer Newton iterations
    linsolve : str or dict ("gmres")
        linear solver for the preconditioned system (I - L) dx = r.
        Options: "gmres", "neumann", or a dict(method=..., maxit=..., tol=...)
    linesearch : str or dict ("backtracking")
        step size stabilization. Options: "backtracking", "golden_section", False, None,
        or a dict(method=..., max_backsteps=...)
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
        tol_eps = kwargs.pop("tol")
    if "interp_method" in kwargs:
        interpolation = kwargs.pop("interp_method")

    complementarities = kwargs.pop("complementarities", None)
    invmethod = kwargs.pop("invmethod", None)
    smaxit = kwargs.pop("smaxit", None)
    linear_maxit = kwargs.pop("linear_maxit", None)
    max_backsteps = kwargs.pop("max_backsteps", None)
    maxbsteps = kwargs.pop("maxbsteps", None)
    mu = kwargs.pop("mu", 2)
    method = kwargs.pop("method", "jac")

    if complementarities is not None:
        pass
    else:
        complementarities = not ignore_constraints

    # Linear solver config
    if isinstance(linsolve, str):
        linsolve_cfg = {
            "method": "neumann" if linsolve == "iti" else linsolve,
            "maxit": 50,
            "tol": 1e-8,
            "precondition": True,
        }
    elif isinstance(linsolve, dict):
        linsolve_cfg = {
            "method": "gmres",
            "maxit": 50,
            "tol": 1e-8,
            "precondition": True,
            **linsolve,
        }
        if linsolve_cfg["method"] == "iti":
            linsolve_cfg["method"] = "neumann"
    else:
        linsolve_cfg = {
            "method": "gmres",
            "maxit": 50,
            "tol": 1e-8,
            "precondition": True,
        }

    if invmethod is not None:
        linsolve_cfg["method"] = "neumann" if invmethod == "iti" else invmethod
    if smaxit is not None:
        linsolve_cfg["maxit"] = smaxit
    if linear_maxit is not None:
        linsolve_cfg["maxit"] = linear_maxit

    # Line search config
    if linesearch is False or linesearch is None or linesearch == "none":
        linesearch_cfg = {"method": None, "max_backsteps": 0}
    elif isinstance(linesearch, str):
        linesearch_cfg = {"method": linesearch, "max_backsteps": 10}
    elif isinstance(linesearch, dict):
        linesearch_cfg = {"method": "backtracking", "max_backsteps": 10, **linesearch}
    else:
        linesearch_cfg = {"method": "backtracking", "max_backsteps": 10}

    if max_backsteps is not None:
        linesearch_cfg["max_backsteps"] = max_backsteps
    elif maxbsteps is not None:
        linesearch_cfg["max_backsteps"] = maxbsteps

    itprint = IterationsPrinter(
        ("N", int),
        ("f_x", float),
        ("d_x", float),
        ("Time_residuals", float),
        ("Time_inversion", float),
        ("Time_search", float),
        ("Lambda_0", float),
        ("N_invert", int),
        ("N_search", int),
        verbose=verbose,
    )
    itprint.print_header("Start Newton Iterations.")

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

    endo_grid = grid["endo"]
    exo_grid = grid["exo"]

    f = dmodel.f
    g = dmodel.g
    x_lb = dmodel.model.functions.get("arbitrage_lb", None)
    x_ub = dmodel.model.functions.get("arbitrage_ub", None)

    parms = dmodel.parms

    n_m = max(dprocess.n_nodes, 1)
    n_s = len(dmodel.model.symbols["states"])

    eng = _resolve_engine(dmodel, engine, backend)
    use_jax = eng in ("jax", "gpu", "cpu")
    if use_jax:
        target_dev = device
        if target_dev is None:
            if eng == "gpu":
                import jax

                gpus = jax.devices("gpu")
                if not gpus:
                    raise RuntimeError(
                        "GPU requested (engine='gpu'), but no CUDA device was found."
                    )
                target_dev = gpus[0]
            elif eng == "cpu":
                import jax

                target_dev = jax.devices("cpu")[0]
    else:
        target_dev = None

    if interpolation in ("cubic", "linear"):
        ddr = DecisionRule(
            exo_grid, endo_grid, dprocess=dprocess, interp_method=interpolation
        )
        ddr_filt = DecisionRule(
            exo_grid, endo_grid, dprocess=dprocess, interp_method=interpolation
        )
    else:
        raise Exception("Unsupported interpolation method.")

    s = endo_grid.nodes
    N = s.shape[0]
    n_x = len(dmodel.model.symbols["controls"])
    x0 = (
        dmodel.model.calibration["controls"][
            None,
            None,
        ]
        .repeat(n_m, axis=0)
        .repeat(N, axis=1)
    )

    if dr0 is not None:
        for i_m in range(n_m):
            x0[i_m, :, :] = dr0(i_m, s)
    ddr.set_values(x0)

    if x_lb is not None and complementarities:
        lb = x0.copy()
        ub = x0.copy()
        for i_m in range(n_m):
            m = dprocess.node(i_m)
            lb[i_m, :] = x_lb(m, s, parms)
            ub[i_m, :] = x_ub(m, s, parms)
    else:
        complementarities = False
        lb = None
        ub = None

    x = x0.copy()
    ddr.set_values(x)

    n_im = dprocess.n_inodes(0)
    jres = numpy.zeros((n_m, n_im, N, n_x, n_x))
    S_ij = numpy.zeros((n_m, n_im, N, n_s))

    trace_details = [] if trace else None

    err_0 = 10.0
    err_2 = 10.0

    # JAX engines with the grid kernels: controls, coefficients and derivatives stay on the
    # device; only the errors go to the host
    cr_fast = None
    if use_jax and not complementarities:
        import jax
        import jax.numpy as jnp
        from dolo.numeric.residual_kernels_jax import axpy, block_solve, max_abs as _max_abs

        cr = _get_or_compile_jax_residual(dmodel, interp_method=interpolation)
        cr_fast = cr._fast
    if cr_fast is not None:
        x_dev = jax.device_put(jnp.asarray(x, dtype=jnp.float64), target_dev)

    for it in range(maxit):
        if cr_fast is not None:
            t1 = time.time()
            if trace:
                ddr.set_values(numpy.asarray(x_dev))
                trace_details.append({"dr": copy.deepcopy(ddr)})

            def prefilter(u):
                C = cr.prefilter(u)
                return C[None] if n_m == 1 else C

            def max_abs(a):
                return float(_max_abs(a))

            # 1. Residuals and Jacobian, at x with the decision rule interpolating x
            coeffs = prefilter(x_dev)
            R, J = cr_fast.F_and_dF_1(x_dev, coeffs)
            err_0 = max_abs(R)
            if err_0 < tol_eps:
                t2 = time.time()
                itprint.print_iteration(
                    N=it,
                    f_x=err_0,
                    d_x=0.0,
                    Time_residuals=t2 - t1,
                    Time_inversion=0.0,
                    Time_search=0.0,
                    Lambda_0=0.0,
                    N_invert=0,
                    N_search=0,
                )
                break
            t2 = time.time()

            # 2. Block-diagonal solve: ddx = dres^{-1} * res
            ddx = block_solve(J, R)

            # 3. Policy linear operator dF_2 and preconditioned inversion
            solver_method = linsolve_cfg.get("method", "gmres")
            solver_maxit = linsolve_cfg.get("maxit", 50)
            solver_tol = linsolve_cfg.get("tol", 1e-8)
            ttol = max(err_0 / 100, solver_tol)
            F_c, S = cr_fast.dF_2_compressed(x_dev, coeffs)
            if n_m == 1:
                F_c, S = F_c[0], S[0]
            L_last = None  # free the operator of the previous iteration
            L_op = PolicyLinearOperator(
                dmodel,
                F_c,
                S,
                engine=eng,
                interp_method=interpolation,
                device=target_dev,
                cr=cr,
                cols=cr_fast.xcols,
            )
            del F_c, R
            L_prec = (-1.0) * L_op.preconditioned(J)
            del L_op, J
            tot = L_prec.solve(ddx, linsolve=solver_method, tol=ttol, maxit=solver_maxit)
            L_last = L_prec
            del L_prec, ddx
            nn = solver_maxit
            lam0 = 0.01
            t3 = time.time()

            # 4. Line search on the residual of the candidate with its own decision rule
            ls_method = linesearch_cfg.get("method", "backtracking")
            ls_maxsteps = linesearch_cfg.get("max_backsteps", 10)

            def eval_res_norm_jax(cand_x):
                return max_abs(cr_fast.F(cand_x, prefilter(cand_x)))

            if ls_method == "golden_section":
                cand_1 = eval_res_norm_jax(axpy(x_dev, -1.0, tot))
                if cand_1 < err_0:
                    best_alpha = 1.0
                    i_bckstps = 0
                else:
                    best_alpha, _ = golden_section_search(
                        lambda a: eval_res_norm_jax(axpy(x_dev, -a, tot)),
                        alpha_min=0.0,
                        alpha_max=1.0,
                        maxiter=ls_maxsteps,
                    )
                    i_bckstps = ls_maxsteps
                new_x = axpy(x_dev, -best_alpha, tot)
            elif ls_method == "backtracking":
                steps = 0.5 ** numpy.arange(max(ls_maxsteps, 1))
                new_x = axpy(x_dev, -1.0, tot)
                i_bckstps = 0
                for i_bckstps, lam in enumerate(steps):
                    new_x = axpy(x_dev, -lam, tot)
                    if eval_res_norm_jax(new_x) < err_0:
                        break
            else:
                new_x = axpy(x_dev, -1.0, tot)
                i_bckstps = 0

            err_2 = max_abs(tot)
            t4 = time.time()
            itprint.print_iteration(
                N=it,
                f_x=err_0,
                d_x=err_2,
                Time_residuals=t2 - t1,
                Time_inversion=t3 - t2,
                Time_search=t4 - t3,
                Lambda_0=lam0,
                N_invert=nn,
                N_search=i_bckstps,
            )
            x_dev = new_x
            if err_0 < tol_eps or err_2 < tol_eta:
                break

        elif use_jax and not complementarities:
            import jax
            import jax.numpy as jnp

            t1 = time.time()
            ddr.set_values(x)

            if trace:
                trace_details.append({"dr": copy.deepcopy(ddr)})

            # 1. Residuals and Jacobian
            res, dres = F_and_dF_1(
                dmodel,
                x,
                dr=x,
                engine=eng,
                interp_method=interpolation,
                device=target_dev,
            )
            res_arr = numpy.asarray(res.data if isinstance(res, GArray) else res)
            dres_arr = numpy.asarray(dres.data if isinstance(dres, GArray) else dres)
            if n_m == 1:
                if res_arr.ndim == 3:
                    res_arr = res_arr[0]
                if dres_arr.ndim == 4:
                    dres_arr = dres_arr[0]

            err_0 = abs(res_arr).max()
            if err_0 < tol_eps:
                t2 = time.time()
                t3 = t2
                t4 = t2
                itprint.print_iteration(
                    N=it,
                    f_x=err_0,
                    d_x=0.0,
                    Time_residuals=t2 - t1,
                    Time_inversion=0.0,
                    Time_search=0.0,
                    Lambda_0=0.0,
                    N_invert=0,
                    N_search=0,
                )
                break

            t2 = time.time()

            # 2. Block-diagonal solve: ddx = dres^{-1} * res
            J = jnp.asarray(dres_arr)
            R = jnp.asarray(res_arr)
            ddx = numpy.asarray(jnp.linalg.solve(J, R[..., None])[..., 0])

            # 3. Policy linear operator dF_2 and preconditioned inversion
            solver_method = linsolve_cfg.get("method", "gmres")
            solver_maxit = linsolve_cfg.get("maxit", 50)
            solver_tol = linsolve_cfg.get("tol", 1e-8)
            ttol = max(err_0 / 100, solver_tol)

            L_op = dF_2(
                dmodel,
                x,
                dr=x,
                engine=eng,
                interp_method=interpolation,
                device=target_dev,
            )
            L_prec = (-1.0) * L_op.preconditioned(J)
            # free the device memory of the unpreconditioned operator before the solve
            del L_op
            tot_2d = L_prec.solve(
                ddx, linsolve=solver_method, tol=ttol, maxit=solver_maxit
            )
            tot = numpy.asarray(tot_2d).reshape(x.shape)
            L_last = L_prec
            nn = solver_maxit
            lam0 = 0.01

            t3 = time.time()

            # 4. Line search
            ls_method = linesearch_cfg.get("method", "backtracking")
            ls_maxsteps = linesearch_cfg.get("max_backsteps", 10)

            def eval_res_norm_jax(cand_x):
                # residual of the candidate with its own decision rule
                c_res = F(
                    dmodel,
                    cand_x,
                    engine=eng,
                    interp_method=interpolation,
                    device=target_dev,
                )
                c_arr = numpy.asarray(
                    c_res.data if isinstance(c_res, GArray) else c_res
                )
                return abs(c_arr).max()

            if ls_method == "golden_section":
                cand_1 = eval_res_norm_jax(x - tot)
                if cand_1 < err_0:
                    best_alpha = 1.0
                    i_bckstps = 0
                else:
                    best_alpha, _ = golden_section_search(
                        lambda a: eval_res_norm_jax(x - tot * a),
                        alpha_min=0.0,
                        alpha_max=1.0,
                        maxiter=ls_maxsteps,
                    )
                    i_bckstps = ls_maxsteps
                new_x = x - tot * best_alpha

            elif ls_method == "backtracking":
                steps = 0.5 ** numpy.arange(max(ls_maxsteps, 1))
                new_x = x - tot
                i_bckstps = 0
                for i_bckstps, lam in enumerate(steps):
                    new_x = x - tot * lam
                    new_err = eval_res_norm_jax(new_x)
                    if new_err < err_0:
                        break
            else:
                new_x = x - tot
                i_bckstps = 0

            err_2 = abs(tot).max()
            t4 = time.time()

            itprint.print_iteration(
                N=it,
                f_x=err_0,
                d_x=err_2,
                Time_residuals=t2 - t1,
                Time_inversion=t3 - t2,
                Time_search=t4 - t3,
                Lambda_0=lam0,
                N_invert=nn,
                N_search=i_bckstps,
            )

            x = new_x
            if err_0 < tol_eps or err_2 < tol_eta:
                break

        else:
            jres[...] = 0.0
            S_ij[...] = 0.0

            t1 = time.time()
            ddr.set_values(x)

            if trace:
                trace_details.append({"dr": copy.deepcopy(ddr)})

            sh_x = x.shape
            ff = SerialDifferentiableFunction(
                lambda u: euler_residuals(
                    f,
                    g,
                    s,
                    u.reshape(sh_x),
                    ddr,
                    dprocess,
                    parms,
                    diff=False,
                    with_jres=False,
                    set_dr=False,
                ).reshape((-1, sh_x[2]))
            )
            res, dres = ff(x.reshape((-1, sh_x[2])))
            res = res.reshape(sh_x)
            dres = dres.reshape((sh_x[0], sh_x[1], sh_x[2], sh_x[2]))
            junk, jres, fut_S = euler_residuals(
                f,
                g,
                s,
                x,
                ddr,
                dprocess,
                parms,
                diff=False,
                with_jres=True,
                set_dr=False,
                jres=jres,
                S_ij=S_ij,
            )

            if complementarities:
                res, dres, jres = smooth(res, dres, jres, x - lb)
                res[...] *= -1
                dres[...] *= -1
                jres[...] *= -1
                res, dres, jres = smooth(res, dres, jres, ub - x, pos=-1.0)
                res[...] *= -1
                dres[...] *= -1
                jres[...] *= -1

            err_0 = abs(res).max()

            jres[...] *= -1.0

            for i_m in range(n_m):
                for j_m in range(n_im):
                    M = jres[i_m, j_m, :, :, :]
                    X = dres[i_m, :, :, :].copy()
                    sol = solve_tensor(X, M)

            t2 = time.time()

            # Inversion of (I - L) dx = F_1^{-1} res
            solver_method = linsolve_cfg.get("method", "gmres")
            solver_maxit = linsolve_cfg.get("maxit", 50)
            solver_tol = linsolve_cfg.get("tol", 1e-8)

            if solver_method == "gmres":
                ddx = solve_gu(dres.copy(), res.copy())
                L = Operator(jres, fut_S, ddr_filt)
                n0 = L.counter
                L.addid = True
                ttol = max(err_0 / 100, solver_tol)
                try:
                    sol, info = scipy.sparse.linalg.gmres(
                        L, ddx.ravel(), rtol=ttol, atol=0.0, maxiter=solver_maxit
                    )
                except TypeError:
                    sol, info = scipy.sparse.linalg.gmres(
                        L, ddx.ravel(), tol=ttol, maxiter=solver_maxit
                    )
                lam0 = 0.01
                nn = L.counter - n0
                tot = sol.reshape(ddx.shape)
            else:
                # Neumann series
                tot, nn, lam0 = invert_jac(
                    res,
                    dres,
                    jres,
                    fut_S,
                    ddr_filt,
                    tol=solver_tol,
                    maxit=solver_maxit,
                    verbose=(verbose == "full"),
                )

            t3 = time.time()

            # Step search / Line search
            ls_method = linesearch_cfg.get("method", "backtracking")
            ls_maxsteps = linesearch_cfg.get("max_backsteps", 10)

            def eval_residual_norm(cand_x):
                cand_res = euler_residuals(
                    f, g, s, cand_x, ddr, dprocess, parms, diff=False, set_dr=True
                )
                if complementarities:
                    cand_res = smooth_nodiff(cand_res, cand_x - lb)
                    cand_res = smooth_nodiff(-cand_res, ub - cand_x)
                return abs(cand_res).max()

            if ls_method == "golden_section":
                # Check full step first
                cand_1 = eval_residual_norm(x - tot)
                if cand_1 < err_0:
                    best_alpha = 1.0
                    i_bckstps = 0
                else:
                    best_alpha, _ = golden_section_search(
                        lambda a: eval_residual_norm(x - tot * a),
                        alpha_min=0.0,
                        alpha_max=1.0,
                        maxiter=ls_maxsteps,
                    )
                    i_bckstps = ls_maxsteps
                new_x = x - tot * best_alpha

            elif ls_method == "backtracking":
                steps = 0.5 ** numpy.arange(max(ls_maxsteps, 1))
                new_x = x - tot
                for i_bckstps, lam in enumerate(steps):
                    new_x = x - tot * lam
                    new_err = eval_residual_norm(new_x)
                    if new_err < err_0:
                        break
            else:
                # No line search (full Newton step)
                new_x = x - tot
                i_bckstps = 0

            err_2 = abs(tot).max()
            t4 = time.time()

            itprint.print_iteration(
                N=it,
                f_x=err_0,
                d_x=err_2,
                Time_residuals=t2 - t1,
                Time_inversion=t3 - t2,
                Time_search=t4 - t3,
                Lambda_0=lam0,
                N_invert=nn,
                N_search=i_bckstps,
            )

            if err_0 < tol_eps:
                break

            x = new_x

    if cr_fast is not None:
        x = numpy.asarray(x_dev)
    ddr.set_values(x)
    if trace:
        trace_details.append({"dr": copy.deepcopy(ddr)})

    itprint.print_finished()

    if not details:
        return ddr
    else:
        if use_jax and not complementarities:
            L = L_last if "L_last" in locals() else None
            if compute_radius and L is not None:
                try:
                    lam_eig = scipy.sparse.linalg.eigs(
                        L, k=1, return_eigenvectors=False
                    )
                    lam = abs(lam_eig[0])
                except Exception:
                    lam = numpy.nan
            else:
                lam = numpy.nan
        else:
            L = Operator(jres, fut_S, ddr_filt)
            if compute_radius:
                try:
                    lam_eig = scipy.sparse.linalg.eigs(
                        L, k=1, return_eigenvectors=False
                    )
                    lam = abs(lam_eig[0])
                except Exception:
                    lam = numpy.nan
            else:
                lam = numpy.nan

        return NewtonResult(
            ddr,
            it + 1,
            err_0,
            err_2,
            err_0 < tol_eps,
            complementarities,
            lam,
            trace_details,
            L,
        )


def euler_residuals(
    f,
    g,
    s,
    x,
    dr,
    dp,
    p_,
    diff=True,
    with_jres=False,
    set_dr=True,
    jres=None,
    S_ij=None,
):
    if set_dr:
        dr.set_values(x)

    N = s.shape[0]
    n_s = s.shape[1]
    n_x = x.shape[2]

    n_ms = max(dp.n_nodes, 1)
    n_im = dp.n_inodes(0)

    res = numpy.zeros_like(x)

    if with_jres:
        if jres is None:
            jres = numpy.zeros((n_ms, n_im, N, n_x, n_x))
        if S_ij is None:
            S_ij = numpy.zeros((n_ms, n_im, N, n_s))

    for i_ms in range(n_ms):
        m_ = dp.node(i_ms)
        xm = x[i_ms, :, :]
        for I_ms in range(n_im):
            M_ = dp.inode(i_ms, I_ms)
            w = dp.iweight(i_ms, I_ms)
            S = g(m_, s, xm, M_, p_, diff=False)
            XM = dr.eval_ijs(i_ms, I_ms, S)
            if with_jres:
                ff = SerialDifferentiableFunction(
                    lambda u: f(m_, s, xm, M_, S, u, p_, diff=False)
                )
                rr, rr_XM = ff(XM)
                jres[i_ms, I_ms, :, :, :] = w * rr_XM
                S_ij[i_ms, I_ms, :, :] = S
            else:
                rr = f(m_, s, xm, M_, S, XM, p_, diff=False)
            res[i_ms, :, :] += w * rr

    if with_jres:
        return res, jres, S_ij
    else:
        return res


improved_time_iteration = newton
