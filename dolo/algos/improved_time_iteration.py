"""Do I need a docstring here ?"""

from .bruteforce_lib import *
from .invert import *


from dolo.numeric.decision_rule import DecisionRule
from dolo.misc.itprinter import IterationsPrinter
from numba import jit
import numpy
import time
import scipy.sparse.linalg

from operator import mul
from functools import reduce

from dolo.numeric.optimize.newton import SerialDifferentiableFunction


def prod(l):
    return reduce(mul, l)


from math import sqrt
from numba import jit
import time
from numpy import array, zeros

import time


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

    # jres is modified
    dinf = dx > 100000
    n_m, N, n_x = res.shape
    sq = sqrt(res ** 2 + (dx) ** 2)
    H = res + (dx) - sq
    Phi_a = 1 - res / sq
    Phi_b = 1 - (dx) / sq
    H[dinf] = res[dinf]
    Phi_a[dinf] = 1.0
    Phi_b[dinf] = 0.0
    H_x = Phi_a[:, :, :, None] * dres
    for i_x in range(n_x):
        H_x[:, :, i_x, i_x] += Phi_b[:, :, i_x] * pos
    # H_xt = Phi_a[:,None,:,:,None]*jres
    inplace(Phi_a, jres)
    return H, H_x, jres
    # return H, H_x, H_xt


def smooth_nodiff(res, dx):
    from numpy import sqrt

    n_m, N, n_x = res.shape
    dinf = dx > 100000
    sq = sqrt(res ** 2 + (dx) ** 2)
    H = res + (dx) - sq
    H[dinf] = res[dinf]
    return H


@jit
def ssmul(A, B):
    # simple serial_mult (matrix times vector)
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
    # simple serial_mult (matrix times vector)
    N, a, b = A.shape
    NN, b = B.shape
    # O = numpy.zeros((N,a))
    for n in range(N):
        for k in range(a):
            for l in range(b):
                O[n, k] += A[n, k, l] * B[n, l]
    return O


# make parallel using guvectorize ?
def d_filt_dx(π, M_ij, S_ij, n_m, N, n_x, dumdr):
    # OK, so res is probably not what we need to filter here.
    # s sh
    n_m, n_im = M_ij.shape[:2]
    dumdr.set_values(π)
    i = 0
    j = 0
    for i in range(n_m):
        π[i, :, :] = 0
        for j in range(n_im):
            A = M_ij[i, j, :, :, :]
            B = dumdr.eval_ijs(i, j, S_ij[i, j, :, :])
            π[i, :, :] += ssmul(A, B)
    return π


from scipy.sparse.linalg import LinearOperator


class Operator(LinearOperator):

    """Special Linear Operator"""

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

    @property
    def shape(self):
        nn = self.n_m * self.N * self.n_x
        return (nn, nn)

    def _matvec(self, x):
        self.counter += 1
        xx = x.reshape((self.n_m, self.N, self.n_x))
        yy = self.apply(xx)
        if self.addid:
            yy = xx - yy  # preconditioned system
        return yy.ravel()

    def apply(self, π, inplace=False):
        M_ij = self.M_ij
        S_ij = self.S_ij
        n_m = self.n_m
        N = self.N
        n_x = self.n_x
        dumdr = self.dumdr
        if not inplace:
            π = π.copy()
        return d_filt_dx(π, M_ij, S_ij, n_m, N, n_x, dumdr)

    def as_matrix(self):

        arg = np.zeros((self.n_m, self.N, self.n_x))
        larg = arg.ravel()
        N = len(larg)
        J = numpy.zeros((N, N))
        for i in range(N):
            if i > 0:
                larg[i - 1] = 0.0
            larg[i] = 1.0
            J[:, i] = self.apply(arg).ravel()
        return J


def invert_jac(res, dres, jres, fut_S, dumdr, tol=1e-10, maxit=1000, verbose=False):

    n_m = res.shape[0]
    N = res.shape[1]
    n_x = res.shape[2]

    err0 = 0.0
    ddx = solve_gu(dres.copy(), res.copy())

    lam = -1.0
    lam_max = -1.0
    err_0 = abs(ddx).max()

    tot = ddx.copy()
    if verbose:
        print("Starting inversion")
    for nn in range(maxit):
        # operations are made in place in ddx
        ddx = d_filt_dx(ddx, jres, fut_S, n_m, N, n_x, dumdr)
        err = abs(ddx).max()
        lam = err / err_0
        lam_max = max(lam_max, lam)
        if verbose:
            print("- {} | {} | {}".format(err, lam, lam_max))
        tot += ddx
        err_0 = err
        if err < tol:
            break

    # tot += ddx*lam/(1-lam)
    return tot, nn, lam


def radius_jac(res, dres, jres, fut_S, dumdr, tol=1e-10, maxit=1000, verbose=False):

    from numpy import sqrt

    n_m = res.shape[0]
    N = res.shape[1]
    n_x = res.shape[2]

    err0 = 0.0

    norm2 = lambda m: sqrt((m ** 2).sum())

    import numpy.random

    π = (numpy.random.random(res.shape) * 2 - 1) * 1
    π /= norm2(π)

    verbose = True
    lam = 1.0
    lam_max = 0.0

    lambdas = []
    if verbose:
        print("Starting inversion")
    for nn in range(maxit):
        # operations are made in place in ddx
        # π = (numpy.random.random(res.shape)*2-1)*1
        # π /= norm2(π)
        π[:, :, :] /= lam
        π = d_filt_dx(π, jres, fut_S, n_m, N, n_x, dumdr)
        lam = norm2(π)
        lam_max = max(lam_max, lam)
        if verbose:
            print("- {} | {}".format(lam, lam_max))
        lambdas.append(lam)
    return (lam, lam_max, lambdas)


from dolo import dprint
from .results import AlgoResult, ImprovedTimeIterationResult


def improved_time_iteration(
    model,
    method="jac",
    dr0=None,
    dprocess=None,
    interp_method="cubic",
    mu=2,
    maxbsteps=10,
    verbose=False,
    tol=1e-8,
    smaxit=500,
    maxit=1000,
    complementarities=True,
    compute_radius=False,
    invmethod="iti",
    details=True,
):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

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
    itprint.print_header("Start Improved Time Iterations.")

    f = model.functions["arbitrage"]
    g = model.functions["transition"]
    x_lb = model.functions["arbitrage_lb"]
    x_ub = model.functions["arbitrage_ub"]

    parms = model.calibration["parameters"]

    grid, dprocess_ = model.discretize()

    if dprocess is None:
        dprocess = dprocess_

    endo_grid = grid["endo"]
    exo_grid = grid["exo"]

    n_m = max(dprocess.n_nodes, 1)
    n_s = len(model.symbols["states"])

    if interp_method in ("cubic", "linear"):
        ddr = DecisionRule(
            exo_grid, endo_grid, dprocess=dprocess, interp_method=interp_method
        )
        ddr_filt = DecisionRule(
            exo_grid, endo_grid, dprocess=dprocess, interp_method=interp_method
        )
    else:
        raise Exception("Unsupported interpolation method.")

    # s = ddr.endo_grid
    s = endo_grid.nodes
    N = s.shape[0]
    n_x = len(model.symbols["controls"])
    x0 = (
        model.calibration["controls"][
            None,
            None,
        ]
        .repeat(n_m, axis=0)
        .repeat(N, axis=1)
    )

    if dr0 is not None:
        for i_m in range(n_m):
            x0[i_m, :, :] = dr0.eval_is(i_m, s)
    ddr.set_values(x0)

    steps = 0.5 ** numpy.arange(maxbsteps)

    lb = x0.copy()
    ub = x0.copy()
    for i_m in range(n_m):
        m = dprocess.node(i_m)
        lb[i_m, :] = x_lb(m, s, parms)
        ub[i_m, :] = x_ub(m, s, parms)

    x = x0

    # both affect the precision

    ddr.set_values(x)

    ## memory allocation

    n_im = dprocess.n_inodes(0)  # we assume it is constant for now

    jres = numpy.zeros((n_m, n_im, N, n_x, n_x))
    S_ij = numpy.zeros((n_m, n_im, N, n_s))

    for it in range(maxit):

        jres[...] = 0.0
        S_ij[...] = 0.0

        t1 = time.time()

        # compute derivatives and residuals:
        # res: residuals
        # dres: derivatives w.r.t. x
        # jres: derivatives w.r.t. ~x
        # fut_S: future states

        ddr.set_values(x)
        #
        # ub[ub>100000] = 100000
        # lb[lb<-100000] = -100000
        #
        # sh_x = x.shape
        # rr =euler_residuals(f,g,s,x,ddr,dp,parms, diff=False, with_jres=False,set_dr=True)
        # print(rr.shape)
        #
        # from iti.fb import smooth_
        # jj = smooth_(rr, x, lb, ub)
        #
        # print("Errors with complementarities")
        # print(abs(jj.max()))
        #
        # exit()
        #

        from dolo.numeric.optimize.newton import SerialDifferentiableFunction

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

        # if there are complementerities, we modify derivatives
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

        # premultiply by A
        jres[...] *= -1.0

        for i_m in range(n_m):
            for j_m in range(n_im):
                M = jres[i_m, j_m, :, :, :]
                X = dres[i_m, :, :, :].copy()
                sol = solve_tensor(X, M)

        t2 = time.time()

        # new version
        if invmethod == "gmres":
            ddx = solve_gu(dres.copy(), res.copy())
            L = Operator(jres, fut_S, ddr_filt)
            n0 = L.counter
            L.addid = True
            ttol = err_0 / 100
            sol = scipy.sparse.linalg.gmres(
                L, ddx.ravel(), tol=ttol
            )  # , maxiter=1, restart=smaxit)
            lam0 = 0.01
            nn = L.counter - n0
            tot = sol[0].reshape(ddx.shape)
        else:
            # compute inversion
            tot, nn, lam0 = invert_jac(
                res,
                dres,
                jres,
                fut_S,
                ddr_filt,
                tol=tol,
                maxit=smaxit,
                verbose=(verbose == "full"),
            )

        # lam, lam_max, lambdas = radius_jac(res,dres,jres,fut_S,tol=tol,maxit=1000,verbose=(verbose=='full'),filt=ddr_filt)

        # backsteps
        t3 = time.time()
        for i_bckstps, lam in enumerate(steps):
            new_x = x - tot * lam
            new_err = euler_residuals(
                f, g, s, new_x, ddr, dprocess, parms, diff=False, set_dr=True
            )

            if complementarities:
                new_err = smooth_nodiff(new_err, new_x - lb)
                new_err = smooth_nodiff(-new_err, ub - new_x)

            new_err = abs(new_err).max()
            if new_err < err_0:
                break

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
        if err_0 < tol:
            break

        x = new_x

    ddr.set_values(x)

    itprint.print_finished()

    # if compute_radius:
    #     return ddx,L
    #     lam, lam_max, lambdas = radius_jac(res,dres,jres,fut_S,ddr_filt,tol=tol,maxit=smaxit,verbose=(verbose=='full'))
    #     return ddr, lam, lam_max, lambdas
    # else:
    if not details:
        return ddr
    else:
        ddx = solve_gu(dres.copy(), res.copy())
        L = Operator(jres, fut_S, ddr_filt)

        if compute_radius:
            lam = scipy.sparse.linalg.eigs(L, k=1, return_eigenvectors=False)
            lam = abs(lam[0])
        else:
            lam = np.nan
        # lam, lam_max, lambdas = radius_jac(res,dres,jres,fut_S,ddr_filt,tol=tol,maxit=smaxit,verbose=(verbose=='full'))
        return ImprovedTimeIterationResult(
            ddr, it, err_0, err_2, err_0 < tol, complementarities, lam, None, L
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

    t1 = time.time()

    if set_dr:
        dr.set_values(x)

    N = s.shape[0]
    n_s = s.shape[1]
    n_x = x.shape[2]

    n_ms = max(dp.n_nodes, 1)  # number of markov states
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

                rr = f(m_, s, xm, M_, S, XM, p_, diff=False)
                jres[i_ms, I_ms, :, :, :] = w * rr_XM
                S_ij[i_ms, I_ms, :, :] = S
            else:
                rr = f(m_, s, xm, M_, S, XM, p_, diff=False)
            res[i_ms, :, :] += w * rr

    t2 = time.time()

    if with_jres:
        return res, jres, S_ij
    else:
        return res
