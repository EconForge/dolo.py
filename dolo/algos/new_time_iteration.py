from numba import _ensure_critical_deps
from dolo.compiler.model import Model

import numpy
from dolo import dprint
from dolo.compiler.model import Model
from dolo.numeric.processes import DiscretizedIIDProcess, DiscretizedProcess
from dolo.numeric.decision_rule import DecisionRule, Linear
from dolo.numeric.grids import CartesianGrid
from dolo.numeric.serial_operations import serial_multiplication_vector, serial_solve
from dolo.numeric.optimize.newton import SerialDifferentiableFunction


class CVector:

    data: object # vector


    def __init__(self, x0):

        assert(x0.ndim>=3)

        self.x0 = x0

        self.datashape = x0.shape[2:]
        self.shape = x0.shape[:2]

        self.data = x0.reshape((-1,)+self.datashape)

        # self.views = [x0[i,:,:] for i in range(x0.shape[0])]

    def __str__(self):

        s = f"CVector(shape={self.shape}, datashape={self.datashape})"
        return s

    def zeros_like(self):

        x1 = self.x0*0
        return CVector(x1)

    def __getitem__(self, args):
        if len(args)!=3 or (args[1]!=slice(None,None)) or (args[2]!=slice(None,None)):
            raise Exception("Unsupported Subscripts.")
        else:
            i = args[0]
            return self.x0[i,:,:]


    def __setitem__(self, args, value):
        if len(args)!=3 or (args[1]!=slice(None,None)) or (args[2]!=slice(None,None)):
            raise Exception("Unsupported Subscripts.")
        else:
            i = args[0]
            self.x0[i,:,:] = value


    def __mul__(self, x):

        if isinstance(x,CVector):
            assert(self.shape == x.shape)
            x = x.x0
        return CVector(self.x0*x)

    def __rmul__(self, x):
        if isinstance(x,CVector):
            assert(self.shape == x.shape)
            x = x.x0
        return CVector(self.x0*x)

    def __add__(self, x):
        if isinstance(x,CVector):
            assert(self.shape == x.shape)
            x = x.x0
        return CVector(self.x0+x)

    def __radd__(self, x):
        if isinstance(x,CVector):
            assert(self.shape == x.shape)
            x = x.x0
        return CVector(self.x0+x)

    def __sub__(self, x):
        if isinstance(x,CVector):
            assert(self.shape == x.shape)
            x = x.x0
        return CVector(self.x0-x)

    def __truediv__(self, x):
        if isinstance(x,CVector):
            assert(self.shape == x.shape)
            x = x.x0
        return CVector(self.x0/x)


    def __matmul__(self, x):

        assert(isinstance(x, CVector))

        A = self.data
        B = x.data
        from dolo.numeric.serial_operations import serial_multiplication, serial_multiplication_vector
        
        if len(A.shape) == len(B.shape) == 3:
            C = serial_multiplication(A,B)
        elif len(A.shape)==3 and len(B.shape)==2:
            C = serial_multiplication_vector(A,B)

        return CVector(C.reshape(self.shape+C.shape[1:]))


    def norm(self):
        return abs(self.data).max()

    def solve(self, x):

        assert(isinstance(x, CVector))

        A = self.data
        B = x.data

        from dolo.numeric.serial_operations import serial_solve_numba
        C = serial_solve(A,B)

        # from dolo.algos.improved_time_iteration import solve_gu 
        # C = solve_gu(A.copy(),B.copy())
        
        return CVector(C.reshape(self.shape+C.shape[1:]))




class Euler:

    model: Model
    grid: object
    dprocess: DiscretizedProcess
    dr: DecisionRule
    x0: object   # CVector[ N_e × N , n_x ]
    p: object   # n_p vector

    def __init__(self, model, grid=dict(), interp_method='cubic', dr0=None):

        self.model = model
        grid, dprocess = model.discretize(**grid)
        self.grid = grid
        self.dprocess = dprocess

        self.dr = DecisionRule(grid['exo'], grid['endo'], dprocess=dprocess, interp_method=interp_method)

        n_x = len(model.symbols["controls"])
        N_e = max(1,grid['exo'].n_nodes)
        N = grid['endo'].n_nodes

        x0 = numpy.zeros((N_e, N, n_x))
        x0[:, :, :] = model.calibration["controls"][None, None, :]


        if dr0 is not None:
            # TODO: revisit this, if dr0, is a d.r. we need to check what inputs it accepts
            s = grid['endo'].nodes
            for i_m in range(dprocess.n_nodes):
                    m = dprocess.node(i_m)
                    x0[i_m, :, :] = dr0(m, s)
            # try:
            #     for i_m in range(dprocess.n_nodes):
            #         m = dprocess.node(i_m)
            #         x0[i_m, :, :] = dr0(m, s)
            # except Exception:
            #     for i_m in range(dprocess.n_nodes):
            #         x0[i_m, :, :] = dr0(i_m, s)


        self.x0 = CVector(x0)
        
        self.p = model.calibration['parameters']
        self.sh_C = (N_e*N, n_x)

    def __call__(self, x0, x1=None):


        f = self.model.functions['arbitrage']
        g = self.model.functions['transition']

        s = self.grid["endo"].nodes      
        dp = self.dprocess
        p = self.p


        if x1 is not None:
            self.dr.set_values(x1.data.reshape(x1.x0.shape))
        

        return residuals_simple(f, g, s, x0, self.dr, dp, p)

    def d_A(self, x0, x1=None):

        sh = x0.x0.shape

        def fun(x):
            return self( CVector( x.reshape( sh )) ).data.copy()

        z0 = x0.data.copy()
        
        from dolo.numeric.optimize.newton import serial_newton

        dfun = SerialDifferentiableFunction(fun)
    
        r, J = dfun(z0)

        R = CVector(r.reshape(x0.x0.shape))
        J = CVector(J.reshape(x0.shape + J.shape[1:]))

        return R,J
    
    def d_B(self, x0, x1=None):





        f = self.model.functions['arbitrage']
        g = self.model.functions['transition']

        s = self.grid["endo"].nodes      
        dp = self.dprocess
        p = self.p
        dr = self.dr

        ## 
        x = x0.x0


        res, jres, S_ij = euler_residuals(f, g, s, x, dr, dp, p, diff=True, with_jres=True, set_dr=False)


        import copy
        cdr = copy.deepcopy(dr)

        R = CVector(res.reshape(x.shape))
        L = Jacobian(jres, S_ij, cdr)

        return R, L


class Jacobian:

    def __init__(self, M_ij, S_ij, cdr):

        self.M_ij = M_ij
        self.S_ij = S_ij
        self.cdr = cdr

    def __matmul__(self, A, inplace=False):

        π = A.x0

        M_ij = self.M_ij
        S_ij = self.S_ij

        if not inplace:
            π = π.copy()

        from dolo.algos.improved_time_iteration import d_filt_dx

        d_filt_dx(π, M_ij, S_ij, self.cdr)

        return CVector(π)

    def ldiv(self, A): # performs the computation in place

        from .invert import solve_tensor

        M_ij = self.M_ij
        n_m, n_im = self.M_ij.shape[:2]

        V = A.x0

        for i_m in range(n_m):
            for j_m in range(n_im):
                M = M_ij[i_m, j_m, :, :, :]
                X = V[i_m, :, :, :].copy()
                rhs = solve_tensor(X, M)

    def __imul__(self, other):

        self.M_ij[...] *= other
        return self

def residuals_simple(f, g, s, x, dr, dprocess, parms)->CVector:

    N = s.shape[0]
    n_s = s.shape[1]

    res = x.zeros_like()

    for i_ms in range(dprocess.n_nodes):

        # solving on grid for markov index i_ms
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


import time

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


import copy
from .results import ImprovedTimeIterationResult, TimeIterationResult, AlgoResult
from dolo.misc.itprinter import IterationsPrinter
from dolo.numeric.optimize.newton import newton


def newton(f, x: CVector, verbose=False, tol=1e-6, maxit=5, jactype="serial"):

    """Solve nonlinear system using safeguarded Newton iterations


    Parameters
    ----------

    Return
    ------
    """
    if verbose:
        print = lambda txt: print(txt)
    else:
        print = lambda txt: None

    it = 0
    error = 10
    converged = False
    maxbacksteps = 30

    x0 = x

    while it < maxit and not converged:

        [v, dv] = f(x)

        # TODO: rewrite starting here

        #        print("Time to evaluate {}".format(ss-tt)0)

        error_0 = (v).norm()

        if error_0 < tol:

            if verbose:
                print(
                    "> System was solved after iteration {}. Residual={}".format(
                        it, error_0
                    )
                )
            converged = True

        else:

            it += 1

            dx = dv.solve(v)

            # dx = solve(dv, v)

            # norm_dx = abs(dx).max()

            for bck in range(maxbacksteps):
                xx = x - dx * (2 ** (-bck))
                vm = f(xx)[0]
                err = (vm).norm()
                if err < error_0:
                    break

            x = xx

            if verbose:
                print("\t> {} | {} | {}".format(it, err, bck))

    if not converged:
        import warnings

        warnings.warn("Did not converge")
    return (x, it)

def time_iteration(
    model: Model,
    *,
    dr0: DecisionRule = None,  #
    grid: dict = dict(),
    verbose: bool = True,  #
    details: bool = True,  #
    ignore_constraints=False,  #
    interp_method="cubic",
    trace: bool = False,  #
    maxit=1000,
    tol_η=1e-6,
    tol_ε=1e-6,
    inner_maxit=5,
    hook=None,
) -> TimeIterationResult:


    F = Euler(model, grid=grid, interp_method=interp_method, dr0=dr0)

    x0 = F.x0

    complementarities = not ignore_constraints

    trace_details = []

    verbit = verbose == "full"


    itprint = IterationsPrinter(
        ("n", int),
        ("f_x", (float,"εₙ=|f(xₙ)|")),
        ("d_x", (float,"ηₙ=|xₙ-xₙ₋₁|")),
        ("λ", (float,"λₙ=ηₙ/ηₙ₋₁")),
        ("Time", float),
        ("nit", int),
        verbose=verbose,
    )
    
    itprint.print_header("Time Iterations.")

    err_η_0 = numpy.nan

    n_x = len(model.symbols["controls"])

    for  it in range(maxit):

        if hook:
            hook()

        t_start = time.time()

        if trace:
            trace_details.append({"dr": copy.deepcopy(F.dr)})

        r = ( F(x0, x0) )
        err_ε = r.norm()

            # r,J = F.d_A(su)

        verbose=False

        x1, nit = newton(F.d_A, x0, maxit=inner_maxit)


        # baby-steps version
        # r, J = ( F.d_A(x0) )
        # dx = J.solve(r)
        # err_η = dx.norm()
        # x1 = x0 - dx

        dx = x1 - x0
        err_η = dx.norm()

        λ = err_η/err_η_0
        err_η_0 = err_η


        t_finish = time.time()
        elapsed = t_finish -t_start

        itprint.print_iteration(n=it, f_x=err_ε, d_x=err_η, λ=λ, Time=elapsed, nit=nit),

        if err_ε<tol_ε or err_η<tol_η:
            break

        x0 = x1

    dr = F.dr

    itprint.print_finished()


    if not details:
        return dr
    
    return TimeIterationResult(
        dr,
        it,
        complementarities,
        F.dprocess,
        err_η < tol_η,  # x_converged: bool
        tol_η,  # x_tol
        err_η,  #: float
        None,  # log: object # TimeIterationLog
        trace_details,  # trace: object #{Nothing,IterationTrace}
    )


import numpy as np

def improved_time_iteration(
    model: Model,
    *,
    dr0: DecisionRule = None,  #
    grid: dict = dict(),
    verbose: bool = True,  #
    details: bool = True,  #
    ignore_constraints=False,  #
    interp_method="cubic",
    maxbsteps=10,
    tol_ε=1e-8,
    tol_ν=1e-10,
    smaxit=500,
    maxit=1000,
    compute_radius=False,
    # invmethod="iti",
) -> ImprovedTimeIterationResult:


    F = Euler(model, grid=grid, interp_method=interp_method, dr0=dr0)

    x0 = F.x0

    complementarities = not ignore_constraints

    itprint = IterationsPrinter(
        ("n", int),
        # ("εₙ=|f(xₙ)|", float),
        # ("ηₙ=|f(xₙ)-f(xₙ₋₁)|", float),
        ("f_x", (float,"εₙ=|f(xₙ)|")),
        ("d_x", (float,"ηₙ=|xₙ-xₙ₋₁|")),
        # ("Time_residuals", float),
        # ("Time_inversion", float),
        ("λ", (float, "λ≈|T'(xₙ)|")),
        ("Time", float),
        ("N_invert", int),
        verbose=verbose,
    )
    itprint.print_header("Improved Time Iterations.")


    for it in range(maxit):

        t_start = time.time()

        dr = F(x0, x0)
        err_ε = dr.norm()

        R, J = F.d_A(x0)
        R, L = F.d_B(x0)

        dr = J.solve(dr)
        L.ldiv(J)
        L *= -1

        # compute dx such that: (I-L).dx = dr
        dx = dr
        du = dr
        err_ν_0 = 1.0
        for n in range(smaxit):
            du = L@du
            dx = dx + du
            err_ν = du.norm()
            λ = err_ν/err_ν_0
            err_ν_0 = err_ν
            if err_ν<tol_ν:
                break

        err_η = dx.norm()

        x0 = x0 - dx

        t_finish = time.time()

        itprint.print_iteration(
            n=it,
            f_x=err_ε,
            d_x=err_η,
            λ=λ,
            Time=t_finish-t_start,
            N_invert=n
        )
        if err_ε<tol_ε:
            # print(f"Error {k} = {err_ε} |")
            break

    # F.dr.set_values(x0)

    dr = F.dr

    itprint.print_finished()


    if not details:
        return dr
    else:
        if compute_radius:
            raise Exception("Not implemented.")
        else:
            lam = np.nan
        return ImprovedTimeIterationResult(
            dr, it, err_ε, err_η, err_ε < tol_ε, complementarities, lam, None, L
        )