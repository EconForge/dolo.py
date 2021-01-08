"""Time Iteration Algorithm"""

import numpy
from dolo import dprint
from dolo.numeric.processes import DiscretizedIIDProcess
from dolo.numeric.decision_rule import DecisionRule
from dolo.numeric.grids import CartesianGrid


def residuals_simple(f, g, s, x, dr, dprocess, parms):

    N = s.shape[0]
    n_s = s.shape[1]

    res = numpy.zeros_like(x)

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


from .results import TimeIterationResult, AlgoResult


def time_iteration(
    model,
    dr0=None,
    with_complementarities=True,
    dprocess=None,
    verbose=True,
    maxit=1000,
    inner_maxit=10,
    tol=1e-6,
    hook=None,
    details=False,
    interp_method="cubic",
):

    """Finds a global solution for ``model`` using backward time-iteration.


    This algorithm iterates on the residuals of the arbitrage equations

    Parameters
    ----------
    model : Model
        model to be solved
    verbose : boolean
        if True, display iterations
    dr0 : decision rule
        initial guess for the decision rule
    with_complementarities : boolean (True)
        if False, complementarity conditions are ignored
    maxit: maximum number of iterations
    inner_maxit: maximum number of iteration for inner solver
    tol: tolerance criterium for successive approximations
    hook: Callable
        function to be called within each iteration, useful for debugging purposes


    Returns
    -------
    decision rule :
        approximated solution
    """

    from dolo import dprint

    def vprint(t):
        if verbose:
            print(t)

    grid, dprocess_ = model.discretize()

    if dprocess is None:
        dprocess = dprocess_

    n_ms = dprocess.n_nodes  # number of exogenous states
    n_mv = dprocess.n_inodes(0)  # this assume number of integration nodes is constant

    x0 = model.calibration["controls"]
    parms = model.calibration["parameters"]
    n_x = len(x0)
    n_s = len(model.symbols["states"])

    endo_grid = grid["endo"]
    exo_grid = grid["exo"]

    mdr = DecisionRule(
        exo_grid, endo_grid, dprocess=dprocess, interp_method=interp_method
    )

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

    if "arbitrage_lb" in model.functions and with_complementarities == True:
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

    else:
        with_complementarities = False

    sh_c = controls_0.shape

    controls_0 = controls_0.reshape((-1, n_x))

    from dolo.numeric.optimize.newton import newton, SerialDifferentiableFunction
    from dolo.numeric.optimize.ncpsolve import ncpsolve

    err = 10
    it = 0

    if with_complementarities:
        lb = lb.reshape((-1, n_x))
        ub = ub.reshape((-1, n_x))

    if verbose:
        headline = "|{0:^4} | {1:10} | {2:8} | {3:8} | {4:3} |".format(
            "N", " Error", "Gain", "Time", "nit"
        )
        stars = "-" * len(headline)
        print(stars)
        print(headline)
        print(stars)

    import time

    t1 = time.time()

    err_0 = numpy.nan

    verbit = verbose == "full"

    while err > tol and it < maxit:

        it += 1

        t_start = time.time()

        mdr.set_values(controls_0.reshape(sh_c))

        fn = lambda x: residuals_simple(
            f, g, s, x.reshape(sh_c), mdr, dprocess, parms
        ).reshape((-1, n_x))
        dfn = SerialDifferentiableFunction(fn)

        res = fn(controls_0)

        if hook:
            hook()

        if with_complementarities:
            [controls, nit] = ncpsolve(
                dfn, lb, ub, controls_0, verbose=verbit, maxit=inner_maxit
            )
        else:
            [controls, nit] = newton(dfn, controls_0, verbose=verbit, maxit=inner_maxit)

        err = abs(controls - controls_0).max()

        err_SA = err / err_0
        err_0 = err

        controls_0 = controls

        t_finish = time.time()
        elapsed = t_finish - t_start

        if verbose:
            print(
                "|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} | {4:3} |".format(
                    it, err, err_SA, elapsed, nit
                )
            )

    controls_0 = controls.reshape(sh_c)

    mdr.set_values(controls_0)

    t2 = time.time()

    if verbose:
        print(stars)
        print("Elapsed: {} seconds.".format(t2 - t1))
        print(stars)

    if not details:
        return mdr

    return TimeIterationResult(
        mdr,
        it,
        with_complementarities,
        dprocess,
        err < tol,  # x_converged: bool
        tol,  # x_tol
        err,  #: float
        None,  # log: object # TimeIterationLog
        None,  # trace: object #{Nothing,IterationTrace}
    )
