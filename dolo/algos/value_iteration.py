import time
import numpy as np
import numpy
import scipy.optimize
from dolo.numeric.processes import DiscretizedIIDProcess

# from dolo.numeric.decision_rules_markov import MarkovDecisionRule, IIDDecisionRule
from dolo.numeric.decision_rule import DecisionRule, ConstantDecisionRule
from dolo.numeric.grids import Grid, CartesianGrid, SmolyakGrid, UnstructuredGrid
from dolo.misc.itprinter import IterationsPrinter


def constant_policy(model):
    return ConstantDecisionRule(model.calibration["controls"])


from .results import AlgoResult, ValueIterationResult


def value_iteration(
    model, tol=1e-6, maxit=500, maxit_howard=20, verbose=False, details=True
):
    """
    Solve for the value function and associated Markov decision rule by iterating over
    the value function.

    Parameters:
    -----------
    model :
        "dtmscc" model. Must contain a 'felicity' function.
    dr :
        decision rule to evaluate

    Returns:
    --------
    mdr : Markov decision rule
        The solved decision rule/policy function
    mdrv: decision rule
        The solved value function
    """

    transition = model.functions["transition"]
    felicity = model.functions["felicity"]
    controls_lb = model.functions["controls_lb"]
    controls_ub = model.functions["controls_ub"]

    parms = model.calibration["parameters"]
    discount = model.calibration["beta"]

    x0 = model.calibration["controls"]
    m0 = model.calibration["exogenous"]
    s0 = model.calibration["states"]
    r0 = felicity(m0, s0, x0, parms)

    process = model.exogenous

    grid, dprocess = model.discretize()
    endo_grid = grid["endo"]
    exo_grid = grid["exo"]

    n_ms = dprocess.n_nodes  # number of exogenous states
    n_mv = dprocess.n_inodes(0)  # this assume number of integration nodes is constant

    mdrv = DecisionRule(exo_grid, endo_grid)

    s = mdrv.endo_grid.nodes
    N = s.shape[0]
    n_x = len(x0)

    mdr = constant_policy(model)

    controls_0 = np.zeros((n_ms, N, n_x))
    for i_ms in range(n_ms):
        controls_0[i_ms, :, :] = mdr.eval_is(i_ms, s)

    values_0 = np.zeros((n_ms, N, 1))
    # for i_ms in range(n_ms):
    #     values_0[i_ms, :, :] = mdrv(i_ms, grid)

    mdr = DecisionRule(exo_grid, endo_grid)
    # mdr.set_values(controls_0)

    # THIRD: value function iterations until convergence
    it = 0
    err_v = 100
    err_v_0 = 0
    gain_v = 0.0
    err_x = 100
    err_x_0 = 0
    tol_x = 1e-5
    tol_v = 1e-7

    itprint = IterationsPrinter(
        ("N", int),
        ("Error_V", float),
        ("Gain_V", float),
        ("Error_x", float),
        ("Gain_x", float),
        ("Eval_n", int),
        ("Time", float),
        verbose=verbose,
    )
    itprint.print_header("Start value function iterations.")

    while (it < maxit) and (err_v > tol or err_x > tol_x):

        t_start = time.time()
        it += 1

        mdr.set_values(controls_0)
        if it > 2:
            ev = evaluate_policy(model, mdr, dr0=mdrv, verbose=False, details=True)
        else:
            ev = evaluate_policy(model, mdr, verbose=False, details=True)

        mdrv = ev.solution
        for i_ms in range(n_ms):
            values_0[i_ms, :, :] = mdrv.eval_is(i_ms, s)

        values = values_0.copy()
        controls = controls_0.copy()

        for i_m in range(n_ms):
            m = dprocess.node(i_m)
            for n in range(N):
                s_ = s[n, :]
                x = controls[i_m, n, :]
                lb = controls_lb(m, s_, parms)
                ub = controls_ub(m, s_, parms)
                bnds = [e for e in zip(lb, ub)]

                def valfun(xx):
                    return -choice_value(
                        transition,
                        felicity,
                        i_m,
                        s_,
                        xx,
                        mdrv,
                        dprocess,
                        parms,
                        discount,
                    )[0]

                res = scipy.optimize.minimize(valfun, x, bounds=bnds)
                controls[i_m, n, :] = res.x
                values[i_m, n, 0] = -valfun(x)

        # compute error, update value and dr
        err_x = abs(controls - controls_0).max()
        err_v = abs(values - values_0).max()
        t_end = time.time()
        elapsed = t_end - t_start

        values_0 = values
        controls_0 = controls

        gain_x = err_x / err_x_0
        gain_v = err_v / err_v_0

        err_x_0 = err_x
        err_v_0 = err_v

        itprint.print_iteration(
            N=it,
            Error_V=err_v,
            Gain_V=gain_v,
            Error_x=err_x,
            Gain_x=gain_x,
            Eval_n=ev.iterations,
            Time=elapsed,
        )

    itprint.print_finished()

    mdr = DecisionRule(exo_grid, endo_grid)

    mdr.set_values(controls)
    mdrv.set_values(values_0)

    if not details:
        return mdr, mdrv
    else:
        return ValueIterationResult(
            mdr,  #:AbstractDecisionRule
            mdrv,  #:AbstractDecisionRule
            it,  #:Int
            dprocess,  #:AbstractDiscretizedProcess
            err_x < tol_x,  #:Bool
            tol_x,  #:Float64
            err_x,  #:Float64
            err_v < tol_v,  #:Bool
            tol_v,  #:Float64
            err_v,  #:Float64
            None,  # log:     #:ValueIterationLog
            None,  # trace:   #:Union{Nothing,IterationTrace
        )


def choice_value(transition, felicity, i_ms, s, x, drv, dprocess, parms, beta):

    m = dprocess.node(i_ms)
    cont_v = 0.0
    for I_ms in range(dprocess.n_inodes(i_ms)):
        M = dprocess.inode(i_ms, I_ms)
        prob = dprocess.iweight(i_ms, I_ms)
        S = transition(m, s, x, M, parms)
        V = drv(I_ms, S)[0]
        cont_v += prob * V
    return felicity(m, s, x, parms) + beta * cont_v


class EvaluationResult:
    def __init__(self, solution, iterations, tol, error):
        self.solution = solution
        self.iterations = iterations
        self.tol = tol
        self.error = error


def evaluate_policy(
    model,
    mdr,
    tol=1e-8,
    maxit=2000,
    grid={},
    verbose=True,
    dr0=None,
    hook=None,
    integration_orders=None,
    details=False,
    interp_method="cubic",
):
    """Compute value function corresponding to policy ``dr``

    Parameters:
    -----------

    model:
        "dtcscc" model. Must contain a 'value' function.

    mdr:
        decision rule to evaluate

    Returns:
    --------

    decision rule:
        value function (a function of the space similar to a decision rule
        object)

    """

    process = model.exogenous
    grid, dprocess = model.discretize()
    endo_grid = grid["endo"]
    exo_grid = grid["exo"]

    n_ms = dprocess.n_nodes  # number of exogenous states
    n_mv = dprocess.n_inodes(0)  # this assume number of integration nodes is constant

    x0 = model.calibration["controls"]
    v0 = model.calibration["values"]
    parms = model.calibration["parameters"]
    n_x = len(x0)
    n_v = len(v0)
    n_s = len(model.symbols["states"])

    if dr0 is not None:
        mdrv = dr0
    else:
        mdrv = DecisionRule(exo_grid, endo_grid, interp_method=interp_method)

    s = mdrv.endo_grid.nodes
    N = s.shape[0]

    if isinstance(mdr, np.ndarray):
        controls = mdr
    else:
        controls = np.zeros((n_ms, N, n_x))
        for i_m in range(n_ms):
            controls[i_m, :, :] = mdr.eval_is(i_m, s)

    values_0 = np.zeros((n_ms, N, n_v))
    if dr0 is None:
        for i_m in range(n_ms):
            values_0[i_m, :, :] = v0[None, :]
    else:
        for i_m in range(n_ms):
            values_0[i_m, :, :] = dr0.eval_is(i_m, s)

    val = model.functions["value"]
    g = model.functions["transition"]

    sh_v = values_0.shape

    err = 10
    inner_maxit = 50
    it = 0

    if verbose:
        headline = "|{0:^4} | {1:10} | {2:8} | {3:8} |".format(
            "N", " Error", "Gain", "Time"
        )
        stars = "-" * len(headline)
        print(stars)
        print(headline)
        print(stars)

    t1 = time.time()

    err_0 = np.nan

    verbit = verbose == "full"

    while err > tol and it < maxit:

        it += 1

        t_start = time.time()

        mdrv.set_values(values_0.reshape(sh_v))
        values = update_value(
            val, g, s, controls, values_0, mdr, mdrv, dprocess, parms
        ).reshape((-1, n_v))
        err = abs(values.reshape(sh_v) - values_0).max()

        err_SA = err / err_0
        err_0 = err

        values_0 = values.reshape(sh_v)

        t_finish = time.time()
        elapsed = t_finish - t_start

        if verbose:
            print(
                "|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} |".format(
                    it, err, err_SA, elapsed
                )
            )

    # values_0 = values.reshape(sh_v)

    t2 = time.time()

    if verbose:
        print(stars)
        print("Elapsed: {} seconds.".format(t2 - t1))
        print(stars)

    if not details:
        return mdrv
    else:
        return EvaluationResult(mdrv, it, tol, err)


def update_value(val, g, s, x, v, dr, drv, dprocess, parms):

    N = s.shape[0]
    n_s = s.shape[1]

    n_ms = dprocess.n_nodes  # number of exogenous states
    n_mv = dprocess.n_inodes(0)  # this assume number of integration nodes is constant

    res = np.zeros_like(v)

    for i_ms in range(n_ms):

        m = dprocess.node(i_ms)[None, :].repeat(N, axis=0)

        xm = x[i_ms, :, :]
        vm = v[i_ms, :, :]

        for I_ms in range(n_mv):

            # M = P[I_ms,:][None,:]
            M = dprocess.inode(i_ms, I_ms)[None, :].repeat(N, axis=0)
            prob = dprocess.iweight(i_ms, I_ms)

            S = g(m, s, xm, M, parms)
            XM = dr.eval_ijs(i_ms, I_ms, S)
            VM = drv.eval_ijs(i_ms, I_ms, S)
            rr = val(m, s, xm, vm, M, S, XM, VM, parms)

            res[i_ms, :, :] += prob * rr

    return res
