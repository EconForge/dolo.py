import time
import numpy as np
import scipy.optimize
from collections import OrderedDict
from dolo.numeric.processes import DiscretizedIIDProcess
from dolo.numeric.decision_rules_markov import MarkovDecisionRule, IIDDecisionRule

from dolo.misc.itprinter import IterationsPrinter


import numpy
from dolo.numeric.decision_rules_markov import MarkovDecisionRule

def solve_policy(model, grid={}, tol=1e-6, maxit=500,
                 maxit_howard=20, verbose=False):
    """
    Solve for the value function and associated Markov decision rule by iterating over
    the value function.

    Parameters:
    -----------
    model :
        "dtmscc" model. Must contain a 'felicity' function.
    grid :
        grid options
    dr :
        decision rule to evaluate

    Returns:
    --------
    mdr : Markov decision rule
        The solved decision rule/policy function
    mdrv: decision rule
        The solved value function
    """

    transition = model.functions['transition']
    felicity = model.functions['felicity']
    controls_lb = model.functions['controls_lb']
    controls_ub = model.functions['controls_ub']

    parms = model.calibration['parameters']
    discount = model.calibration['beta']

    x0 = model.calibration['controls']
    m0 = model.calibration['exogenous']
    s0 = model.calibration['states']
    r0 = felicity(m0, s0, x0, parms)

    process = model.exogenous
    dprocess = process.discretize()

    n_ms = dprocess.n_nodes() # number of exogenous states
    n_mv = dprocess.n_inodes(0) # this assume number of integration nodes is constant

    approx = model.get_grid(**grid)
    a = approx.a
    b = approx.b
    orders = approx.orders


    if isinstance(dprocess, DiscretizedIIDProcess):
        mdrv = IIDDecisionRule(n_ms, a, b, orders)
    else:
        mdrv = MarkovDecisionRule(n_ms, a, b, orders)

    grid = mdrv.grid
    N = grid.shape[0]
    n_x = len(x0)

    controls_0 = np.zeros((n_ms, N, n_x))
    controls_0[:, :, :] = model.calibration['controls'][None, None, :]
    #
    values_0 = np.zeros((n_ms, N, 1))
    values_0[:, :, :] = r0/(1-discount)

    itprint = IterationsPrinter(('N', int), ('Error', float), ('Gain', float),
                                ('Time', float), verbose=verbose)
    itprint.print_header('Evaluating value of initial guess')

    # FIRST: value function iterations, 10 iterations to start
    it = 0
    err_v = 100
    err_v_0 = 0.0
    gain_v = 0.0
    err_x = 100
    err_x_0 = 100

    if verbose:
        print('-----')
        print('Starting value function iteration')
        print('-----')

    while it < 10 and err_v > tol:

        t_start = time.time()
        it += 1

        # update interpolation object with current values
        mdrv.set_values(values_0)

        values = values_0.copy()
        controls = controls_0.copy()

        for i_m in range(dprocess.n_nodes()):
            m = dprocess.node(i_m)
            for n in range(N):
                s = grid[n, :]
                x = controls[i_m, n, :]
                lb = controls_lb(m, s, parms)
                ub = controls_ub(m, s, parms)
                bnds = [e for e in zip(lb, ub)]

                def valfun(xx):
                    return -choice_value(transition, felicity, i_m, s, xx,
                                         mdrv, dprocess, parms, discount)[0]
                res = scipy.optimize.minimize(valfun, x, bounds=bnds)

                controls[i_m, n, :] = res.x
                values[i_m, n, 0] = -valfun(res.x)

        # compute error, update value and dr
        err_x = abs(controls - controls_0).max()
        err_v = abs(values - values_0).max()
        t_end = time.time()
        elapsed = t_end-t_start

        values_0 = values
        controls_0 = controls

        gain_x = err_x / err_x_0
        gain_v = err_v / err_v_0

        err_x_0 = err_x
        err_v_0 = err_v

        itprint.print_iteration(N=it, Error=err_v, Gain=gain_v,
                                Time=elapsed)

    # SECOND: Howard improvement step, 10-20 iterations
    it = 0
    err_v = 100
    err_v_0 = 0.0
    gain_v = 1.0

    if verbose:
        print('-----')
        print('Starting Howard improvement step')
        print('-----')

    while it < maxit_howard and err_v > tol:

        t_start = time.time()
        it += 1

        # update interpolation object with current values
        mdrv.set_values(values_0)
        values = values_0.copy()

        for i_m in range(n_ms):
            for n in range(N):
                m = dprocess.node(i_m)
                s = grid[n, :]
                x = controls_0[i_m, n, :]
                values[i_m, n, 0] = choice_value(transition, felicity, i_m, s, x, mdrv, dprocess, parms, discount)

        # compute error, update value function
        err_v = abs(values - values_0).max()
        values_0 = values

        t_end = time.time()
        elapsed = t_end-t_start

        gain_v = err_v / err_v_0
        err_v_0 = err_v
        itprint.print_iteration(N=it, Error=err_v, Gain=gain_v, Time=elapsed)
        # vprint(fmt_str.format(it, err_v, gain_v, elapsed))

    # THIRD: value function iterations until convergence
    it = 0
    err_v = 100
    err_v_0 = 0.0
    gain_v = 0.0
    err_x = 100
    err_x_0 = 100

    # if verbose:
    #     print('-----')
    #     print('Starting value function iteration')
    #     print('-----')

    itprint = IterationsPrinter(('N', int), ('Error_V', float), ('Gain_V', float),
                                ('Error_x', float), ('Gain_x', float), ('Time', float), verbose=verbose)
    itprint.print_header('Start value function iterations.')


    while it < maxit and err_v > tol:

        t_start = time.time()
        it += 1

        # update interpolation object with current values
        mdrv.set_values(values_0)

        values = values_0.copy()
        controls = controls_0.copy()

        for i_m in range(n_ms):
            for n in range(N):
                m = dprocess.node(i_m)
                s = grid[n, :]
                x = controls[i_m, n, :]
                lb = controls_lb(m, s, parms)
                ub = controls_ub(m, s, parms)
                bnds = [e for e in zip(lb, ub)]

                def valfun(xx):
                    return -choice_value(transition, felicity, i_m, s, xx, mdrv, dprocess, parms, discount)[0]
                res = scipy.optimize.minimize(valfun, x, bounds=bnds)

                controls[i_m, n, :] = res.x
                values[i_m, n, 0] = -valfun(res.x)

        # compute error, update value and dr
        err_x = abs(controls - controls_0).max()
        err_v = abs(values - values_0).max()
        t_end = time.time()
        elapsed = t_end-t_start

        values_0 = values
        controls_0 = controls

        gain_x = err_x / err_x_0
        gain_v = err_v / err_v_0

        err_x_0 = err_x
        err_v_0 = err_v

        itprint.print_iteration(N=it,
                                Error_V=err_v,
                                Gain_V=gain_v,
                                Error_x=err_x,
                                Gain_x=gain_x,
                                Time=elapsed)

    itprint.print_finished()


    if isinstance(dprocess, DiscretizedIIDProcess):
        mdr = IIDDecisionRule(n_ms, a, b, orders)
    else:
        mdr = MarkovDecisionRule(n_ms, a, b, orders)

    mdr.set_values(controls)
    mdrv.set_values(values_0)


    return mdr, mdrv


def choice_value(transition, felicity, i_ms, s, x, drv, dprocess, parms, beta):

    m = dprocess.node(i_ms)
    cont_v = 0.0
    for I_ms in range(dprocess.n_inodes(i_ms)):
        M = dprocess.inode(i_ms,I_ms)
        prob = dprocess.iweight(i_ms, I_ms)
        S = transition(m, s, x, M, parms)
        V = drv(I_ms, S)[0]
        cont_v += prob*V
    return felicity(m, s, x, parms) + beta*cont_v



def evaluate_policy(model, mdr, tol=1e-8,  maxit=2000, grid={}, verbose=True, initial_guess=None, hook=None, integration_orders=None):

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

    assert(model.is_dtmscc())

    [P, Q] = model.exogenous

    n_ms = P.shape[0]   # number of markov states
    n_mv = P.shape[1]   # number of markov variables

    x0 = model.calibration['controls']
    v0 = model.calibration['values']
    parms = model.calibration['parameters']
    n_x = len(x0)
    n_v = len(v0)
    n_s = len(model.symbols['states'])

    approx = model.get_grid(**grid)
    a = approx.a
    b = approx.b
    orders = approx.orders

    from dolo.numeric.decision_rules_markov import MarkovDecisionRule
    mdrv = MarkovDecisionRule(n_ms, a, b, orders) # values

    grid = mdrv.grid
    N = grid.shape[0]

    controls = np.zeros((n_ms, N, n_x))
    for i_m in range(n_ms):
        controls[i_m, :, :] = mdr(i_m, grid) #x0[None,:]

    values_0 = np.zeros((n_ms, N, n_v))
    if initial_guess is None:
        for i_m in range(n_ms):
            values_0[i_m, :, :] = v0[None, :]
    else:
        for i_m in range(n_ms):
            values_0[i_m, :, :] = initial_guess(i_m, grid)

    val = model.functions['value']
    g = model.functions['transition']

    sh_v = values_0.shape

    err = 10
    inner_maxit = 50
    it = 0

    if verbose:
        headline = '|{0:^4} | {1:10} | {2:8} | {3:8} |'.format( 'N',' Error', 'Gain','Time')
        stars = '-'*len(headline)
        print(stars)
        print(headline)
        print(stars)

    t1 = time.time()

    err_0 = np.nan

    verbit = (verbose == 'full')

    while err>tol and it<maxit:

        it += 1

        t_start = time.time()

        mdrv.set_values(values_0.reshape(sh_v))

        values = update_value(val, g, grid, controls, values_0, mdr, mdrv, P, Q, parms).reshape((-1,n_v))

        err = abs(values.reshape(sh_v)-values_0).max()

        err_SA = err/err_0
        err_0 = err

        values_0 = values.reshape(sh_v)

        t_finish = time.time()
        elapsed = t_finish - t_start

        if verbose:
            print('|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} |'.format( it, err, err_SA, elapsed  ))

    # values_0 = values.reshape(sh_v)

    t2 = time.time()

    if verbose:
        print(stars)
        print("Elapsed: {} seconds.".format(t2-t1))
        print(stars)

    return mdrv


def update_value(val, g, s, x, v, dr, drv, P, Q, parms):

    N = s.shape[0]
    n_s = s.shape[1]

    n_ms = P.shape[0]   # number of markov states

    res = np.zeros_like(v)

    for i_ms in range(n_ms):

        m = P[i_ms, :][None, :].repeat(N, axis=0)

        xm = x[i_ms, :, :]
        vm = v[i_ms, :, :]

        for I_ms in range(n_ms):

            # M = P[I_ms,:][None,:]
            M = P[I_ms, :][None, :].repeat(N, axis=0)
            prob = Q[i_ms, I_ms]

            S = g(m, s, xm, M, parms)
            XM = dr(I_ms, S)
            VM = drv(I_ms, S)

            rr = val(m, s, xm, vm, M, S, XM, VM, parms)

            res[i_ms, :, :] += prob*rr

    return res
