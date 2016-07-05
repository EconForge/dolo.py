import time
import warnings
import numpy
from dolo.numeric.discretization import gauss_hermite_nodes
from dolo.numeric.interpolation.splines import MultivariateSplines
from dolo.numeric.interpolation import create_interpolator

def evaluate_policy(model, dr, tol=1e-8, grid={}, distribution={}, maxit=2000, verbose=False, hook=None,
                    integration_orders=None):
    """Compute value function corresponding to policy ``dr``

    Parameters:
    -----------

    model:
        "dtcscc" model. Must contain a 'value' function.

    dr:
        decision rule to evaluate

    Returns:
    --------

    decision rule:
        value function (a function of the space similar to a decision rule
        object)

    """

    assert (model.model_type == 'dtcscc')

    vfun = model.functions["value"]
    gfun = model.functions['transition']

    parms = model.calibration['parameters']

    n_vals = len(model.symbols['values'])

    t1 = time.time()
    err = 1.0

    it = 0

    approx = model.get_grid(**grid)
    interp_type = approx.interpolation

    distrib = model.get_distribution(**distribution)
    sigma = distrib.sigma

    drv = create_interpolator(approx, approx.interpolation)

    grid = drv.grid

    N = drv.grid.shape[0]

    controls = dr(grid)

    guess_0 = model.calibration['values']
    guess_0 = guess_0[None, :].repeat(N, axis=0)


    if not integration_orders:
        integration_orders = [3] * sigma.shape[0]
    [nodes, weights] = gauss_hermite_nodes(integration_orders, sigma)

    if verbose:
        headline = '|{0:^4} | {1:10} | {2:8} | {3:8} |'.format('N', ' Error', 'Gain', 'Time')
        stars = '-' * len(headline)
        print(stars)
        print(headline)
        print(stars)

    err_0 = 1.0

    while err > tol and it < maxit:

        if hook:
            hook()

        t_start = time.time()
        it += 1

        # update spline coefficients with current values
        drv.set_values(guess_0)

        # update the geuss of value functions
        guess = update_value(gfun, vfun, grid, controls, dr, drv,
                             nodes, weights, parms, n_vals)

        # compute error
        err = abs(guess - guess_0).max()
        err_SA = err / err_0
        err_0 = err

        # update guess
        guess_0[:] = guess.copy()

        # print update to user, if verbose
        t_end = time.time()
        elapsed = t_end - t_start
        if verbose:
            print('|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} |'.format(
                it, err, err_SA, elapsed))

    if it == maxit:
        warnings.warn(UserWarning("Maximum number of iterations reached"))

    t2 = time.time()
    if verbose:
        print(stars)
        print('Elapsed: {} seconds.'.format(t2 - t1))
        print(stars)

    return drv


def update_value(g, v, s, x, dr, drv, nodes, weights, parms, n_vals):

    N = s.shape[0]
    n_s = s.shape[1]
    n_x = x.shape[1]
    n_e = nodes.shape[1]

    res = numpy.zeros((N, n_vals))

    # Compute expectation over future value function
    for i in range(nodes.shape[0]):

        e = nodes[i, :]
        w = weights[i]

        S = g(s, x, e, parms)

        X = dr(S)
        V = drv(S)

        res += w * v(s, x, S, X, V, parms)

    return res








import time
import numpy as np
from dolo.algos.dtcscc.perturbations import approximate_controls
from dolo.numeric.interpolation import create_interpolator
from scipy.optimize import minimize


def solve_policy(model, tol=1e-8, grid={}, distribution={}, integration_orders=None, maxit=100, maxit_howard=20, verbose=False, hook=None, initial_dr=None, pert_order=1):
    """
    Solve for the value function and associated decision rule by iterating over
    the value function.

    Parameters:
    -----------
    model:
        "dtcscc" model. Must contain a 'felicity' function.

    Returns:
    --------
        dr : Markov decision rule
            The solved decision rule/policy function
        drv: decision rule
            The solved value function
    """

    assert (model.model_type == 'dtcscc')

    def vprint(t):
        if verbose:
            print(t)

    # transition(s, x, e, p, out), felicity(s, x, p, out)
    transition = model.functions['transition']
    felicity = model.functions['felicity']
    controls_lb = model.functions['controls_lb']
    controls_ub = model.functions['controls_ub']

    parms = model.calibration['parameters']
    discount = model.calibration['beta']

    x0 = model.calibration['controls']
    s0 = model.calibration['states']
    r0 = felicity(s0, x0, parms)

    approx = model.get_grid(**grid)
    # a = approx.a
    # b = approx.b
    # orders = approx.orders
    distrib = model.get_distribution(**distribution)
    sigma = distrib.sigma

    # Possibly use the approximation orders?
    if integration_orders is None:
        integration_orders = [3] * sigma.shape[0]
    [nodes, weights] = gauss_hermite_nodes(integration_orders, sigma)

    interp_type = approx.interpolation
    drv = create_interpolator(approx, interp_type)
    if initial_dr is None:
        if pert_order == 1:
            dr = approximate_controls(model)
        if pert_order > 1:
            raise Exception("Perturbation order > 1 not supported (yet).")
    else:
        dr = initial_dr

    grid = drv.grid
    N = grid.shape[0]       # Number of states
    n_x = grid.shape[1]     # Number of controls

    controls = dr(grid)
    controls_0 = np.zeros((N, n_x))
    controls_0[:, :] = model.calibration['controls'][None, :]

    values_0 = np.zeros((N, 1))
    values_0[:, :] = r0/(1-discount)

    if verbose:
        headline = '|{0:^4} | {1:10} | {2:8} | {3:8} |'.format('N', ' Error',
                                                               'Gain', 'Time')
        stars = '-' * len(headline)
        print(stars)
        print(headline)
        print(stars)

    t1 = time.time()

    # FIRST: value function iterations, 10 iterations to start
    it = 0
    err_v = 100
    err_v_0 = 0.0
    gain_v = 0.0
    err_x = 100
    err_x_0 = 100

    if verbose:
        print(stars)
        print('Starting value function iteration')
        print(stars)

    while err_v > tol and it < 10:

        t_start = time.time()
        it += 1

        # update interpolation object with current values
        drv.set_values(values_0)

        values = values_0.copy()
        controls = controls_0.copy()

        for n in range(N):
            s = grid[n, :]
            x = controls[n, :]
            lb = controls_lb(s, parms)
            ub = controls_ub(s, parms)
            bnds = [e for e in zip(lb, ub)]

            def valfun(xx):
                return -choice_value(transition, felicity, s, xx, drv, nodes,
                                     weights, parms, discount)[0]
            res = minimize(valfun, x, bounds=bnds, tol=1e-4)

            controls[n, :] = res.x
            values[n, 0] = -valfun(res.x)

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

        # print update to user, if verbose
        if verbose:
            print('|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} |'.format(
                it, err_v, gain_v, elapsed))

    # SECOND: Howard improvement step, 10-20 iterations
    it = 0
    err_v = 100
    err_v_0 = 0.0
    gain_v = 0.0

    if verbose:
        print(stars)
        print('Starting Howard improvement step')
        print(stars)

    while err_v > tol and it < maxit_howard:

        t_start = time.time()
        it += 1

        # update interpolation object with current values
        drv.set_values(values_0)

        values = values_0.copy()
        controls = controls_0.copy()
        # controls = controls_0.copy()  # No need to keep updating

        for n in range(N):
            s = grid[n, :]
            x = controls[n, :]
            values[n, 0] = choice_value(transition, felicity, s, x, drv, nodes,
                                        weights, parms, discount)[0]

        # compute error, update value function
        err_v = abs(values - values_0).max()
        values_0 = values

        t_end = time.time()
        elapsed = t_end - t_start

        gain_v = err_v / err_v_0
        err_v_0 = err_v

        # print update to user, if verbose
        if verbose:
            print('|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} |'.format(
                it, err_v, gain_v, elapsed))

    # THIRD: Back to value function iteration until convergence.
    it = 0
    err_v = 100
    err_v_0 = 0.0
    gain_v = 0.0
    err_x = 100
    err_x_0 = 100

    if verbose:
        print(stars)
        print('Starting value function iteration')
        print(stars)

    while err_v > tol and it < maxit:

        t_start = time.time()
        it += 1

        # update interpolation object with current values
        drv.set_values(values_0)

        values = values_0.copy()
        controls = controls_0.copy()

        for n in range(N):
            s = grid[n, :]
            x = controls[n, :]
            lb = controls_lb(s, parms)
            ub = controls_ub(s, parms)
            bnds = [e for e in zip(lb, ub)]

            def valfun(xx):
                return -choice_value(transition, felicity, s, xx, drv, nodes,
                                     weights, parms, discount)[0]
            res = minimize(valfun, x, bounds=bnds, tol=1e-4)

            controls[n, :] = res.x
            values[n, 0] = -valfun(res.x)

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

        # print update to user, if verbose
        if verbose:
            print('|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} |'.format(
                it, err_v, gain_v, elapsed))

    if it == maxit:
        warnings.warn(UserWarning("Maximum number of iterations reached"))

    t2 = time.time()
    if verbose:
        print(stars)
        print('Elapsed: {} seconds.'.format(t2 - t1))
        print(stars)

    # final value function and decision rule
    drv.set_values(values_0)
    dr = create_interpolator(approx, interp_type)
    dr.set_values(controls_0)

    return dr, drv


def choice_value(transition, felicity, s, x, drv, nodes, weights, parms, beta):
    cont_v = 0.0
    for i in range(nodes.shape[0]):
        e = nodes[i, :]
        w = weights[i]
        S = transition(s, x, e, parms)
        # X = dr(S)
        V = drv(S)[0]
        cont_v += w*V
    return felicity(s, x, parms) + beta*cont_v
