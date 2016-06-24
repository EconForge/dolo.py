import time
import warnings

import numpy

from dolo.numeric.discretization import gauss_hermite_nodes
from dolo.numeric.interpolation.splines import MultivariateSplines
from dolo.numeric.interpolation import create_interpolator


def evaluate_policy(model, dr, tol=1e-8, grid={}, distribution={}, maxit=2000, verbose=False, hook=None,
                    integration_orders=None):
    """Compute value function corresponding to a given policy ``dr``.

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
    [epsilons, weights] = gauss_hermite_nodes(integration_orders, sigma)

    if verbose:
        headline = '|{0:^4} | {1:10} | {2:8} | {3:8} |'.format('N', ' Error',
                                                               'Gain', 'Time')
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
                             epsilons, weights, parms, n_vals)

        # compute error
        err = abs(guess - guess_0).max()
        err_SA = err / err_0
        err_0 = err

        # update guess
        guess_0[:] = guess.copy()

        # print update to user, if verbose
        t_finish = time.time()
        elapsed = t_finish - t_start
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


def update_value(g, v, s, x, dr, drv, epsilons, weights, parms, n_vals):

    N = s.shape[0]
    n_s = s.shape[1]
    n_x = x.shape[1]
    n_e = epsilons.shape[1]

    res = numpy.zeros((N, n_vals))

    for i in range(epsilons.shape[0]):

        e = epsilons[i, :]
        w = weights[i]

        S = g(s, x, e, parms)

        X = dr(S)
        V = drv(S)

        res += w * v(s, x, S, X, V, parms)

    return res
