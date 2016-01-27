import time
import warnings

import numpy as np
from numba import jit
from scipy.linalg import lstsq

from dolo.algos.dtcscc.simulations import simulate
from dolo.numeric.discretization import gauss_hermite_nodes
from dolo.numeric.interpolation.complete_poly import (
    _complete_poly_impl, _complete_poly_impl_vec, complete_polynomial,
    n_complete)


def gssa(model, maxit=100, tol=1e-8, initial_dr=None, verbose=False,
         n_sim=10000, deg=3, damp=0.1, seed=42):
    """
    Sketch of algorithm:

    0. Choose levels for the initial states and the simulation length (n_sim)
    1. Obtain an initial decision rule -- here using first order perturbation
    2. Draw a sequence of innovations epsilon
    3. Iterate on the following steps:
        - Use the epsilons, initial states, and proposed decision rule to
          simulate model forward. Will leave us with time series of states and
          controls
        - Evaluate expectations using quadrature
        - Use direct response to get alternative proposal for controls
        - Regress updated controls on the simulated states to get proposal
          coefficients. New coefficients are convex combination of previous
          coefficients and proposal coefficients. Weights controlled by damp,
          where damp is the weight on the old coefficients. This should be
          fairly low to increase chances of convergence.
        - Check difference between the simulated series of controls and the
          direct response version of controls

    """
    # verify input arguments
    if deg < 0 or deg > 5:
        raise ValueError("deg must be in [1, 5]")

    if damp < 0 or damp > 1:
        raise ValueError("damp must be in [0, 1]")

    t1 = time.time()

    # extract model functions and parameters
    g = model.__original_functions__['transition']
    g_gu = model.__original_gufunctions__['transition']
    h_gu = model.__original_gufunctions__['expectation']
    d_gu = model.__original_gufunctions__['direct_response']
    p = model.calibration['parameters']
    n_s = len(model.symbols["states"])
    n_x = len(model.symbols["controls"])
    n_z = len(model.symbols["expectations"])
    n_eps = len(model.symbols["shocks"])
    s0 = model.calibration["states"]
    x0 = model.calibration["controls"]

    # construct initial decision rule if not supplied
    if initial_dr is None:
        drp = approximate_controls(model)
    else:
        drp = initial_dr

    # set up quadrature weights and nodes
    sigma = model.covariances
    nodes, weights = gauss_hermite_nodes([5], model.covariances)

    # draw sequence of innovations
    np.random.seed(seed)
    epsilon = numpy.random.multivariate_normal(np.zeros(n_eps), sigma, n_sim)

    # simulate initial decision rule and do initial regression for coefs
    init_sim = simulate(model, drp, horizon=n_sim, return_array=True,
                        forcing_shocks=epsilon)
    s_sim = init_sim[:, 0, 0:n_s]
    x_sim = init_sim[:, 0, n_s:n_s + n_x]
    Phi_sim = complete_polynomial(s_sim.T, deg).T
    coefs = np.ascontiguousarray(lstsq(Phi_sim, x_sim)[0])

    # NOTE: the ascontiguousarray above was needed for numba to compile the
    #       `np.dot` in the simulation function in no python mode. Appearantly
    #       the array returned from lstsq is not C-contiguous

    # allocate for simulated series of expectations and next period states
    z_sim = np.empty((n_sim, n_z))
    S = np.empty_like(s_sim)
    X = np.empty_like(x_sim)
    H = np.empty_like(z_sim)
    new_x = np.empty_like(x_sim)

    # set initial states and controls
    s_sim[0, :] = s0
    x_sim[0, :] = x0

    Phi_t = np.empty(n_complete(n_s, deg))  # buffer array for simulation

    # create jitted function that will simulate states and controls, using
    # the epsilon shocks from above (define here as closure over all data
    # above).
    @jit(nopython=True)
    def simulate_states_controls(s, x, Phi_t, coefs):
        for t in range(1, n_sim):
            g(s[t - 1, :], x[t - 1, :], epsilon[t, :], p, s[t, :])

            # fill Phi_t with new complete poly version of s[t, :]
            _complete_poly_impl_vec(s[t, :], deg, Phi_t)

            # do inner product to get new controls
            x[t, :] = Phi_t @coefs

    it = 0
    err = 10.0
    err_0 = 10

    if verbose:
        headline = '|{0:^4} | {1:10} | {2:8} | {3:8} |'
        headline = headline.format('N', ' Error', 'Gain', 'Time')
        stars = '-' * len(headline)
        print(stars)
        print(headline)
        print(stars)

        # format string for within loop
        fmt_str = '|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} |'

    while err > tol and it <= maxit:
        t_start = time.time()

        # simulate with new coefficients
        simulate_states_controls(s_sim, x_sim, Phi_t, coefs)

        # update expectations of z
        # update_expectations(s_sim, x_sim, z_sim, Phi_sim)
        z_sim[:, :] = 0.0
        for i in range(weights.shape[0]):
            e = nodes[i, :]  # extract nodes
            # evaluate future states at each node (stores in S)
            g_gu(s_sim, x_sim, e, p, S)

            # evaluate future controls at each future state
            _complete_poly_impl(S.T, deg, Phi_sim.T)
            np.dot(Phi_sim, coefs, out=X)

            # compute expectation (stores in H)
            h_gu(S, X, p, H)
            z_sim += weights[i] * H

        # get controls on the simulated points from direct_resposne
        # (stores in new_x)
        d_gu(s_sim, z_sim, p, new_x)

        # update basis matrix and do regression of new_x on s_sim to get
        # updated coefficients
        _complete_poly_impl(s_sim.T, deg, Phi_sim.T)
        new_coefs = np.ascontiguousarray(lstsq(Phi_sim, new_x)[0])

        # check whether they differ from the preceding guess
        err = (abs(new_x - x_sim).max())

        # update the series of controls and coefficients
        x_sim[:, :] = new_x
        coefs = (1 - damp) * new_coefs + damp * coefs

        if verbose:
            # update error and print if `verbose`
            err_SA = err / err_0
            err_0 = err
            t_finish = time.time()
            elapsed = t_finish - t_start
            if verbose:
                print(fmt_str.format(it, err, err_SA, elapsed))

        it += 1

    if it == maxit:
        warnings.warn(UserWarning("Maximum number of iterations reached"))

    # compute final fime and do final printout if `verbose`
    t2 = time.time()
    if verbose:
        print(stars)
        print('Elapsed: {} seconds.'.format(t2 - t1))
        print(stars)

    return coefs


if __name__ == '__main__':

    from dolo import *
    from dolo.algos.dtcscc.accuracy import omega

    model = yaml_import("../../../examples/models/rbc_full.yaml")

    gssa(model, deg=5, verbose=True, damp=0.1)

    # TODO: time and check the returned coefficients
