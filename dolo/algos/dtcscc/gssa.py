import numpy as np
from numba import jit
from scipy.linalg import lstsq

from dolo.algos.dtcscc.simulations import simulate
from dolo.numeric.discretization import gauss_hermite_nodes
from dolo.numeric.interpolation.complete_poly import (complete_polynomial,
                                                      n_complete,
                                                      _complete_poly_impl,
                                                      _complete_poly_impl_vec)


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

    """
    # verify input arguments
    if deg < 0 or deg > 5:
        raise ValueError("deg must be in [1, 5]")

    if damp < 0 or damp > 1:
        raise ValueError("damp must be in [0, 1]")

    # extract model functions and parameters
    g = model.__original_functions__['transition']
    d = model.functions['direct_response']
    # h = model.__original_functions__['expectation']  # TODO: not working
    h = model.functions['expectation']
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
    x_sim = init_sim[:, 0, n_s:n_s+n_x]
    Phi_sim = complete_polynomial(s_sim.T, deg).T
    coefs = np.ascontiguousarray(lstsq(Phi_sim, x_sim)[0])

    # NOTE: the ascontiguousarray above was needed for numba to compile the
    #       `np.dot` in the simulation function in no python mode. Appearantly
    #       the array returned from lstsq is not C-contiguous

    # allocate for simulated series of expectations
    z_sim = np.empty((n_sim, n_z))

    # set initial states and controls
    s_sim[0, :] = s0
    x_sim[0, :] = x0

    Phi_t = np.empty(n_complete(n_s, deg))  # buffer array for simulation

    # create jitted function that will simulate states and controls, using
    # the epsilon shocks from above (define here as closure over all data
    # above).
    # @jit(nopython=True)
    def simulate_states_controls(s, x, Phi_t, coefs):
        for t in range(1, n_sim):
            s[t, :] = g(s[t-1, :], x[t-1, :], epsilon[t, :], p)

            # fill Phi_t with new complete poly version of s[t, :]
            _complete_poly_impl_vec(s[t, :], deg, Phi_t)

            # do inner product to get new controls
            x[t, :] = Phi_t @ coefs

    # @jit(nopython=True)
    def update_expectations(s, x, z, Phi):
        # evaluates expectations on simulated points using quadrature
        z[:, :] = 0.0
        for i in range(weights.shape[0]):
            e = nodes[i, :]  # extract nodes
            S = g(s, x, e, p)  # evaluate future states at each node

            # evaluate future controls at each future state
            _complete_poly_impl(S.T, deg, Phi.T)
            X = Phi @ coefs

            # compute expectation
            z += weights[i] * h(S, X, p)

    it = 0
    err = 10.0

    while err > tol and it <= maxit:
        # simulate with new coefficients
        simulate_states_controls(s_sim, x_sim, Phi_t, coefs)

        # update expectations of z
        update_expectations(s_sim, x_sim, z_sim, Phi_sim)

        # get controls on the simulated points from direct_resposne
        new_x = d(s_sim, z_sim, p)

        # update basis matrix and do regression of new_x on s_sim to get
        # updated coefficients
        _complete_poly_impl(s_sim.T, deg, Phi_sim.T)
        new_coefs = np.ascontiguousarray(lstsq(Phi_sim, new_x)[0])

        # check whether they differ from the preceding guess
        err = (abs(new_x - x_sim).max())

        # update the series of controls and coefficients
        x_sim[:, :] = new_x
        coefs = (1-damp)*new_coefs + damp*coefs

        if verbose:
            print(err)

    return coefs


if __name__ == '__main__':

    from dolo import *
    from dolo.algos.dtcscc.accuracy import omega

    model = yaml_import(
        "/Users/sglyon/src/Python/dolo/examples/models/rbc_full.yaml")

    gssa(model, deg=5, verbose=True, damp=0.5)

    # TODO: time and check the returned coefficients
