import numpy as np
import pandas as pd
from numpy import array, atleast_2d, linspace, zeros
from scipy.optimize import root

from dolo.algos.dtcscc.steady_state import find_deterministic_equilibrium
from dolo.numeric.optimize.ncpsolve import ncpsolve
from dolo.numeric.optimize.newton import newton
from dolo.numeric.serial_operations import serial_multiplication as smult

def _shocks_to_epsilons(model, shocks, T):
    """
    Helper function to support input argument `shocks` being one of many
    different data types. Will always return a `T, n_e` matrix.
    """
    n_e = len(model.calibration['shocks'])

    # if we have a DataFrame, convert it to a dict and rely on the method below
    if isinstance(shocks, pd.DataFrame):
        shocks = {k: shocks[k].tolist() for k in shocks.columns}

    # handle case where shocks might be a dict. Be careful to handle case where
    # value arrays are not the same length
    if isinstance(shocks, dict):
        epsilons = np.zeros((T+1, n_e))
        for (i, k) in enumerate(model.symbols["shocks"]):
            if k in shocks:
                this_shock = shocks[k]
                epsilons[:len(this_shock)-1, i] = this_shock[1:]
                epsilons[(len(this_shock)-1):, i] = this_shock[-1]
            else:
                # otherwise set to value in calibration
                epsilons[:, i] = model.calibration["shocks"][i]

        return epsilons

    # read from calibration if not given
    if shocks is None:
        shocks = model.calibration["shocks"]

    # now we just assume that shocks is array-like and try using the output of
    # np.asarray(shocks)
    shocks = np.asarray(shocks)
    shocks = shocks.reshape((-1, n_e))

    # until last period, exogenous shock takes its last value
    epsilons = np.zeros((T+1, n_e))
    epsilons[:(shocks.shape[0]-1), :] = shocks[1:, :]
    epsilons[(shocks.shape[0]-1):, :] = shocks[-1:, :]

    return epsilons


def deterministic_solve(model, shocks=None, start_states=None, T=100,
                        ignore_constraints=False, maxit=100,
                        initial_guess=None, verbose=False, tol=1e-6):
    """
    Computes a perfect foresight simulation using a stacked-time algorithm.

    The initial state is specified either by providing a series of exogenous
    shocks and assuming the model is initially in equilibrium with the first
    value of the shock, or by specifying an initial value for the states.

    Parameters
    ----------
    model : NumericModel
        "fg" or "fga" model to be solved
    shocks : array-like, dict, or pandas.DataFrame
        A specification of the shocks to the model. Can be any of the
        following (note by "declaration order" below we mean the order
        of `model.symbols["shocks"]`):

        - A 1d numpy array-like specifying a time series for a single
          shock, or all shocks stacked into a single array.
        - A 2d numpy array where each column specifies the time series
          for one of the shocks in declaration order. This must be an
          `N` by number of shocks 2d array.
        - A dict where keys are strings found in
          `model.symbols["shocks"]` and values are a time series of
          values for that shock. For model shocks that do not appear in
          this dict, the shock is set to the calibrated value. Note
          that this interface is the most flexible as it allows the user
          to pass values for only a subset of the model shocks and it
          allows the passed time series to be of different lengths.
        - A DataFrame where columns map shock names into time series.
          The same assumptions and behavior that are used in the dict
          case apply here

        If nothing is given here, `shocks` is set equal to the
        calibrated values found in `model.calibration["shocks"]` for
        all periods.

        If the length of any time-series in shocks is less than `T`
        (see below) it is assumed that that particular shock will
        remain at the final given value for the duration of the
        simulaiton.
    start_states : ndarray or dict
        a vector with the value of initial states, or a calibration
        dictionary with the initial values of states and controls
    T : int
        horizon for the perfect foresight simulation
    maxit : int
        maximum number of iteration for the nonlinear solver
    verbose : boolean
        if True, the solver displays iterations
    tol : float
        stopping criterium for the nonlinear solver
    ignore_constraints : bool
        if True, complementarity constraints are ignored.

    Returns
    -------
    pandas dataframe
        a dataframe with T+1 observations of the model variables along the
        simulation (states, controls, auxiliaries). The first observation is
        the steady-state corresponding to the first value of the shocks. The
        simulation should return to a steady-state corresponding to the last
        value of the exogenous shocks.

    """

    # TODO:

    # if model.model_spec == 'fga':
    #     from dolo.compiler.converter import GModel_fg_from_fga
    #     model = GModel_fg_from_fga(model)

    # definitions
    n_s = len(model.calibration['states'])
    n_x = len(model.calibration['controls'])

    epsilons = _shocks_to_epsilons(model, shocks, T)

    # final initial and final steady-states consistent with exogenous shocks
    if start_states is None:
        start_states = model.calibration

    if isinstance(start_states, dict):
        # at least that part is clear
        start_equilibrium = start_states
        start_s = start_equilibrium['states']
        start_x = start_equilibrium['controls']
        final_s = start_equilibrium['states']
        final_x = start_equilibrium['controls']
    elif isinstance(start_states, np.ndarray):
        start_s = start_states
        start_x = model.calibration['controls']
        final_s = model.calibration['states']
        final_x = model.calibration['controls']

    # if start_constraints:
    #     # we ignore start_constraints
    #     start_dict.update(start_constraints)
    #     final_equilibrium = start_constraints.copy()
    # else:
    #     final_eqm = find_deterministic_equilibrium(model,
    #                                                constraints=final_dict)
    # final_s = final_eqm['states']
    # final_x = final_eqm['controls']
    #
    # start_s = start_states
    # start_x = final_x

    # TODO: for start_x, it should be possible to use first order guess

    final = np.concatenate([final_s, final_x])
    start = np.concatenate([start_s, start_x])

    if verbose is True:
        print("Initial states : {}".format(start_s))
        print("Final controls : {}".format(final_x))

    p = model.calibration['parameters']

    if initial_guess is None:
        initial_guess = np.row_stack([start*(1-l) + final*l
                                         for l in linspace(0.0, 1.0, T+1)])

    else:
        if isinstance(initial_guess, pd.DataFrame):
            initial_guess = np.array(initial_guess).T.copy()
        initial_guess = initial_guess[:, :n_s+n_x]
        initial_guess[0, :n_s] = start_s
        initial_guess[-1, n_s:] = final_x

    sh = initial_guess.shape

    if model.x_bounds and not ignore_constraints:
        initial_states = initial_guess[:, :n_s]
        [lb, ub] = [u(initial_states, p) for u in model.x_bounds]
        lower_bound = initial_guess*0 - np.inf
        lower_bound[:, n_s:] = lb
        upper_bound = initial_guess*0 + np.inf
        upper_bound[:, n_s:] = ub
        test1 = max(lb.max(axis=0) - lb.min(axis=0))
        test2 = max(ub.max(axis=0) - ub.min(axis=0))
        if test1 > 0.00000001 or test2 > 0.00000001:
            msg = "Not implemented: perfect foresight solution requires that "
            msg += "controls have constant bounds."
            raise Exception(msg)
    else:
        ignore_constraints = True
        lower_bound = None
        upper_bound = None

    nn = sh[0]*sh[1]

    def fobj(vec):
        o = det_residual(model, vec.reshape(sh), start_s, final_x, epsilons)[0]
        return o.ravel()

    if not ignore_constraints:
        def ff(vec):
            return det_residual(model, vec.reshape(sh), start_s, final_x,
                                epsilons, jactype='sparse')

        x0 = initial_guess.ravel()
        sol, nit = ncpsolve(ff, lower_bound.ravel(), upper_bound.ravel(),
                            initial_guess.ravel(), verbose=verbose,
                            maxit=maxit, tol=tol, jactype='sparse')

        sol = sol.reshape(sh)

    else:

        def ff(vec):
            return det_residual(model, vec.reshape(sh), start_s, final_x,
                                epsilons, diff=False).ravel()

        x0 = initial_guess.ravel()
        sol = root(ff, x0, jac=False)
        res = ff(sol.x)
        sol = sol.x.reshape(sh)

    if 'auxiliary' in model.functions:
        colnames = (model.symbols['states'] + model.symbols['controls'] +
                    model.symbols['auxiliaries'])
        # compute auxiliaries
        y = model.functions['auxiliary'](sol[:, :n_s], sol[:, n_s:], p)
        sol = np.column_stack([sol, y])
    else:
        colnames = model.symbols['states'] + model.symbols['controls']

    sol = np.column_stack([sol, epsilons])
    colnames = colnames + model.symbols['shocks']

    ts = pd.DataFrame(sol, columns=colnames)
    return ts


def det_residual(model, guess, start, final, shocks, diff=True,
                 jactype='sparse'):
    '''
    Computes the residuals, the derivatives of the stacked-time system.
    :param model: an fga model
    :param guess: the guess for the simulated values. An `(n_s.n_x) x N` array,
                  where n_s is the number of states,
    n_x the number of controls, and `N` the length of the simulation.
    :param start: initial boundary condition (initial value of the states)
    :param final: final boundary condition (last value of the controls)
    :param shocks: values for the exogenous shocks
    :param diff: if True, the derivatives are computes
    :return: a list with two elements:
        - an `(n_s.n_x) x N` array with the residuals of the system
        - a `(n_s.n_x) x N x (n_s.n_x) x N` array representing the jacobian of
             the system
    '''

    # TODO: compute a sparse derivative and ensure the solvers can deal with it

    n_s = len(model.symbols['states'])
    n_x = len(model.symbols['controls'])

    n_e = len(model.symbols['shocks'])
    N = guess.shape[0]

    p = model.calibration['parameters']

    f = model.functions['arbitrage']
    g = model.functions['transition']

    vec = guess[:-1, :]
    vec_f = guess[1:, :]

    s = vec[:, :n_s]
    x = vec[:, n_s:]
    S = vec_f[:, :n_s]
    X = vec_f[:, n_s:]

    e = shocks[:-1, :]
    E = shocks[1:, :]

    if diff:
        SS, SS_s, SS_x, SS_e = g(s, x, e, p, diff=True)
        R, R_s, R_x, R_e, R_S, R_X = f(s, x, E, S, X, p, diff=True)
    else:
        SS = g(s, x, e, p)
        R = f(s, x, E, S, X, p)

    res_s = SS - S
    res_x = R

    res = np.zeros((N, n_s+n_x))

    res[1:, :n_s] = res_s
    res[:-1, n_s:] = res_x

    res[0, :n_s] = - (guess[0, :n_s] - start)
    res[-1, n_s:] = - (guess[-1, n_s:] - guess[-2, n_s:])

    if not diff:
        return res
    else:

        sparse_jac = False
        if not sparse_jac:

            # we compute the derivative matrix
            res_s_s = SS_s
            res_s_x = SS_x

            # next block is probably very inefficient
            jac = np.zeros((N, n_s+n_x, N, n_s+n_x))
            for i in range(N-1):
                jac[i, n_s:, i, :n_s] = R_s[i, :, :]
                jac[i, n_s:, i, n_s:] = R_x[i, :, :]
                jac[i, n_s:, i+1, :n_s] = R_S[i, :, :]
                jac[i, n_s:, i+1, n_s:] = R_X[i, :, :]
                jac[i+1, :n_s, i, :n_s] = SS_s[i, :, :]
                jac[i+1, :n_s, i, n_s:] = SS_x[i, :, :]
                jac[i+1, :n_s, i+1, :n_s] = -np.eye(n_s)
                # jac[i,n_s:,i,:n_s] = R_s[i,:,:]
                # jac[i,n_s:,i,n_s:] = R_x[i,:,:]
                # jac[i+1,n_s:,i,:n_s] = R_S[i,:,:]
                # jac[i+1,n_s:,i,n_s:] = R_X[i,:,:]
                # jac[i,:n_s,i+1,:n_s] = SS_s[i,:,:]
                # jac[i,:n_s,i+1,n_s:] = SS_x[i,:,:]
                # jac[i+1,:n_s,i+1,:n_s] = -np.eye(n_s)
            jac[0, :n_s, 0, :n_s] = - np.eye(n_s)
            jac[-1, n_s:, -1, n_s:] = - np.eye(n_x)
            jac[-1, n_s:, -2, n_s:] = + np.eye(n_x)
            nn = jac.shape[0]*jac.shape[1]
            res = res.ravel()
            jac = jac.reshape((nn, nn))

        if jactype == 'sparse':
            from scipy.sparse import csc_matrix, csr_matrix
            jac = csc_matrix(jac)
            # scipy bug ? I don't get the same with csr

        return [res, jac]


if __name__ == '__main__':

    # this example computes the response of the rbc economy to a series of
    # expected productivity shocks. investment is bounded by an exogenous value
    # 0.2, so that investment is constrained in the first periods

    # TODO: propose a meaningful economic example

    from dolo import yaml_import

    m = yaml_import("../../../examples/models/Figv4_1191.yaml")
    T = 100
    g_list = [0.2]*10+[0.4]

    # first try using a list
    sol1 = deterministic_solve(m, shocks=g_list)

    # then try using a 1d array
    sol2 = deterministic_solve(m, shocks=np.asarray(g_list))

    # then try using a 2d array
    g_shock = np.array(g_list)[:, None]
    sol3 = deterministic_solve(m, shocks=g_shock)

    # now try using a dict
    sol4 = deterministic_solve(m, shocks={"g": g_list})

    # now try using a DataFrame
    sol5 = deterministic_solve(m, shocks=pd.DataFrame({"g": g_list}))

    # check that they are all the same
    for s in [sol2, sol3, sol4, sol5]:
        assert max(abs(sol1-s).max()) == 0.0

    m2 = yaml_import("../../../examples/models/rmt3_ch11.yaml")
    sol = deterministic_solve(m, shocks={"g": [0.2]*10+[0.4]}, T=T)
