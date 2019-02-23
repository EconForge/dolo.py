import numpy as np
import pandas as pd
from numpy import array, atleast_2d, linspace, zeros
from scipy.optimize import root

from dolo.numeric.optimize.ncpsolve import ncpsolve


def _shocks_to_epsilons(model, shocks, T):
    """
    Helper function to support input argument `shocks` being one of many
    different data types. Will always return a `T, n_e` matrix.
    """
    n_e = len(model.calibration['exogenous'])

    # if we have a DataFrame, convert it to a dict and rely on the method below
    if isinstance(shocks, pd.DataFrame):
        shocks = {k: shocks[k].tolist() for k in shocks.columns}

    # handle case where shocks might be a dict. Be careful to handle case where
    # value arrays are not the same length
    if isinstance(shocks, dict):
        epsilons = np.zeros((T + 1, n_e))
        for (i, k) in enumerate(model.symbols["exogenous"]):
            if k in shocks:
                this_shock = shocks[k]
                epsilons[:len(this_shock), i] = this_shock
                epsilons[len(this_shock):, i] = this_shock[-1]
            else:
                # otherwise set to value in calibration
                epsilons[:, i] = model.calibration["exogenous"][i]

        return epsilons

    # read from calibration if not given
    if shocks is None:
        shocks = model.calibration["exogenous"]

    # now we just assume that shocks is array-like and try using the output of
    # np.asarray(shocks)
    shocks = np.asarray(shocks)
    shocks = shocks.reshape((-1, n_e))

    # until last period, exogenous shock takes its last value
    epsilons = np.zeros((T + 1, n_e))
    epsilons[:(shocks.shape[0] - 1), :] = shocks[1:, :]
    epsilons[(shocks.shape[0] - 1):, :] = shocks[-1:, :]

    return epsilons


def deterministic_solve(
        model,
        shocks=None,
        s1=None,
        T=100,
        ignore_constraints=False,
        maxit=100,
        initial_guess=None,
        verbose=True,
        solver='ncpsolve',
        tol=1e-6):
    """
    Computes a perfect foresight simulation using a stacked-time algorithm.

    The initial state is specified either by providing a series of exogenous
    shocks and assuming the model is initially in equilibrium with the first
    value of the shock, or by specifying an initial value for the states.

    Parameters
    ----------
    model : Model
        Model to be solved
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
    s1 : ndarray or dict
        a vector with the value of initial states
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

    # definitions
    n_s = len(model.calibration['states'])
    n_x = len(model.calibration['controls'])
    p = model.calibration['parameters']

    epsilons = _shocks_to_epsilons(model, shocks, T)

    m0 = epsilons[0, :]

    # get initial steady-state
    from dolo.algos.steady_state import find_steady_state
    # TODO: use initial_guess for steady_state
    # TODO:

    if s1 is None:
        start_state = find_steady_state(model, m=m0)
        s0 = start_state['states']
        x0 = start_state['controls']
        m1 = epsilons[1,:]
        s1 = model.functions['transition'](m0, s0, x0, m1, p)
    else:
        s0 = model.calibration['states']*np.nan
        x0 = model.calibration['controls']*np.nan
        s1 = np.array(s1)

    x1_g = model.calibration['controls']  # we can do better here
    sT_g = model.calibration['states']  # we can do better here
    xT_g = model.calibration['controls']  # we can do better here


    if initial_guess is None:
        start = np.concatenate([s1, x1_g])
        final = np.concatenate([sT_g, xT_g])
        initial_guess = np.row_stack(
            [start * (1 - l) + final * l for l in linspace(0.0, 1.0, T)])

    else:
        if isinstance(initial_guess, pd.DataFrame):
            initial_guess = np.array(initial_guess[model.symbols['states']+model.symbols['controls']])
        initial_guess = initial_guess[1:,:]
        initial_guess = initial_guess[:, :n_s + n_x]

    sh = initial_guess.shape


    if model.x_bounds and not ignore_constraints:
        initial_states = initial_guess[:, :n_s]
        [lb, ub] = [u(epsilons[1:,:], initial_states, p) for u in model.x_bounds]
        lower_bound = initial_guess * 0 - np.inf
        lower_bound[:, n_s:] = lb
        upper_bound = initial_guess * 0 + np.inf
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

    if not ignore_constraints:

        def ff(vec):
            return det_residual(
                model, vec.reshape(sh), s1, xT_g, epsilons[1:, :], jactype='sparse')

        v0 = initial_guess.ravel()
        if solver=='ncpsolve':
            sol, nit = ncpsolve(
                ff,
                lower_bound.ravel(),
                upper_bound.ravel(),
                initial_guess.ravel(),
                verbose=verbose,
                maxit=maxit,
                tol=tol,
                jactype='sparse')
        else:
            from dolo.numeric.extern.lmmcp import lmmcp
            sol = lmmcp(
                lambda u: ff(u)[0],
                lambda u: ff(u)[1].todense(),
                lower_bound.ravel(),
                upper_bound.ravel(),
                initial_guess.ravel(),
                verbose=verbose)
            nit = -1

        sol = sol.reshape(sh)

    else:

        def ff(vec):
            ll = det_residual(
                model, vec.reshape(sh), s1, xT_g, epsilons[1:, :],
                diff=True)
            return(ll)

        v0 = initial_guess.ravel()
        # from scipy.optimize import root
        # sol = root(ff, v0, jac=True)
        # sol = sol.x.reshape(sh)
        from dolo.numeric.optimize.newton import newton
        sol, nit = newton(ff, v0, jactype='sparse')
        sol = sol.reshape(sh)

    sx = np.concatenate([s0, x0])
    # sol = sol[:-1, :]

    sol = np.concatenate([sx[None, :], sol], axis=0)
    # epsilons = np.concatenate([epsilons[:1,:], epsilons], axis=0)

    if 'auxiliary' in model.functions:
        colnames = (model.symbols['states'] + model.symbols['controls'] +
                    model.symbols['auxiliaries'])
        # compute auxiliaries
        y = model.functions['auxiliary'](epsilons, sol[:, :n_s], sol[:, n_s:],
                                         p)
        sol = np.column_stack([sol, y])
    else:
        colnames = model.symbols['states'] + model.symbols['controls']

    sol = np.column_stack([sol, epsilons])
    colnames = colnames + model.symbols['exogenous']

    ts = pd.DataFrame(sol, columns=colnames)
    return ts


def det_residual(model,
                 guess,
                 start,
                 final,
                 shocks,
                 diff=True,
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

    # n_e = len(model.symbols['shocks'])
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

    m = shocks[:-1, :]
    M = shocks[1:, :]

    if diff:
        SS, SS_m, SS_s, SS_x, SS_M = g(m, s, x, M, p, diff=True)
        R, R_m, R_s, R_x, R_M, R_S, R_X = f(m, s, x, M, S, X, p, diff=True)
    else:
        SS = g(m, s, x, M, p)
        R = f(m, s, x, M, S, X, p)

    res_s = SS - S
    res_x = R

    res = np.zeros((N, n_s + n_x))

    res[1:, :n_s] = res_s
    res[:-1, n_s:] = res_x

    res[0, :n_s] = -(guess[0, :n_s] - start)
    res[-1, n_s:] = -(guess[-1, n_s:] - guess[-2, n_s:])

    if not diff:
        return res
    else:

        sparse_jac = False
        if not sparse_jac:

            # we compute the derivative matrix
            res_s_s = SS_s
            res_s_x = SS_x

            # next block is probably very inefficient
            jac = np.zeros((N, n_s + n_x, N, n_s + n_x))
            for i in range(N - 1):
                jac[i, n_s:, i, :n_s] = R_s[i, :, :]
                jac[i, n_s:, i, n_s:] = R_x[i, :, :]
                jac[i, n_s:, i + 1, :n_s] = R_S[i, :, :]
                jac[i, n_s:, i + 1, n_s:] = R_X[i, :, :]
                jac[i + 1, :n_s, i, :n_s] = SS_s[i, :, :]
                jac[i + 1, :n_s, i, n_s:] = SS_x[i, :, :]
                jac[i + 1, :n_s, i + 1, :n_s] = -np.eye(n_s)
                # jac[i,n_s:,i,:n_s] = R_s[i,:,:]
                # jac[i,n_s:,i,n_s:] = R_x[i,:,:]
                # jac[i+1,n_s:,i,:n_s] = R_S[i,:,:]
                # jac[i+1,n_s:,i,n_s:] = R_X[i,:,:]
                # jac[i,:n_s,i+1,:n_s] = SS_s[i,:,:]
                # jac[i,:n_s,i+1,n_s:] = SS_x[i,:,:]
                # jac[i+1,:n_s,i+1,:n_s] = -np.eye(n_s)
            jac[0, :n_s, 0, :n_s] = -np.eye(n_s)
            jac[-1, n_s:, -1, n_s:] = -np.eye(n_x)
            jac[-1, n_s:, -2, n_s:] = +np.eye(n_x)
            nn = jac.shape[0] * jac.shape[1]
            res = res.ravel()
            jac = jac.reshape((nn, nn))

        if jactype == 'sparse':
            from scipy.sparse import csc_matrix, csr_matrix
            jac = csc_matrix(jac)
            # scipy bug ? I don't get the same with csr

        return [res, jac]
