import numpy
import pandas
from matplotlib import pyplot

from dolo.algos.dtcscc.time_iteration import step_residual
from dolo.numeric.optimize.ncpsolve import ncpsolve
from dolo.numeric.optimize.newton import newton as newton_solver
from dolo.numeric.optimize.newton import SerialDifferentiableFunction


def simulate(model, dr, s0=None, n_exp=0, horizon=40, seed=1, discard=False,
             solve_expectations=False, nodes=None, weights=None,
             forcing_shocks=None, return_array=False):
    '''
    Simulate a model using the specified decision rule.

    Parameters
    ---------
    model: NumericModel
        an "fg" or "fga" model

    dr: decision rule

    s0: ndarray
        initial state where all simulations start
    n_exp: int
        number of simulations. Use 0 for impulse-response functions
    horizon: int
        horizon for the simulations
    seed: int
        used to initialize the random number generator. Use it to replicate
        exact same results among simulations
    discard: boolean (False)
        if True, then all simulations containing at least one non finite value
        are discarded
    solve_expectations: boolean (False)
        if True, Euler equations are solved at each step using the controls to
        form expectations
    nodes: ndarray
        if `solve_expectations` is True use ``nodes`` for integration
    weights: ndarray
        if `solve_expectations` is True use ``weights`` for integration
    forcing_shocks: ndarray
        specify an exogenous process of shocks (requires ``n_exp<=1``)
    return_array: boolean (False)
        if True, then all return a numpy array containing simulated data,
        otherwise return a pandas DataFrame or Panel.

    Returns
    -------
    ndarray or pandas.Dataframe:
         - if `n_exp<=1` returns a DataFrame object
         - if `n_exp>1` returns a ``horizon x n_exp x n_v`` array where ``n_v``
           is the number of variables.
    '''

    if n_exp == 0:
        irf = True
        n_exp = 1
    else:
        irf = False

    calib = model.calibration

    parms = numpy.array(calib['parameters'])

    sigma = model.covariances

    if s0 is None:
        s0 = calib['states']

    # s0 = numpy.atleast_2d(s0.flatten()).T
    x0 = dr(s0)
    s_simul = numpy.zeros((horizon, n_exp, s0.shape[0]))
    x_simul = numpy.zeros((horizon, n_exp, x0.shape[0]))

    s_simul[0, :, :] = s0[None, :]
    x_simul[0, :, :] = x0[None, :]

    fun = model.functions

    f = model.functions['arbitrage']
    g = model.functions['transition']

    numpy.random.seed(seed)

    for i in range(horizon):
        mean = numpy.zeros(sigma.shape[0])
        if irf:
            if forcing_shocks is not None and i < forcing_shocks.shape[0]:
                epsilons = forcing_shocks[i, :]
            else:
                epsilons = numpy.zeros((1, sigma.shape[0]))
        else:
            epsilons = numpy.random.multivariate_normal(mean, sigma, n_exp)
        s = s_simul[i, :, :]

        x = dr(s)

        if solve_expectations:

            lbfun = model.functions['controls_lb']
            ubfun = model.functions['controls_ub']
            lb = lbfun(s, parms)
            ub = ubfun(s, parms)

            def fobj(t):
                return step_residual(s, t, dr, f, g, parms, nodes, weights)

            dfobj = SerialDifferentiableFunction(fobj)
            [x, nit] = ncpsolve(dfobj, lb, ub, x)


        x_simul[i, :, :] = x

        ss = g(s, x, epsilons, parms)
        if i < (horizon - 1):
            s_simul[i + 1, :, :] = ss

    if 'auxiliary' not in fun:  # TODO: find a better test than this
        l = [s_simul, x_simul]
        varnames = model.symbols['states'] + model.symbols['controls']
    else:
        aux = fun['auxiliary']

        a_simul = aux(
            s_simul.reshape((n_exp * horizon, -1)),
            x_simul.reshape((n_exp * horizon, -1)), parms)
        a_simul = a_simul.reshape(horizon, n_exp, -1)

        l = [s_simul, x_simul, a_simul]
        varnames = model.symbols['states'] + model.symbols[
            'controls'] + model.symbols['auxiliaries']

    simul = numpy.concatenate(l, axis=2)

    if discard:
        iA = -numpy.isnan(x_simul)
        valid = numpy.all(numpy.all(iA, axis=0), axis=1)
        simul = simul[:, valid, :]
        n_kept = s_simul.shape[1]
        if n_exp > n_kept:
            print('Discarded {}/{}'.format(n_exp - n_kept, n_exp))

    if return_array:
        return simul

    if irf or (n_exp == 1):
        simul = simul[:, 0, :]
        ts = pandas.DataFrame(simul, columns=varnames)
        return ts
    else:
        panel = pandas.Panel(simul.swapaxes(0, 1), minor_axis=varnames)
        return panel


def plot_decision_rule(model, dr, state, plot_controls=None, bounds=None,
                       n_steps=10, s0=None, **kwargs):
    """
    Plots decision rule

    Parameters:
    -----------

    model:
        "fg" or "fga" model

    dr:
        decision rule

    state:
        state variable that is supposed to vary

    plot_controls: string, list or None
        - if None, return a pandas dataframe
        - if a string denoting a control variable, plot this variable as a
          function of the state
        - if a list of strings, plot several variables

    bounds: array_like
        the state variable varies from bounds[0] to bounds[1]. By default,
        boundaries are looked for in the the decision rule then in
        the model.

    n_steps: int
        number of points to be plotted

    s0: array_like or None
        value for the state variables, that remain constant. Defaults to
        `model.calibration['states']`

    Returns:
    --------

    dataframe or plot, depending on the value of `plot_controls`

    """

    states_names = model.symbols['states']
    controls_names = model.symbols['controls']
    index = states_names.index(str(state))

    if bounds is None:
        if hasattr(dr, 'a'):
            bounds = [dr.a[index], dr.b[index]]
        else:
            approx = model.options['approximation_space']
            bounds = [approx['a'][index], approx['b'][index]]

    values = numpy.linspace(bounds[0], bounds[1], n_steps)
    if s0 is None:
        s0 = model.calibration['states']
    svec = numpy.row_stack([s0] * n_steps)
    svec[:, index] = values
    xvec = dr(svec)

    l = [svec, xvec]
    series = model.symbols['states'] + model.symbols['controls']

    if 'auxiliary' in model.functions:

        p = model.calibration['parameters']
        pp = numpy.row_stack([p] * n_steps)
        avec = model.functions['auxiliary'](svec, xvec, pp)
        l.append(avec)
        series.extend(model.symbols['auxiliaries'])

    tb = numpy.concatenate(l, axis=1)
    df = pandas.DataFrame(tb, columns=series)

    if plot_controls is None:
        return df
    else:
        if isinstance(plot_controls, str):
            cn = plot_controls
            pyplot.plot(values, df[cn], **kwargs)
        else:
            for cn in plot_controls:
                pyplot.plot(values, df[cn], label=cn, **kwargs)
            pyplot.legend()
        pyplot.xlabel('state = {}'.format(state))


def test_simulations():

    import time
    from matplotlib.pyplot import hist, show, figure, plot, title
    from dolo import yaml_import, approximate_controls
    from dolo.numeric.discretization import gauss_hermite_nodes
    model = yaml_import('../../examples/models/rbc.yaml')

    dr = approximate_controls(model)

    parms = model.calibration['parameters']
    sigma = model.covariances

    s0 = dr.S_bar

    horizon = 50

    t1 = time.time()
    simul = simulate(model,
                     dr,
                     s0,
                     sigma,
                     n_exp=1000,
                     parms=parms,
                     seed=1,
                     horizon=horizon)
    t2 = time.time()

    print("Took: {}".format(t2 - t1))

    N = 80
    [x, w] = gauss_hermite_nodes(N, sigma)

    t3 = time.time()
    simul_2 = simulate(model,
                       dr,
                       s0,
                       sigma,
                       n_exp=1000,
                       parms=parms,
                       horizon=horizon,
                       seed=1,
                       solve_expectations=True,
                       nodes=x,
                       weights=w)
    t4 = time.time()

    print("Took: {}".format(t4 - t3))

    timevec = numpy.array(range(simul.shape[2]))

    figure()
    for k in range(10):
        plot(simul[:, k, 0] - simul_2[:, k, 0])
    title("Productivity")
    show()

    figure()
    for k in range(10):
        plot(simul[:, k, 1] - simul_2[:, k, 1])
    title("Investment")
    show()

    #
    #    figure()
    #    plot(simul[0,0,:])
    #    plot(simul_2[0,0,:])
    #    show()

    figure()
    for i in range(horizon):
        hist(simul[i, :, 0], bins=50)

    show()
    # plot(timevec,s_simul[0,0,:])


if __name__ == "__main__":

    test_simulations()
