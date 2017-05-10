import numpy
import pandas
from matplotlib import pyplot

from dolo.numeric.optimize.ncpsolve import ncpsolve
from dolo.numeric.optimize.newton import newton as newton_solver
from dolo.numeric.optimize.newton import SerialDifferentiableFunction

## TODO: extend for mc process

def response(model, dr, varname, T=40, impulse:float=None):

    i_exo = model.symbols["exogenous"].index(varname)

    if impulse is None:
        impulse = numpy.sqrt( model.exogenous.sigma[i_exo, i_exo] ) # works only for IID/AR1

    e1 = numpy.zeros(len(model.symbols["exogenous"]))
    e1[i_exo] = impulse

    m_simul = model.exogenous.response(T, e1)
    m_simul = m_simul[:,None,:]

    sim = simulate(model, dr, N=1, T=T, driving_process=m_simul, stochastic=False)

    irf = sim.sel(N=0)
    return irf


def simulate(model, dr, N=1, T=40, s0=None, driving_process=None, m0=None, seed=42, stochastic=True):
    '''
    Simulate a model using the specified decision rule.

    Parameters
    ---------
    model: NumericModel

    dr: decision rule

    s0: ndarray
        initial state where all simulations start

    driving_process: ndarray
        realization of exogenous driving process (drawn randomly if None)

    N: int
        number of simulations
    T: int
        horizon for the simulations
    seed: int
        used to initialize the random number generator. Use it to replicate
        exact same results among simulations
    discard: boolean (False)
        if True, then all simulations containing at least one non finite value
        are discarded

    Returns
    -------
    xarray.DataArray:
        returns a ``T x N x n_v`` array where ``n_v``
           is the number of variables.
    '''

    calib = model.calibration
    parms = numpy.array(calib['parameters'])

    if s0 is None:
        s0 = calib['states']

    if m0 is None:
        m0 = calib['exogenous']

    x0 = dr.eval_ms(m0,s0)

    if driving_process is None:
        m_simul = model.exogenous.simulate(N, T, m0=m0, stochastic=stochastic)
    else:
        m_simul = driving_process
    s_simul = numpy.zeros((T, N, s0.shape[0]))
    x_simul = numpy.zeros((T, N, x0.shape[0]))

    s_simul[0, :, :] = s0[None, :]
    x_simul[0, :, :] = x0[None, :]

    fun = model.functions
    f = model.functions['arbitrage']
    g = model.functions['transition']

    numpy.random.seed(seed)

    mp = m0
    for i in range(T):
        m = m_simul[i,:,:]
        s = s_simul[i,:,:]
        x = dr.eval_ms(m,s)
        x_simul[i,:,:] = x
        ss = g(mp, s, x, m, parms)
        if i < T-1:
            s_simul[i + 1, :, :] = ss
        mp = m

    if 'auxiliary' not in fun:  # TODO: find a better test than this
        l = [s_simul, x_simul]
        varnames = model.symbols['states'] + model.symbols['controls']
    else:
        aux = fun['auxiliary']
        a_simul = aux(
            m_simul.reshape((N * T, -1)),
            s_simul.reshape((N * T, -1)),
            x_simul.reshape((N * T, -1)), parms)
        a_simul = a_simul.reshape(T, N, -1)
        l = [m_simul, s_simul, x_simul, a_simul]
        varnames = model.symbols['exogenous'] + model.symbols['states'] + model.symbols[
            'controls'] + model.symbols['auxiliaries']

    simul = numpy.concatenate(l, axis=2)

    import xarray as xr
    data = xr.DataArray(
            simul,
            dims=['T','N','V'],
            coords={'T': range(T), 'N': range(N), 'V': varnames}
        )

    return data


def plot_decision_rule(model, dr, state, plot_controls=None, bounds=None, n_steps=100, s0=None, i0=None, m0=None, **kwargs):

    import numpy

    states_names = model.symbols['states']
    controls_names = model.symbols['controls']
    index = states_names.index(str(state))

    if bounds is None:
        try:
            endo_grid = dr.endo_grid
            bounds = [endo_grid.min[index], endo_grid.max[index]]
        except:
            approx = model.get_grid(**grid)
            bounds = [approx.a[index], approx.b[index]]
        if bounds is None:
            raise Exception("No bounds provided for simulation or by model.")

    values = numpy.linspace(bounds[0], bounds[1], n_steps)

    if s0 is None:
        s0 = model.calibration['states']



    svec = numpy.row_stack([s0]*n_steps)
    svec[:,index] = values

    dp = model.exogenous.discretize()

    if (i0 is None) and (m0 is None):
        from dolo.numeric.grids import UnstructuredGrid
        if isinstance(dp.grid, UnstructuredGrid):
            n_ms = dp.n_nodes
            [q,r] = divmod(n_ms,2)
            i0 = q-1+r
        else:
            m0 = model.calibration["exogenous"]


    if i0 is not None:
        m = dp.node(i0)
        xvec = dr.eval_is(i0,svec)
    elif m0 is not None:
        m = m0
        xvec = dr.eval_ms(m0,svec)

    mm = numpy.row_stack([m]*n_steps)
    l = [mm, svec, xvec]

    series = model.symbols['exogenous'] + model.symbols['states'] + model.symbols['controls']

    if 'auxiliary' in model.functions:
        p = model.calibration['parameters']
        pp = numpy.row_stack([p]*n_steps)
        avec = model.functions['auxiliary'](mm, svec,xvec,pp)
        l.append(avec)
        series.extend(model.symbols['auxiliaries'])

    import pandas
    tb = numpy.concatenate(l, axis=1)
    df = pandas.DataFrame(tb, columns=series)

    if plot_controls is None:
        return df
    else:
        from matplotlib import pyplot
        if isinstance(plot_controls, str):
            cn = plot_controls
            pyplot.plot(values, df[cn], **kwargs)
        else:
            for cn in  plot_controls:
                pyplot.plot(values, df[cn], label=cn, **kwargs)
            pyplot.legend()
        pyplot.xlabel('state = {} | mstate = {}'.format(state, i0))
