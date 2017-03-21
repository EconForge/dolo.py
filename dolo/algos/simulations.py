import numpy
import pandas
from matplotlib import pyplot

from dolo.numeric.optimize.ncpsolve import ncpsolve
from dolo.numeric.optimize.newton import newton as newton_solver
from dolo.numeric.optimize.newton import SerialDifferentiableFunction

## TODO: extend for mc process

def simulate(model, dr, s0=None, m0=None, n_exp=1, horizon=40, seed=42, stochastic=True):
    '''
    Simulate a model using the specified decision rule.

    Parameters
    ---------
    model: NumericModel

    dr: decision rule

    m0: ndarray
        initial exogenous state where all simulations start
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

    Returns
    -------
    xarray.DataArray:
        returns a ``horizon x n_exp x n_v`` array where ``n_v``
           is the number of variables.
    '''

    calib = model.calibration
    parms = numpy.array(calib['parameters'])

    if s0 is None:
        s0 = calib['states']
    if m0 is None:
        m0 = calib['exogenous']


    m_simul = model.exogenous.simulate(n_exp, horizon, m0=m0, stochastic=stochastic)
    s_simul = numpy.zeros((horizon, n_exp, s0.shape[0]))
    x_simul = numpy.zeros((horizon, n_exp, x0.shape[0]))

    s_simul[0, :, :] = s0[None, :]
    x0 = dr(m0,s0)
    x_simul[0, :, :] = x0[None, :]

    fun = model.functions
    f = model.functions['arbitrage']
    g = model.functions['transition']

    numpy.random.seed(seed)

    mp = m0
    for i in range(horizon):
        m = m_simul[i,:,:]
        s = s_simul[i,:,:]
        x = dr(m,s)
        x_simul[i,:,:] = x
        ss = g(mp, s, x, m, parms)
        if i < horizon-1:
            s_simul[i + 1, :, :] = ss
        mp = m

    if 'auxiliary' not in fun:  # TODO: find a better test than this
        l = [s_simul, x_simul]
        varnames = model.symbols['states'] + model.symbols['controls']
    else:
        aux = fun['auxiliary']
        a_simul = aux(
            m_simul.reshape((n_exp * horizon, -1)),
            s_simul.reshape((n_exp * horizon, -1)),
            x_simul.reshape((n_exp * horizon, -1)), parms)
        a_simul = a_simul.reshape(horizon, n_exp, -1)
        l = [s_simul, x_simul, a_simul]
        varnames = model.symbols['states'] + model.symbols[
            'controls'] + model.symbols['auxiliaries']

    simul = numpy.concatenate(l, axis=2)

    import xarray as xr
    data = xr.DataArray(
            simul,
            dims=['T','N','V'],
            coords={'T': range(horizon), 'N': range(n_exp), 'V': varnames}
        )

    return data
