from __future__ import division

from dolo.numeric.misc import mlinspace
import numpy
import numpy as np
from dolo.algos.fg.convert import get_fg_functions

class EulerErrors(dict):

    @property
    def max_errors(self):
        return self['max_errors']

    @property
    def mean_errors(self):
        return self['ergodic']

    @property
    def time_weighted(self):
        return self['time_weighted']

    def __repr__(self):

        measures = ['max_errors', 'ergodic']
        measures += ['time_weighted'] if 'time_weighted' in self else []
        s = 'Euler Errors:\n'
        for m in measures:
            s += '- {:15}: {}\n'.format(m, self[m])
        return s

class DenHaanErrors(dict):

    @property
    def max_errors(self):
        return self['max_errors']

    @property
    def  mean_errors(self):
        return self['mean_errors']


    def __repr__(self):

        measures = ['max_errors', 'mean_errors']
        s = 'Den Haan errors:\n'
        for m in measures:
            s += '- {:15}: {}\n'.format(m, self[m])
        return s

def omega(model, dr, n_exp=10000, orders=None, bounds=None,
          n_draws=100, seed=0, horizon=50, s0=None,
          solve_expectations=False, time_discount=None):

    assert(model.model_type =='fga')

    [f,g] = get_fg_functions(model)

    sigma = model.covariances
    parms = model.calibration['parameters']

    mean = numpy.zeros(sigma.shape[0])

    numpy.random.seed(seed)
    epsilons = numpy.random.multivariate_normal(mean, sigma, n_draws)
    weights = np.ones(epsilons.shape[0])/n_draws

    if bounds is None:
        approx = model.options['approximation_space']
        a = approx['a']
        b = approx['b']
        bounds = numpy.row_stack([a,b])
    else:
        a,b =numpy.row_stack(bounds)

    if orders is None:
        orders = [100]*len(a)

    domain = RectangularDomain(a, b, orders)

    grid = domain.grid

    n_s = len(model.symbols['states'])

    errors = test_residuals( grid, dr, f, g, parms, epsilons, weights )
    errors = abs(errors)

    if s0 is None:
        s0 = model.calibration['states']

    from dolo.algos.fg.simulations import simulate
    simul = simulate(model, dr, s0, n_exp=n_exp, horizon=horizon+1,
                     discard=True, solve_expectations=solve_expectations)

    s_simul = simul[:,:,:n_s]

    densities = [domain.compute_density(s_simul[t,:,:]) for t in range(horizon)]
    ergo_dens = densities[-1]

    max_error = numpy.max(errors,axis=0)        # maximum errors
    ergo_error = numpy.dot(ergo_dens,errors)    # weighted by ergodic distribution

    d = dict(
            errors = errors,
            densities = densities,
            bounds = bounds,
            max_errors = max_error,
            ergodic = ergo_error,
            domain = domain
        )

    if time_discount is not None:
        beta = time_discount
        time_weighted_errors = max_error*0
        for i in range(horizon):
            err = numpy.dot(densities[i], errors)
            time_weighted_errors += beta**i * err
        time_weighted_errors /= (1-beta**(horizon-1))/(1-beta)
        d['time_weighted'] = time_weighted_errors


    return EulerErrors(d)


def denhaanerrors( model, dr, s0=None, horizon=100, n_sims=10, seed=0, integration_orders=None):

    assert(model.model_type in ('fg', 'fga'))

    from dolo.numeric.discretization.quadrature import gauss_hermite_nodes
    from dolo.algos.fg.simulations import simulate

    n_x = len(model.symbols['controls'])
    n_s = len(model.symbols['states'])

    sigma = model.covariances
    mean = sigma[0,:]*0

    if integration_orders is None:
        integration_orders = [5]*len(mean)
    [nodes, weights] = gauss_hermite_nodes(integration_orders, sigma)

    if s0 is None:
        s0 = model.calibration['states']

    # standard simulation
    simul = simulate(model, dr, s0, horizon=horizon, n_exp=n_sims, seed=seed, solve_expectations=False)
    simul_se = simulate(model, dr, s0, horizon=horizon, n_exp=n_sims, seed=seed, solve_expectations=True, nodes=nodes, weights=weights)

    x_simul = simul[:,n_s:n_s+n_x,:]
    x_simul_se = simul_se[:,n_s:n_s+n_x,:]

    diff = abs( x_simul_se - x_simul )
    error_1 = (diff).max(axis=0).mean(axis=1)
    error_2 = (diff).mean(axis=0).mean(axis=1)

    d = dict(
        max_errors=error_1,
        mean_errors=error_2,
        horizon=horizon,
        n_sims=n_sims
    )

    return DenHaanErrors(d)


class RectangularDomain:

    def __init__(self,a,b,orders):
        self.d = len(a)
        self.a = a
        self.b = b
        self.bounds = np.row_stack( [a,b] )
        self.orders = numpy.array(orders, dtype=int)
        nodes = [np.linspace(a[i], b[i], orders[i]) for i in range(len(orders))]

        self.nodes = nodes
        self.grid = mlinspace(a,b,orders)

    def find_cell(self, x):
        """
        @param x: Nxd array
        @return: Nxd array with line i containing the indices of cell containing x[i,:]
        """

        inf = self.a
        sup = self.b
        N = x.shape[0]
        indices = numpy.zeros((N, self.d), dtype=int)
        for i in range(self.d):
            xi =(x[:,i] - inf[i])/(sup[i]-inf[i])
            ni = numpy.floor( xi*self.orders[i] )
            ni = numpy.minimum(numpy.maximum(ni,0),self.orders[i]-1)
            indices[:,i] = ni

        return numpy.ravel_multi_index(indices.T, self.orders)

    def compute_density(self, x):

        import time
        t1 = time.time()


        cell_indices = self.find_cell(x)
        t2 = time.time()

        keep = numpy.isfinite( cell_indices)
        cell_linear_indices = cell_indices[keep]

        npoints = cell_indices.shape[0]

        counts = numpy.bincount(cell_linear_indices, minlength=self.orders.prod())

        dens = counts/npoints

        t3 = time.time()

        return dens



def test_residuals(s, dr, f, g, parms, epsilons, weights):

    n_draws = epsilons.shape[0]

    x = dr(s)

    [N,n_x] = x.shape
    ss = np.tile(s, (n_draws,1))
    xx = np.tile(x, (n_draws,1))
    ee = np.repeat(epsilons, N , axis=0)

    ssnext = g(ss,xx,ee,parms)
    xxnext = dr(ssnext)

    val = f(ss,xx,ee,ssnext,xxnext,parms)

    res = np.zeros( (N,n_x) )
    for i in range(n_draws):
        res += weights[i] * val[N*i:N*(i+1),:]

    return res
