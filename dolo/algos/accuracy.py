from __future__ import division

from dolo.numeric.misc import mlinspace
import numpy
import numpy as np

class RectangularDomain:

    def __init__(self,smin,smax,orders):
        self.d = len(smin)
        self.smin = smin
        self.smax = smax
        self.bounds = np.row_stack( [smin,smax] )
        self.orders = numpy.array(orders, dtype=int)
        nodes = [np.linspace(smin[i], smax[i], orders[i]) for i in range(len(orders))]

        self.nodes = nodes
        self.grid = mlinspace(smin,smax,orders)

    def find_cell(self, x):
        """
        @param x: Nxd array
        @return: Nxd array with line i containing the indices of cell containing x[i,:]
        """

        inf = self.smin
        sup = self.smax
        N = x.shape[0]
        indices = numpy.zeros((N, self.d), dtype=int)
        for i in range(self.d):
            xi =(x[:,i] - inf[i])/(sup[i]-inf[i])
            ni = numpy.floor( xi*self.orders[i] )
            ni = numpy.minimum(numpy.maximum(ni,0),self.orders[i]-1)
            indices[:,i] = ni

        return indices

    def compute_density(self, x):

        import time
        t1 = time.time()


        cell_indices = self.find_cell(x)
        t2 = time.time()

        keep = numpy.isfinite( numpy.sum(cell_indices, axis=1) )
        cell_indices = cell_indices[keep,:]

        npoints = cell_indices.shape[0]
        counts = numpy.zeros( self.orders.prod(), dtype=numpy.int)

        #print cell_indices
        for j in range(cell_indices.shape[0]):
            ind = cell_indices[j,:].prod() #### that is not true
            counts[ind] += 1

        dens = counts/npoints

        t3 = time.time()

        print("So long : {}, {}".format(t2-t1, t3-t2 ))
        return dens

def test_residuals(s,dr, f,g,parms, epsilons, weights):

    n_draws = epsilons.shape[1]

    n_g = s.shape[1]
    x = dr(s)
    n_x = x.shape[0]

    ss = np.tile(s, (1,n_draws))
    xx = np.tile(x, (1,n_draws))
    ee = np.repeat(epsilons, n_g , axis=1)

    ssnext = g(ss,xx,ee,parms)
    xxnext = dr(ssnext)

    val = f(ss,xx,ee,ssnext,xxnext,parms)

    errors = np.zeros( (n_x,n_g) )
    for i in range(n_draws):
        errors += weights[i] * val[:,n_g*i:n_g*(i+1)]

    return errors

#    squared_errors = np.power(errors,2)
#    std_errors = np.sqrt( np.sum(squared_errors,axis=1) ) /(squared_errors.shape[1])
#    return std_errors

def get_fg_functions(model):

    if model.model_type == 'fga':
        ff = model.functions['arbitrage']
        gg = model.functions['transition']
        aa = model.functions['auxiliary']
        g = lambda s,x,e,p : gg(s,x,aa(s,x,p),e,p)
        f = lambda s,x,e,S,X,p : ff(s,x,aa(s,x,p),S,X,aa(S,X,p),p)
    else:
        f = model.functions['arbitrage']
        g = model.functions['transition']

    return [f,g]

def omega(dr, model, bounds, orders, exponent='inf', n_exp=10000, time_weight=None, return_everything=False):



    assert(model.model_type =='fga')

    [f,g] = get_fg_functions(model)

    # TODO: this is 2d-only !


    N_epsilons = 1000

    sigma = model.covariances
    parms = model.calibration['parameters']
    mean = numpy.zeros(sigma.shape[0])
    N_epsilons=100
    numpy.random.seed(1)
    epsilons = numpy.random.multivariate_normal(mean, sigma, N_epsilons)
    weights = np.ones(epsilons.shape[1])/N_epsilons

    domain = RectangularDomain(bounds[0,:], bounds[1,:], orders)
    grid = domain.grid

    print(grid.shape)
    n_s = len(model.symbols['states'])

    errors = test_residuals( grid, dr, f, g, parms, epsilons, weights )
    errors = abs(errors)

    print(errors.shape)
    print(orders)
    errors = errors.reshape( list(orders)+ [-1] )
    print(errors.shape)


    if exponent == 'inf':
        criterium = numpy.max(abs(errors), axis=1)
    elif exponent == 'L2':
        squared_errors = np.power(errors,2)
        criterium = np.sqrt( np.sum(squared_errors,axis=1) ) /(squared_errors.shape[1])

    if time_weight:
        horizon = time_weight[0]
        beta = time_weight[1]
        s0 = time_weight[2]
    else:
        raise Exception()

    from dolo.numeric.simulations import simulate
    simul = simulate( model ,dr,s0, sigma, n_exp=n_exp, horizon=horizon+1, discard=True)

    # s_simul = simul[:n_s,:,:]
    s_simul = simul[:,:,:n_s]

    densities = [domain.compute_density(s_simul[t,:,:]) for t in range(horizon)]

    ergo_dens = densities[-1]

    print(errors.shape)
    print(ergo_dens.shape)

    ergo_error = numpy.tensordot( errors, ergo_dens, axes=((0,1),(0,1)))
    mean_error = numpy.tensordot( errors, (ergo_dens*0+1)/len(ergo_dens.flatten()), axes=((0,1),(0,1)))
    max_error = numpy.max(errors,axis=0)
    max_error = numpy.max(max_error,axis=0)

    time_weighted_errors  = max_error*0

    for i in range(horizon):
        err =  numpy.tensordot( errors, densities[i], axes=((0,1),(0,1)))
        time_weighted_errors += beta**i * err
    time_weighted_errors /= (1-beta**(horizon-1))/(1-beta)

#        print(numpy.mean(errors[0,:,:].flatten()))
#        print(numpy.mean(errors[1,:,:].flatten()))
    if return_everything:
        d = dict(
            errors = errors,
            densities = densities,
            bounds = bounds,
            mean = mean_error,
            max = max_error,
            ergo = ergo_error,
            time_weighted = time_weighted_errors,
            simulations = s_simul,
            domain = domain
        )
        return d
    else:
        return [mean_error, max_error, ergo_error, time_weighted_errors]

    return criterium


def denhaanerrors( model, dr, s0=None, horizon=100, n_sims=10, seed=0 ):

    assert(model.model_type in ('fg', 'fga'))

    from dolo.numeric.discretization.quadrature import gauss_hermite_nodes
    from dolo.numeric.simulations import simulate

    parms = model.calibration['parameters']

    n_x = len(model.symbols['controls'])
    n_s = len(model.symbols['states'])

    sigma = model.covariances
    mean = sigma[0,:]*0


    orders = [5]*len(mean)
    [nodes, weights] = gauss_hermite_nodes(orders, sigma)

    if s0 is None:
        s0 = model.calibration['states']

    # standard simulation
    simul = simulate(model, dr, s0, sigma, horizon=horizon, n_exp=n_sims, parms=parms, seed=seed)
    simul_se = simulate(model, dr, s0, sigma, horizon=horizon, n_exp=n_sims, parms=parms, seed=seed, solve_expectations=True, nodes=nodes, weights=weights)



    x_simul = simul[:,n_s:n_s+n_x,:]
    x_simul_se = simul_se[:,n_s:n_s+n_x,:]



    print(model.symbols)
    diff = abs( x_simul_se - x_simul )
    error_1 = (diff).max(axis=0).mean(axis=1)
    error_2 = (diff).mean(axis=0).mean(axis=1)



    return [error_1, error_2]
