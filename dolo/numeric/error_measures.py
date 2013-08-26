#from dolo.compiler.compiler_global import test_residuals
from dolo.numeric.interpolation.interpolation import RectangularDomain

from dolo.compiler.compiler_global import CModel
import numpy
import numpy as np

#testgrid = RectangularGrid(domain,[20,20])

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
    val = f(ss,xx,ssnext,xxnext,ee,parms)

    errors = np.zeros( (n_x,n_g) )
    for i in range(n_draws):
        errors += weights[i] * val[:,n_g*i:n_g*(i+1)]

    return errors

#    squared_errors = np.power(errors,2)
#    std_errors = np.sqrt( np.sum(squared_errors,axis=1) ) /(squared_errors.shape[1])
#    return std_errors


def omega(dr, model, bounds, orders, exponent='inf', n_exp=10000, time_weight=None, return_everything=False):

    N_epsilons = 1000


    [y,x,parms] = model.read_calibration()
    sigma = model.read_covariances()
    mean = numpy.zeros(sigma.shape[0])
    N_epsilons=100
    numpy.random.seed(1)
    epsilons = numpy.random.multivariate_normal(mean, sigma, N_epsilons).T
    weights = np.ones(epsilons.shape[1])/N_epsilons

    domain = RectangularDomain(bounds[0,:], bounds[1,:], orders)

    gc = CModel(model)
    f = gc.f
    g = gc.g

    errors = test_residuals( domain.grid, dr, f, g, parms, epsilons, weights )
    errors = abs(errors)

    errors = errors.reshape( [errors.shape[0]] + orders )


    if exponent == 'inf':
        criterium = numpy.max(abs(errors), axis=1)
    elif exponent == 'L2':
        squared_errors = np.power(errors,2)
        criterium = np.sqrt( np.sum(squared_errors,axis=1) ) /(squared_errors.shape[1])

    if time_weight:
        horizon = time_weight[0]
        beta = time_weight[1]
        s0 = time_weight[2]

        from dolo.numeric.simulations import simulate
        simul = simulate( gc ,dr,s0, sigma, n_exp=n_exp, horizon=horizon+1, discard=True)

        s_simul = simul[:len(gc.controls),:,:]

        densities = [domain.compute_density(s_simul[:,:,i]) for i in range(horizon)]

        ergo_dens = densities[-1]

        ergo_error = numpy.tensordot( errors, ergo_dens, axes=((1,2),(0,1)))
        mean_error = numpy.tensordot( errors, (ergo_dens*0+1)/len(ergo_dens.flatten()), axes=((1,2),(0,1)))
        max_error = numpy.max(errors,axis=1)
        max_error = numpy.max(max_error,axis=1)

        time_weighted_errors  = max_error*0
        for i in range(horizon):
            err =  numpy.tensordot( errors, densities[i], axes=((1,2),(0,1)))
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

# def step_residual(s, x, dr, f, g, parms, epsilons, weights, x_bounds=None, serial_grid=True, with_derivatives=True):

def denhaanerrors( cmodel, dr, s0, horizon=100, n_sims=10, sigma=None, seed=0 ):

    from dolo.numeric.global_solution import step_residual
    from dolo.numeric.quadrature import gauss_hermite_nodes
    from dolo.numeric.newton import newton_solver
    from dolo.numeric.simulations import simulate

    # cmodel should be an fg model

    # the code is almost duplicated from simulate_without_error

    # dr is used to approximate future steps

    # monkey patch:


    if sigma is None:
        sigma = cmodel.sigma

    # from dolo.symbolic.model import SModel
    #
    # if isinstance(cmodel,SModel):
    #     from dolo.compiler.compiler_global import CModel_fg
    #     model = cmodel
    #     cmodel = CModel_fg(model)
    #     [y,x,parms] = model.read_calibration()
    #

    # parms = cmodel.model.read_calibration()[2]

    parms = cmodel.calibration['parameters']

    mean = sigma[0,:]*0

    n_x = len(cmodel.symbols['controls'])
    n_s = len(cmodel.symbols['states'])

    orders = [5]*len(mean)
    [nodes, weights] = gauss_hermite_nodes(orders, sigma)

    s0 = numpy.atleast_2d(s0.flatten()).T

    x0 = dr(s0)

    # standard simulation
    simul = simulate(cmodel, dr, s0, sigma, horizon=horizon, n_exp=n_sims, parms=parms, seed=seed)

    simul_se = simulate(cmodel, dr, s0, sigma, horizon=horizon, n_exp=n_sims, parms=parms, seed=seed, solve_expectations=True, nodes=nodes, weights=weights)

    x_simul = simul[n_s:,:,:]
    x_simul_se = simul_se[n_s:,:,:]


    diff = abs( x_simul_se - x_simul )
    error_1 = (diff).max(axis=2).mean(axis=1)
    error_2 = (diff).mean(axis=2).mean(axis=1)


    return [error_1, error_2]