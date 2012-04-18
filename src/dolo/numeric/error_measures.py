#from dolo.compiler.compiler_global import test_residuals
from dolo.numeric.interpolation import RectangularDomain

from dolo.compiler.compiler_global import GlobalCompiler
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

    [ssnext] = g(ss,xx,ee,parms)[:1]
    xxnext = dr(ssnext)
    [val] = f(ss,xx,ssnext,xxnext,ee,parms)[:1]

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

    gc = GlobalCompiler(model, substitute_auxiliary=True, solve_systems=True)
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

        from dolo.numeric.simulations import simulate_without_aux as simulate
        [s_simul, x_simul] = simulate( model ,dr,s0, sigma, n_exp=n_exp, horizon=horizon+1, discard=True)

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




if __name__ == '__main__':
    from dolo import *
    import numpy
    from dolo.numeric.global_solve import global_solve

    from risky_ss import solve_model_around_risky_ss
    bounds = numpy.row_stack([ [2, 0.6], [ 4, 1.4]])
    fname = '../../../examples/global_models/capital.yaml'
#    bounds = numpy.row_stack([ [0.8,0.8], [ 1.2,1.2]])
#    fname = '../../../examples/global_models/open_economy.yaml'

    model =  yaml_import(fname)

    orders = [20,20]
#    error1 = omega(dr, model, bounds, orders)



    dr1 = approximate_controls(model, order=1, substitute_auxiliary=True)
    dr2 = approximate_controls(model, order=2, substitute_auxiliary=True)
    #dr3 = global_solve(model, smolyak_order=5,maxit=10)
    #dr4 = global_solve(model, interp_type='spline')

    s0 = dr1.S_bar

    decision_rules = [dr1, dr2] #, dr3]#, dr4]

    error_results = [omega(dr, model, bounds, orders, time_weight=[50, 0.96, s0],return_everything=True) for dr in decision_rules]

    exit()


    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot
    pyplot.imshow(error1)
    pyplot.show()

    #dr2 = approximate_controls(model, order=2, substitute_auxiliary=True)
    #drss  = solve_model_around_risky_ss(fname)
    #dr_smol = global_solve(model)
    #dr_spline = global_solve(model, interp_type='spline', interp_orders=[50,5])
    #drs = [dr1, dr2, drss, dr_smol, dr_spline]


    #    dr3 = approximate_controls(model, order=3, substitute_auxiliary=True)
#    drg_smol = global_solve(model, bounds=bounds)
#    drg_smol_4 = global_solve(model, bounds=bounds, smolyak_order=2, N_e=80)
#    drs = [dr1,dr2,dr3,drg_smol, drg_smol_4]
    #errors_sup = [omega(dr, model, bounds, [20,20], exponent='inf') for dr in drs]
    #errors_L2 = [omega(dr, model, bounds, [20,20], exponent='L2') for dr in drs]
    #print errors_sup
    #print errors_L2

    exit()


#    dr3 = approximate_controls(model, order=3)

    drg_smol = global_solve(model)
    drg_spli = global_solve(model, interp_type='spline')
#    drs = [dr1,dr2,dr3]
    drs = [dr1,dr2, drg_smol, drg_spli]
#    print dr2.S_bar

    print bounds

    errors_sup = [omega_sup(dr, model, bounds, [50,50]) for dr in drs]

    print errors_sup