from dolo import *
import numpy
from numpy import *

def global_solve(model, bounds=None, initial_dr=None, interp_type='smolyak', pert_order=2, T=200, n_s=2, N_e=40, maxit=500, polish=True, memory_hungry=True, smolyak_order=3, interp_orders=None):
    [y,x,parms] = model.read_calibration()
    sigma = model.read_covariances()
    
    if initial_dr == None:
        initial_dr = approximate_controls(model, order=pert_order, substitute_auxiliary=True, solve_systems=True)
        
    if bounds == None:
        from dolo.numeric.timeseries import asymptotic_variance
        # this will work only if initial_dr is a Taylor expansion
        Q = asymptotic_variance(initial_dr.A.real, initial_dr.B.real, initial_dr.sigma, T=T)
        
        devs = numpy.sqrt( numpy.diag(Q) )
        bounds  = numpy.row_stack([
                                   initial_dr.S_bar - devs * n_s,
                                   initial_dr.S_bar + devs * n_s,
                                   ])

    if interp_orders == None:
            interp_orders = [5]*bounds.shape[1]
    if interp_type == 'smolyak':
        from dolo.numeric.smolyak import SmolyakGrid
        sg = SmolyakGrid( bounds, smolyak_order )
    elif interp_type == 'spline':
        polish = False
        from dolo.numeric.interpolation import SplineInterpolation
        sg = SplineInterpolation( bounds[0,:], bounds[1,:], interp_orders )
    elif interp_type == 'linear':
        from dolo.numeric.interpolation import MLinInterpolation
        sg = MLinInterpolation( bounds[0,:], bounds[1,:], interp_orders )


    
    xinit = initial_dr(sg.grid)
    xinit = xinit.real  # just in case...

    from dolo.compiler.compiler_global import GlobalCompiler, time_iteration, stochastic_residuals_2, stochastic_residuals_3
    gc = GlobalCompiler(model, substitute_auxiliary=True, solve_systems=True)
    
    from dolo.numeric.quantization import quantization_weights
    # number of shocks
    [weights,epsilons] = quantization_weights(N_e, sigma)
    
    dr = time_iteration(sg.grid, sg, xinit, gc.f, gc.g, parms, epsilons, weights, maxit=maxit, nmaxit=50 )
    
    if polish: # this will only work with smolyak
        from dolo.compiler.compiler_global import GlobalCompiler, time_iteration, stochastic_residuals_2, stochastic_residuals_3
        
        from dolo.numeric.solver import solver
        xinit = dr(dr.grid)
        dr.fit_values(xinit)
        shape = dr.theta.shape
        theta_0 = dr.theta.copy().flatten()
        if memory_hungry:
            fobj = lambda t: stochastic_residuals_3(dr.grid, t, dr, gc.f, gc.g, parms, epsilons, weights, shape, no_deriv=True)[0]
            dfobj = lambda t: stochastic_residuals_3(dr.grid, t, dr, gc.f, gc.g, parms, epsilons, weights, shape)[1]
        else :
            fobj = lambda t: stochastic_residuals_2(dr.grid, t, dr, gc.f, gc.g, parms, epsilons, weights, shape, no_deriv=True)
            dfobj = lambda t: stochastic_residuals_2(dr.grid, t, dr, gc.f, gc.g, parms, epsilons, weights, shape)[1]
        
        theta = solver(fobj, theta_0, jac=dfobj, verbose=True)
        dr.theta = theta.reshape(shape)
    
    return dr
    
