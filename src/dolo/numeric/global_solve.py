from dolo import *
import numpy
from numpy import *

def global_solve(model, bounds=None, initial_dr=None, smolyak_order=3, T=200, n_s=2, N_e=40, maxit=500, polish=True, memory_hungry=True):
    [y,x,parms] = model.read_calibration()
    sigma = model.read_covariances()
    
    if bounds == None or initial_dr == None:
        pert_dr = approximate_controls(model, order=2, substitute_auxiliary=True, solve_systems=True)
        
    if bounds == None:
        from dolo.numeric.timeseries import asymptotic_variance
        
        Q = asymptotic_variance(pert_dr.A.real, pert_dr.B.real, pert_dr.sigma, T=T)
        
        devs = numpy.sqrt( numpy.diag(Q) )
        bounds  = numpy.row_stack([
                                   pert_dr.S_bar - devs * n_s,
                                   pert_dr.S_bar + devs * n_s,
                                   ])
        
    from dolo.numeric.smolyak import SmolyakGrid
    sg = SmolyakGrid( bounds, smolyak_order )
    
    if initial_dr == None:
        xinit = pert_dr(sg.grid) # initial value on the grid
        xinit = xinit.real
        xinit[:2,:] = numpy.maximum(xinit[:2,:],0.01)
    else:
        xinit = initial_dr(sg.grid)
            
    from dolo.compiler.compiler_global import GlobalCompiler, time_iteration, stochastic_residuals_2, stochastic_residuals_3
    gc = GlobalCompiler(model, substitute_auxiliary=True, solve_systems=True)
    
    from dolo.numeric.quantization import quantization_weights
    # number of shocks
    [weights,epsilons] = quantization_weights(N_e, sigma)
    
    dr = time_iteration(sg.grid, sg, xinit, gc.f, gc.g, parms, epsilons, weights, maxit=maxit, nmaxit=50 )
    
    if polish:
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
    
