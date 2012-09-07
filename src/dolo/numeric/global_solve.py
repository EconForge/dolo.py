from __future__ import print_function

from dolo import *
import numpy
from numpy import *

from dolo.compiler.global_solution import stochastic_residuals_2, stochastic_residuals, time_iteration

def global_solve(model, bounds=None, initial_dr=None, interp_type='smolyak', pert_order=2, T=200, n_s=2, N_e=40, integration='gauss-hermite', integration_orders=[], maxit=500, numdiff=True, polish=True, compiler=None, memory_hungry=True, smolyak_order=3, interp_orders=None, test_solution=False, verbose=False):

    def vprint(t):
        if verbose:
            print(t)

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
    elif interp_type == 'mlinear':
        from dolo.numeric.multilinear import MultilinearInterpolator
        sg = MultilinearInterpolator( bounds[0,:], bounds[1,:], interp_orders)

    
    xinit = initial_dr(sg.grid)
    xinit = xinit.real  # just in case...

    from dolo.compiler.compiler_global import GlobalCompiler
    from dolo.compiler.global_solution import stochastic_residuals_2, stochastic_residuals_3
    if compiler == 'theano':
        from dolo.compiler.cmodel_theano import CModel
        cm = CModel(model)
        gc = cm.as_type('fg')
    else:
        gc = GlobalCompiler(model, substitute_auxiliary=True, solve_systems=True)

    if integration == 'optimal_quantization':
        from dolo.numeric.quantization import quantization_nodes
        # number of shocks
        [epsilons,weights] = quantization_nodes(N_e, sigma)
    elif integration == 'gauss-hermite':
        from dolo.numeric.quadrature import gauss_hermite_nodes
        if not integration_orders:
            integration_orders = [3]*sigma.shape[0]
        [epsilons, weights] = gauss_hermite_nodes( integration_orders, sigma )

    from dolo.compiler.global_solution import time_iteration, stochastic_residuals_2, stochastic_residuals_3
    vprint('Starting time iteration')
    dr = time_iteration(sg.grid, sg, xinit, gc.f, gc.g, parms, epsilons, weights, maxit=maxit, nmaxit=50, numdiff=numdiff, verbose=verbose )
    
    if polish: # this will only work with smolyak
        vprint('\nStarting global optimization')
        from dolo.compiler.compiler_global import GlobalCompiler
        
        from dolo.numeric.solver import solver
        xinit = dr(dr.grid)
        dr.set_values(xinit)
        shape = dr.theta.shape
        theta_0 = dr.theta.copy().flatten()
        if not memory_hungry:
            fobj = lambda t: stochastic_residuals_3(dr.grid, t, dr, gc.f, gc.g, parms, epsilons, weights, shape, no_deriv=True)
            dfobj = lambda t: stochastic_residuals_3(dr.grid, t, dr, gc.f, gc.g, parms, epsilons, weights, shape)[1]
        else :
            fobj = lambda t: stochastic_residuals_2(dr.grid, t, dr, gc.f, gc.g, parms, epsilons, weights, shape, no_deriv=True)
            dfobj = lambda t: stochastic_residuals_2(dr.grid, t, dr, gc.f, gc.g, parms, epsilons, weights, shape)[1]
        
        theta = solver(fobj, theta_0, jac=dfobj, verbose=verbose)
        dr.theta = theta.reshape(shape)

    if test_solution:
        res = stochastic_residuals_2(dr.grid, dr.theta , dr, gc.f, gc.g, parms, epsilons, weights, shape, no_deriv=True)
        if numpy.isfinite(res.flatten()).sum() > 0:
            raise( Exception('Non finite value in residuals.'))

    return dr
    
