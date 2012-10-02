from __future__ import print_function

from dolo import *
import numpy
from numpy import *

from dolo.compiler.global_solution import stochastic_residuals_2, stochastic_residuals, time_iteration

def global_solve(model, bounds=None, initial_dr=None, interp_type='smolyak', pert_order=2, T=200, n_s=2, N_e=40,
                 integration='gauss-hermite', integration_orders=[], maxit=500, numdiff=True, polish=True, tol=1e-8,
                 compiler='numpy', memory_hungry=True, smolyak_order=3, interp_orders=None, test_solution=False,
                 verbose=False, serial_grid=True):

    def vprint(t):
        if verbose:
            print(t)

    [y,x,parms] = model.read_calibration()
    sigma = model.read_covariances()
    
    if initial_dr == None:
        initial_dr = approximate_controls(model, order=pert_order )
        
    if bounds is not None:
        pass

    elif 'approximation' in model['original_data']:

        vprint('Using bounds specified by model')

        # this should be moved to the compiler
        ssmin =  model['original_data']['approximation']['bounds']['smin']
        ssmax =  model['original_data']['approximation']['bounds']['smax']
        ssmin = [model.eval_string( str(e) ) for e in ssmin]
        ssmax = [model.eval_string( str(e) ) for e in ssmax]
        ssmin = [model.eval_string( str(e) ) for e in ssmin]
        ssmax = [model.eval_string( str(e) ) for e in ssmax]

        [y,x,p] = model.read_calibration()
        d = { v: y[i] for i,v in enumerate(model.variables )}
        d.update( { v: p[i] for i,v in enumerate(model.parameters )} )

        smin = [expr.subs( d ) for expr in ssmin ]
        smax = [expr.subs( d ) for expr in ssmax ]

        bounds = numpy.row_stack([smin,smax])
        bounds = numpy.array(bounds,dtype=float)

    else:

        vprint('Using bounds given by second order solution.')

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
        sg = SmolyakGrid( bounds[0,:], bounds[1,:], smolyak_order )
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

    from dolo.compiler.global_solution import stochastic_residuals_2, stochastic_residuals_3

    from dolo.compiler.compiler_global import CModel
    cm = CModel(model,  substitute_auxiliary=True, solve_systems=True, compiler=compiler)
    gc = cm.as_type('fg')

    if integration == 'optimal_quantization':
        from dolo.numeric.quantization import quantization_nodes
        # number of shocks
        [epsilons,weights] = quantization_nodes(N_e, sigma)
    elif integration == 'gauss-hermite':
        from dolo.numeric.quadrature import gauss_hermite_nodes
        if not integration_orders:
            integration_orders = [3]*sigma.shape[0]
        [epsilons, weights] = gauss_hermite_nodes( integration_orders, sigma )

    from dolo.numeric.global_solution import time_iteration

    vprint('Starting time iteration')
    dr = time_iteration(sg.grid, sg, xinit, gc.f, gc.g, parms, epsilons, weights, maxit=maxit, tol=tol, nmaxit=50, numdiff=numdiff, verbose=verbose, serial_grid=serial_grid)
    
    if polish and interp_type=='smolyak' : # this works with smolyak only
        vprint('\nStarting global optimization')

        import time
        t1 = time.time()

        if not serial_grid:
            lb = gc.x_bounds[0](sg.grid,parms)
            ub = gc.x_bounds[1](sg.grid,parms)
        else:
            lb = None
            ub = None
        
        from dolo.numeric.solver import solver
        xinit = dr(dr.grid)
        dr.set_values(xinit)
        shape = xinit.shape

        if not memory_hungry:
            fobj = lambda t: stochastic_residuals_3(dr.grid, t, dr, gc.f, gc.g, parms, epsilons, weights, shape, no_deriv=True)
            dfobj = lambda t: stochastic_residuals_3(dr.grid, t, dr, gc.f, gc.g, parms, epsilons, weights, shape)[1]
        else:
            fobj = lambda t: stochastic_residuals_2(dr.grid, t, dr, gc.f, gc.g, parms, epsilons, weights, shape, no_deriv=True)
            dfobj = lambda t: stochastic_residuals_2(dr.grid, t, dr, gc.f, gc.g, parms, epsilons, weights, shape)[1]

        x = solver(fobj, xinit.flatten(), lb=lb, ub=ub, jac=dfobj, verbose=verbose, method='lmmcp')
        x = x.reshape(shape)
        dr.set_values(x) # just in case

        t2 = time.time()
        vprint('Finished in {} s'.format(t2-t1))

    if test_solution:
        res = stochastic_residuals_2(dr.grid, dr.theta , dr, gc.f, gc.g, parms, epsilons, weights, shape, no_deriv=True)
        if numpy.isfinite(res.flatten()).sum() > 0:
            raise( Exception('Non finite value in residuals.'))

    return dr