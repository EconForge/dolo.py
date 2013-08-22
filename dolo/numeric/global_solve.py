from __future__ import print_function

import numpy

from dolo.numeric.perturbations_to_states import approximate_controls


def global_solve(cmodel,
                 bounds=None, verbose=False,
                 initial_dr=None, pert_order=1,
                 interp_type='smolyak', smolyak_order=3, interp_orders=None,
                 maxit=500, numdiff=True, polish=True, tol=1e-8,
                 integration='gauss-hermite', integration_orders=[],
                 compiler='numpy', memory_hungry=True, method='newton',
                 T=200, n_s=2, N_e=40 ):

    def vprint(t):
        if verbose:
            print(t)


    model = cmodel.model
    cm = cmodel

    parms = model.calibration['parameters']
    sigma = model.calibration['covariances']

    if initial_dr == None:
        initial_dr = approximate_controls(model, order=pert_order)
        if interp_type == 'perturbations':
            return initial_dr

    if bounds is not None:
        pass

    elif model.__data__ and 'approximation' in model.__data__:
        vprint('Using bounds specified by model')

        # this should be moved to the compiler
        ssmin = model.__data__['approximation']['bounds']['smin']
        ssmax = model.__data__['approximation']['bounds']['smax']
        ssmin = [model.eval_string(str(e)) for e in ssmin]
        ssmax = [model.eval_string(str(e)) for e in ssmax]
        ssmin = [model.eval_string(str(e)) for e in ssmin]
        ssmax = [model.eval_string(str(e)) for e in ssmax]

        d = model.calibration_dict

        smin = [expr.subs(d) for expr in ssmin]
        smax = [expr.subs(d) for expr in ssmax]

        smin = numpy.array(smin, dtype=numpy.float)
        smax = numpy.array(smax, dtype=numpy.float)

        bounds = numpy.row_stack([smin, smax])
        bounds = numpy.array(bounds, dtype=float)

    else:
        vprint('Using bounds given by second order solution.')

        from dolo.numeric.timeseries import asymptotic_variance
        # this will work only if initial_dr is a Taylor expansion
        Q = asymptotic_variance(initial_dr.A.real, initial_dr.B.real, initial_dr.sigma, T=T)

        devs = numpy.sqrt(numpy.diag(Q))
        bounds = numpy.row_stack([
            initial_dr.S_bar - devs * n_s,
            initial_dr.S_bar + devs * n_s,
        ])

    smin = bounds[0, :]
    smax = bounds[1, :]

    if interp_orders == None:
        interp_orders = [5] * bounds.shape[1]

    if interp_type == 'smolyak':
        from dolo.numeric.interpolation.smolyak import SmolyakGrid

        dr = SmolyakGrid(bounds[0, :], bounds[1, :], smolyak_order)
    elif interp_type == 'spline':
        from dolo.numeric.interpolation.splines import MultivariateSplines

        dr = MultivariateSplines(bounds[0, :], bounds[1, :], interp_orders)
    elif interp_type == 'multilinear':
        from dolo.numeric.interpolation.multilinear import MultilinearInterpolator

        dr = MultilinearInterpolator(bounds[0, :], bounds[1, :], interp_orders)
    elif interp_type == 'sparse_linear':
        from dolo.numeric.interpolation.interpolation import SparseLinear

        dr = SparseLinear(bounds[0, :], bounds[1, :], smolyak_order)
    elif interp_type == 'linear':
        from dolo.numeric.interpolation.interpolation import LinearTriangulation, TriangulatedDomain, RectangularDomain
        rec = RectangularDomain(smin, smax, interp_orders)
        domain = TriangulatedDomain(rec.grid)
        dr = LinearTriangulation(domain)

    if integration == 'optimal_quantization':
        from dolo.numeric.quantization import quantization_nodes
        # number of shocks
        [epsilons, weights] = quantization_nodes(N_e, sigma)
    elif integration == 'gauss-hermite':
        from dolo.numeric.quadrature import gauss_hermite_nodes

        if not integration_orders:
            integration_orders = [3] * sigma.shape[0]
        [epsilons, weights] = gauss_hermite_nodes(integration_orders, sigma)

    vprint('Starting time iteration')

    from dolo.numeric.global_solution import time_iteration
    from dolo.numeric.global_solution import stochastic_residuals_2, stochastic_residuals_3


    xinit = initial_dr(dr.grid)
    xinit = xinit.real  # just in case...


    f = cm.functions['arbitrage']
    g = cm.functions['transition']
    a = cm.functions['auxiliary']

#    cm.x_bounds = None

    dr = time_iteration(dr.grid, dr, xinit, f, g, a, parms, epsilons, weights, maxit=maxit,
                        tol=tol, nmaxit=50, numdiff=numdiff, verbose=verbose, method=method)


    polish = False

    if polish and interp_type == 'smolyak': # this works with smolyak only

        vprint('\nStarting global optimization')

        import time

        t1 = time.time()

        if cm.x_bounds is not None:
            lb = cm.x_bounds[0](dr.grid, parms)
            ub = cm.x_bounds[1](dr.grid, parms)
        else:
            lb = None
            ub = None

        xinit = dr(dr.grid)
        dr.set_values(xinit)
        shape = xinit.shape

        if not memory_hungry:
            fobj = lambda t: stochastic_residuals_3(dr.grid, t, dr, cm.f, cm.g, parms, epsilons, weights, shape, deriv=False)
            dfobj = lambda t: stochastic_residuals_3(dr.grid, t, dr, cm.f, cm.g, parms, epsilons, weights, shape, deriv=True)[1]
        else:
            fobj = lambda t: stochastic_residuals_2(dr.grid, t, dr, cm.f, cm.g, parms, epsilons, weights, shape, deriv=False)
            dfobj = lambda t: stochastic_residuals_2(dr.grid, t, dr, cm.f, cm.g, parms, epsilons, weights, shape, deriv=True)[1]

        from dolo.numeric.solver import solver

        x = solver(fobj, xinit, lb=lb, ub=ub, jac=dfobj, verbose=verbose, method='ncpsolve', serial_problem=False)

        dr.set_values(x) # just in case

        t2 = time.time()

        # test solution
        res = stochastic_residuals_2(dr.grid, x, dr, cm.f, cm.g, parms, epsilons, weights, shape, deriv=False)
        if numpy.isinf(res.flatten()).sum() > 0:
            raise ( Exception('Non finite values in residuals.'))

        vprint('Finished in {} s'.format(t2 - t1))

    return dr


# def plot_decision_rule():
