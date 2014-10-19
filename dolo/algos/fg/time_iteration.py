import numpy
import numpy as np


def time_iteration(model,  bounds=None, verbose=False, initial_dr=None,
                 pert_order=1, with_complementarities=True,
                 interp_type='smolyak', smolyak_order=3, interp_orders=None,
                 maxit=500, tol=1e-8,
                 integration='gauss-hermite', integration_orders=None,
                 T=200, n_s=3, hook=None):

    """Finds a global solution for ``model`` using backward time-iteration.

    Parameters:
    -----------

    model: NumericModel
        "fg" or "fga" model to be solved

    bounds: ndarray
        boundaries for approximations. First row contains minimum values. Second row contains maximum values.

    verbose: boolean
        if True, display iterations

    initial_dr: decision rule
        initial guess for the decision rule

    pert_order: {1}
        if no initial guess is supplied, the perturbation solution at order ``pert_order`` is used as initial guess

    with_complementarities: boolean (True)
        if False, complementarity conditions are ignored

    interp_type: {`smolyak`, `spline`}
        type of interpolation to use for future controls

    smolyak_orders: int
        parameter ``l`` for Smolyak interpolation

    interp_orders: 1d array-like
        list of integers specifying the number of nods in each dimension if ``interp_type="spline" ``


    Returns:
    --------
    decision rule object (SmolyakGrid or MultivariateSplines)
    """

    def vprint(t):
        if verbose:
            print(t)

    parms = model.calibration['parameters']
    sigma = model.covariances

    if initial_dr is None:
        if pert_order==1:
            from dolo.algos.fg.perturbations import approximate_controls
            initial_dr = approximate_controls(model)

        if pert_order>1:
            raise Exception("Perturbation order > 1 not supported (yet).")

        if interp_type == 'perturbations':
            return initial_dr

    if bounds is not None:
        pass

    elif model.options and 'approximation_space' in model.options:

        vprint('Using bounds specified by model')

        approx = model.options['approximation_space']
        a = approx['a']
        b = approx['b']

        bounds = numpy.row_stack([a, b])
        bounds = numpy.array(bounds, dtype=float)

    else:
        vprint('Using asymptotic bounds given by first order solution.')

        from dolo.numeric.timeseries import asymptotic_variance
        # this will work only if initial_dr is a Taylor expansion
        Q = asymptotic_variance(initial_dr.A.real, initial_dr.B.real, initial_dr.sigma, T=T)

        devs = numpy.sqrt(numpy.diag(Q))
        bounds = numpy.row_stack([
            initial_dr.S_bar - devs * n_s,
            initial_dr.S_bar + devs * n_s,
        ])

    if interp_orders is None:
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

    if integration == 'optimal_quantization':
        from dolo.numeric.discretization import quantization_nodes
        # number of shocks
        [epsilons, weights] = quantization_nodes(N_e, sigma)
    elif integration == 'gauss-hermite':
        from dolo.numeric.discretization import gauss_hermite_nodes
        if not integration_orders:
            integration_orders = [3] * sigma.shape[0]
        [epsilons, weights] = gauss_hermite_nodes(integration_orders, sigma)


    vprint('Starting time iteration')

    # TODO: transpose

    grid = dr.grid

    xinit = initial_dr(grid)
    xinit = xinit.real  # just in case...


    from dolo.algos.fg.convert import get_fg_functions

    f,g = get_fg_functions(model)


    import time

    fun = lambda x: step_residual(grid, x, dr, f, g, parms, epsilons, weights)

    ##
    t1 = time.time()
    err = 1
    x0 = xinit
    it = 0

    verbit = True if verbose=='full' else False

    if with_complementarities:
        lbfun = model.functions['arbitrage_lb']
        ubfun = model.functions['arbitrage_ub']
        lb = lbfun(grid, parms)
        ub = ubfun(grid, parms)
    else:
        lb = None
        ub = None

    if verbose:
        headline = '|{0:^4} | {1:10} | {2:8} | {3:8} | {4:3} |'.format( 'N',' Error', 'Gain','Time',  'nit' )
        stars = '-'*len(headline)
        print(stars)
        print(headline)
        print(stars)

    err_0 = 1

    while err > tol and it < maxit:
        t_start = time.time()
        it +=1

        dr.set_values(x0)

        from dolo.numeric.optimize.newton import serial_newton, SerialDifferentiableFunction
        from dolo.numeric.optimize.ncpsolve import ncpsolve
        sdfun = SerialDifferentiableFunction(fun)

        if with_complementarities:
            [x,nit] = ncpsolve(sdfun, lb, ub, x0, verbose=verbit)

        else:
            [x,nit] = serial_newton(sdfun, x0, verbose=verbit)

        err = abs(x-x0).max()
        err_SA = err/err_0
        err_0 = err

        t_finish = time.time()
        elapsed = t_finish - t_start
        if verbose:
            print('|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} | {4:3} |'.format( it, err, err_SA, elapsed, nit  ))

        x0 = x0 + (x-x0)
        if hook:
            hook(dr,it,err)
        if False in np.isfinite(x0):
            print('iteration {} failed : non finite value')
            return [x0, x]
    
    if it == maxit:
        import warnings
        warnings.warn(UserWarning("Maximum number of iterations reached"))


    t2 = time.time()
    if verbose:
        print(stars)
        print('Elapsed: {} seconds.'.format(t2 - t1))
        print(stars)

    return dr

def step_residual(s, x, dr, f, g, parms, epsilons, weights):


    # TODO: transpose
    n_draws = epsilons.shape[0]
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

    squared_errors = np.power(errors,2)
    std_errors = np.sqrt( np.sum(squared_errors,axis=0)/len(squared_errors) )

    return std_errors
