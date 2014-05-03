import numpy
import numpy as np


def time_iteration(model,  bounds=None, verbose=False,
                 initial_dr=None, pert_order=1,
                 interp_type='smolyak', smolyak_order=3, interp_orders=None,
                 maxit=500, numdiff=True, polish=True, tol=1e-8,
                 integration='gauss-hermite', integration_orders=[],
                 compiler='numpy', memory_hungry=True, method='newton',
                 T=200, n_s=2, N_e=40 ):

    def vprint(t):
        if verbose:
            print(t)

    # model = modelodel.model
    # model = modelodel

    parms = model.calibration['parameters']
    sigma = model.covariances # calibration['covariances']

    if initial_dr == None:
        if pert_order==1:
            from dolo.algos.perturbations import approximate_controls
            initial_dr = approximate_controls(model)

        if pert_order>1:
            raise Exception("Perturbation order > 1 not supported (yet).")

        if interp_type == 'perturbations':
            return initial_dr

    if bounds is not None:
        pass

    elif False and model.options and 'approximation_space' in model.options:

        vprint('Using bounds specified by model')

        approx = model.options['approximation_space']
        a = approx['a']
        b = approx['b']

        bounds = numpy.row_stack([a, b])
        bounds = numpy.array(bounds, dtype=float)

        print(bounds)

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
        print(bounds)

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



    if model.model_type == 'fga':
        ff = model.functions['arbitrage']
        gg = model.functions['transition']
        aa = model.functions['auxiliary']
        g = lambda s,x,e,p : gg(s,x,aa(s,x,p),e,p)
        f = lambda s,x,e,S,X,p : ff(s,x,aa(s,x,p),S,X,aa(S,X,p),p)
    elif model.model_type == 'fg':
        g = model.functions['transition']
        ff = model.functions['arbitrage']
        f = lambda s,x,e,S,X,p : ff(s,x,S,X,p)
    else:
        f = model.functions['arbitrage']
        g = model.functions['transition']


    dr = time_iteration_helper(grid, dr, xinit, f, g, parms, epsilons, weights, maxit=maxit,
                        tol=tol, nmaxit=50, numdiff=numdiff, verbose=verbose, method=method)


    return dr


def time_iteration_helper(grid, dr, xinit, f, g, parms, epsilons, weights, x_bounds=None, tol = 1e-8, serial_grid=True, numdiff=True, verbose=True, method='ncpsolve', maxit=500, nmaxit=5, backsteps=10, hook=None, options={}):

    import time


    if numdiff == True:
        fun = lambda x: step_residual(grid, x, dr, f, g, parms, epsilons, weights, x_bounds=x_bounds, with_derivatives=False)
    else:
        fun = lambda x: step_residual(grid, x, dr, f, g, parms, epsilons, weights, x_bounds=x_bounds)

    ##
    t1 = time.time()
    err = 1
    x0 = xinit
    it = 0

    verbit = True if verbose=='full' else False

    if x_bounds:
        lb = x_bounds[0](grid,parms)
        ub = x_bounds[1](grid,parms)
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
        sdfun = SerialDifferentiableFunction(fun)
        [x,nit] = serial_newton(sdfun,x0)

        err = abs(x-x0).max()
        err_SA = err/err_0
        err_0 = err

        t_finish = time.time()
        elapsed = t_finish - t_start
        if verbose:
            print('|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} | {4:3} |'.format( it, err, err_SA, elapsed, nit  ))
            print(stars)

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


def deterministic_residuals(s, x, interp, f, g, sigma, parms):
    """
    :param s:
    :param x:
    :param interp:
    :param f:
    :param g:
    :param sigma:
    :param parms:
    :return:
    """

    n_x = x.shape[0]
    n_g = x.shape[1]
    n_e = sigma.shape[0]
    interp.set_values(x)
    dummy_epsilons = np.zeros((n_e,n_g))
    snext = g(s,x,dummy_epsilons,parms)
    xnext = interp.interpolate(snext)
    val = f(s,x,dummy_epsilons,snext,xnext,parms)
    return val


def stochastic_residuals(s, x, dr, f, g, parms, epsilons, weights):
    n_draws = epsilons.shape[1]
    [n_x,n_g] = x.shape
    dr.set_values(x)
    ss = np.tile(s, (1,n_draws))
    xx = np.tile(x, (1,n_draws))
    ee = np.repeat(epsilons, n_g , axis=1)
    ssnext = g(ss,xx,ee,parms)
    xxnext = dr.interpolate(ssnext)
    val = f(ss,xx,ee,ssnext,xxnext,parms)

    res = np.zeros( (n_x,n_g) )
    for i in range(n_draws):
        res += weights[i] * val[:,n_g*i:n_g*(i+1)]
    return res


def stochastic_residuals_2(s, x, dr, f, g, parms, epsilons, weights, shape, deriv=False):
    #memory_hungry version

    n_t = np.prod(x.shape)

    dr.set_values(x)  # shortcut : let's call x theta for now

    #    [x, x_s, x_theta] = dr.interpolate(s, with_theta_deriv=True)
    n_draws = epsilons.shape[1]
    [n_s,n_g] = s.shape
    ss = np.tile(s, (1,n_draws))
    [xx, xx_s, junk, xx_theta] = dr.interpolate(ss, with_derivative=True,  with_theta_deriv=True, with_X_deriv=True) # should repeat theta instead
    n_x = xx.shape[0]
    #    xx = np.tile(x, (1,n_draws))
    ee = np.repeat(epsilons, n_g , axis=1)
    [SS, SS_ss, SS_xx, SS_ee] = g(ss, xx, ee, parms, diff=True)
    [XX, XX_SS, junk, XX_t] = dr.interpolate(SS, with_derivative=True, with_theta_deriv=True, with_X_deriv=True)
    [F, F_ss, F_xx, F_ee, F_SS, F_XX] = f(ss, xx, ee, SS, XX, parms, diff=True)


    res = np.zeros( (n_x,n_g) )
    for i in range(n_draws):
        res += weights[i] * F[:,n_g*i:n_g*(i+1)]

    if not deriv:
        return res

    from dolo.numeric.serial_operations import serial_multiplication as stm
    SS_theta = stm( SS_xx, xx_theta)
    XX_theta = stm( XX_SS, SS_theta) + XX_t
    dF = stm(F_xx, xx_theta) + stm( F_SS, SS_theta) + stm( F_XX , XX_theta)
    dres = np.zeros( (n_x,n_t,n_g))
    for i in range(n_draws):
        dres += weights[i] * dF[:,:,n_g*i:n_g*(i+1)]
    return [res,dres.swapaxes(1,2)]

def stochastic_residuals_3(s, x, dr, f, g, parms, epsilons, weights, deriv=False):

    import numpy


    n_t = np.prod(x.shape)

    dr.set_values(x)  # shortcut : let's call x theta for now

    n_draws = epsilons.shape[1]
    [n_s,n_g] = s.shape

    #    xx = np.tile(x, (1,n_draws))
    #ee = np.repeat(epsilons, n_g , axis=1)
    #
    from dolo.numeric.serial_operations import serial_multiplication as stm
    #

    if not deriv:
        x = dr.interpolate(s, with_theta_deriv=False, with_derivative=False) # should repeat theta instead
        n_x = x.shape[0]
        res = np.zeros( (n_x,n_g) )
        for i in range(n_draws):
            tt = [epsilons[:,i:i+1]]*n_g
            e = numpy.column_stack(tt)
            S = g(s, x, e, parms)
            X = dr(S)
            F = f(s, x, e, S, X, parms)
            res += weights[i] * F
        return res
    else:
        [x, x_s, junk, x_theta] = dr.interpolate(s, with_derivative=True, with_theta_deriv=True, with_X_deriv=True) # should repeat theta instead
        n_x = x.shape[0]
        res = np.zeros( (n_x,n_g) )
        dres = np.zeros( (n_x,n_t,n_x,n_t))
        for i in range(n_draws):
            tt = [epsilons[:,i:i+1]]*n_g
            e = numpy.column_stack(tt)
            [S, S_s, S_x, S_e] = g(s, x, e, parms, diff=True)
            [X, X_S, junk, X_t] = dr.interpolate(S, with_derivative=True, with_theta_deriv=True, with_X_deriv=True)
            [F, F_s, F_x, F_e, F_S, F_X] = f(s, x, e, S, X, parms, diff=True)
            res += weights[i] * F
            S_theta = stm(S_x, x_theta)
            X_theta = stm(X_S, S_theta) + X_t
            dF = stm(F_x, x_theta) + stm( F_S, S_theta) + stm( F_X , X_theta)
            dres += weights[i] * dF
        return [res,dres.swapaxes(1,2)]


def step_residual_fg(s, x, dr, f, g, parms, epsilons, weights, x_bounds=None, serial_grid=True, with_derivatives=True):

    n_draws = epsilons.shape[1]
    [n_x,n_g] = x.shape
    ss = np.tile(s, (1,n_draws))
    xx = np.tile(x, (1,n_draws))
    ee = np.repeat(epsilons, n_g , axis=1)

    if with_derivatives:
        raise Exception('not implementeds')
    else:
        ssnext = g(ss,xx,ee,parms)

        if False:
            smin = s.min(axis=1)[:,None]
            smax = s.max(axis=1)[:,None]
            ssnext = np.maximum(np.minimum(smax,ssnext),smin)

        xxnext = dr.interpolate(ssnext)

#        if x_bounds:
#            lb = x_bounds[0](ssnext, parms)
#            ub = x_bounds[1](ssnext, parms)
#            xxnext = np.maximum(np.minimum(ub,xxnext),lb)

        val = f(ss,xx,ssnext,xxnext,parms)

        res = np.zeros( (n_x,n_g) )
        for i in range(n_draws):
            res += weights[i] * val[:,n_g*i:n_g*(i+1)]

        return res

def step_residual(s, x, dr, f, g, parms, epsilons, weights, x_bounds=None, serial_grid=True, with_derivatives=True):


    # TODO: transpose
    n_draws = epsilons.shape[0]
    [N,n_x] = x.shape
    ss = np.tile(s, (n_draws,1))
    xx = np.tile(x, (n_draws,1))
    ee = np.repeat(epsilons, N , axis=0)

    if with_derivatives:
        raise Exception('not implemented')

    else:

        ssnext = g(ss,xx,ee,parms)
        xxnext = dr.interpolate(ssnext)

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

