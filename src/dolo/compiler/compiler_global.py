import numpy as np

from dolo.numeric.perturbations_to_states import simple_global_representation
from dolo.compiler.compiling import compile_function_2

def model_functions(model,substitute_auxiliary=False):
    sgm = simple_global_representation(model,substitute_auxiliary=substitute_auxiliary)

    controls = sgm['controls']
    states = sgm['states']
    parameters = sgm['parameters']
    shocks = sgm['shocks']

    f_eqs =  sgm['f_eqs']
    g_eqs =  sgm['g_eqs']

    controls_f = [c(1) for c in controls]
    states_f = [c(1) for c in states]
    controls_p = [c(-1) for c in controls]
    states_p = [c(-1) for c in states]
    shocks_f = [c(1) for c in shocks]


    args_g =  [states_p, controls_p, shocks]
    args_f =  [states, controls, states_f, controls_f, shocks_f]

    g = compile_function_2(g_eqs, args_g, ['s','x','e'], parameters, 'g' )
    f = compile_function_2(f_eqs, args_f, ['s','x','snext','xnext','e'], parameters, 'f' )

    return [f,g]

class GlobalCompiler:
    def __init__(self,model,substitute_auxiliary=False):
        self.model = model

        [f,g] = model_functions(model,substitute_auxiliary=substitute_auxiliary)
        self.f = f
        self.g = g



def deterministic_residuals(s, x, interp, f, g, parms):
    n_x = x.shape[0]
    n_g = x.shape[1]
    interp.fit_values(x)
    dummy_epsilons = np.zeros((n_x,n_g))
    [snext] = g(s,x,dummy_epsilons,parms)[:1]
    [xnext] = interp.interpolate(snext)[:1]
    [val] = f(s,x,snext,xnext,dummy_epsilons,parms)[:1]
    return val


def stochastic_residuals(s, x, dr, f, g, parms, epsilons, weights):
    n_draws = epsilons.shape[1]
    [n_x,n_g] = x.shape
    dr.fit_values(x)
    ss = np.tile(s, (1,n_draws))
    xx = np.tile(x, (1,n_draws))
    ee = np.repeat(epsilons, n_g , axis=1)
    [ssnext] = g(ss,xx,ee,parms)[:1]
    [xxnext] = dr.interpolate(ssnext)[:1]
    [val] = f(ss,xx,ssnext,xxnext,ee,parms)[:1]

    res = np.zeros( (n_x,n_g) )
    for i in range(n_draws):
        res += weights[i] * val[:,n_g*i:n_g*(i+1)]
    return res


def stochastic_residuals_2(s, theta, dr, f, g, parms, epsilons, weights, shape, no_deriv=False):

    n_t = len(theta)
    dr.theta = theta.copy().reshape(shape)
    #    [x, x_s, x_theta] = dr.interpolate(s, with_theta_deriv=True)
    n_draws = epsilons.shape[1]
    [n_s,n_g] = s.shape
    ss = np.tile(s, (1,n_draws))
    [xx, xx_s, xx_theta] = dr.interpolate(ss, with_theta_deriv=True) # should repeat theta instead
    n_x = xx.shape[0]
#    xx = np.tile(x, (1,n_draws))
    ee = np.repeat(epsilons, n_g , axis=1)
    [SS, SS_ss, SS_xx, SS_ee] = g(ss, xx, ee, parms)
    [XX, XX_SS, XX_t] = dr.interpolate(SS, with_theta_deriv=True)
    [F, F_ss, F_xx, F_SS, F_XX, F_ee] = f(ss, xx, SS, XX, ee, parms)


    res = np.zeros( (n_x,n_g) )
    for i in range(n_draws):
        res += weights[i] * F[:,n_g*i:n_g*(i+1)]

    if no_deriv:
        return res

    from dolo.numeric.serial_operations import strange_tensor_multiplication as stm
    SS_theta = stm( SS_xx, xx_theta)
    XX_theta = stm( XX_SS, SS_theta) + XX_t
    dF = stm(F_xx, xx_theta) + stm( F_SS, SS_theta) + stm( F_XX , XX_theta)
    dres = np.zeros( (n_x,n_t,n_g))
    for i in range(n_draws):
        dres += weights[i] * dF[:,:,n_g*i:n_g*(i+1)]
    return [res,dres.swapaxes(1,2)]

def step_residual(s, x, dr, f, g, parms, epsilons, weights, with_derivatives=True):
    n_draws = epsilons.shape[1]
    [n_x,n_g] = x.shape
    from dolo.numeric.serial_operations import strange_tensor_multiplication as stm
    ss = np.tile(s, (1,n_draws))
    xx = np.tile(x, (1,n_draws))
    ee = np.repeat(epsilons, n_g , axis=1)
    if with_derivatives:
        [ssnext, g_ss, g_xx] = g(ss,xx,ee,parms)[:3]
        [xxnext, xxold_ss] = dr.interpolate(ssnext)[:2]
        [val, f_ss, f_xx, f_ssnext, f_xxnext] = f(ss,xx,ssnext,xxnext,ee,parms)[:5]
        dval = f_xx + stm(f_ssnext, g_xx) + stm(f_xxnext, stm(xxold_ss, g_xx))

        res = np.zeros( (n_x,n_g) )
        for i in range(n_draws):
            res += weights[i] * val[:,n_g*i:n_g*(i+1)]

        dres = np.zeros( (n_x,n_x,n_g) )
        for i in range(n_draws):
            dres += weights[i] * dval[:,:,n_g*i:n_g*(i+1)]

        dval = np.zeros( (n_x,n_g,n_x,n_g))
        for i in range(n_g):
            dval[:,i,:,i] = dres[:,:,i]

        return [res, dval]
    else:
        [ssnext] = g(ss,xx,ee,parms)[:1]
        [xxnext] = dr.interpolate(ssnext)[:1]
        [val] = f(ss,xx,ssnext,xxnext,ee,parms)[:1]

        res = np.zeros( (n_x,n_g) )
        for i in range(n_draws):
            res += weights[i] * val[:,n_g*i:n_g*(i+1)]

        return [res]
#f = model_fun['f']
#g = model_fun['g']
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

    squared_errors = np.power(errors,2)
    std_errors = np.sqrt( np.sum(squared_errors,axis=0) )
    
    return std_errors


def time_iteration(grid, interp, xinit, f, g, parms, epsilons, weights, options={}, verbose=True, maxit=500, debug=None):

    from dolo.numeric.solver import solver

    fun = lambda x: step_residual(grid, x, interp, f, g, parms, epsilons, weights)[0]
    dfun = lambda x: step_residual(grid, x, interp, f, g, parms, epsilons, weights)[1]

    if debug:
        maxit = debug

    #
    tol = 1e-8
    ##
    import time
    t1 = time.time()
    err = 1
    x0 = xinit
    it = 0
    if debug:
        x_values = []
    while err > tol and it < maxit:
        t_start = time.time()
        it +=1
        interp.fit_values(x0)
    #    x = solver(fun, x0, method='lmmcp', jac='default', verbose=False, options=options)
	verbit = True if verbose=='full' else False
        x = solver(fun, x0, method='lmmcp', jac=dfun, verbose=verbit, options=options)
        res = abs(fun(x)).max()
        err = abs(x-x0).max()
        t_finish = time.time()
        elapsed = t_finish - t_start
        if verbose:
            print("iteration {} : {} : {}".format(it,err,elapsed))
        x0 = x0 + (x-x0)
        if debug:
            x_values.append(x0)
    #

    if it == maxit:
        if not debug:
            raise("Maximum number of iterations reached")
        else:
            return x_values


    t2 = time.time()
    print('Elapsed: {}'.format(t2 - t1))

    return interp
