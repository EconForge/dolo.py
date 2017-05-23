import numpy
from dolo import dprint
from dolo.numeric.processes import DiscretizedIIDProcess
from dolo.numeric.decision_rule import DecisionRule
from dolo.numeric.grids import CartesianGrid

def residuals_simple(f, g, s, x, dr, dprocess, parms):

    N = s.shape[0]
    n_s = s.shape[1]

    res = numpy.zeros_like(x)

    for i_ms in range(dprocess.n_nodes()):

        # solving on grid for markov index i_ms
        m = numpy.tile(dprocess.node(i_ms),(N,1))
        xm = x[i_ms,:,:]

        for I_ms in range(dprocess.n_inodes(i_ms)):
            M = numpy.tile(dprocess.inode(i_ms, I_ms), (N,1))
            prob = dprocess.iweight(i_ms, I_ms)
            S = g(m, s, xm, M, parms)
            XM = dr.eval_ijs(i_ms, I_ms, S)
            rr = f(m,s,xm,M,S,XM,parms)
            res[i_ms,:,:] += prob*rr

    return res


def time_iteration(model, initial_guess=None, with_complementarities=True,
                        verbose=True, grid={},
                        maxit=1000, inner_maxit=10, tol=1e-6, hook=None) :

    '''
    Finds a global solution for ``model`` using backward time-iteration.

    This algorithm iterates on the residuals of the arbitrage equations

    Parameters
    ----------
    model : NumericModel
        "dtmscc" model to be solved
    verbose : boolean
        if True, display iterations
    initial_dr : decision rule
        initial guess for the decision rule
    with_complementarities : boolean (True)
        if False, complementarity conditions are ignored
    grid: grid options
    maxit: maximum number of iterations
    inner_maxit: maximum number of iteration for inner solver
    tol: tolerance criterium for successive approximations

    Returns
    -------
    decision rule :
        approximated solution
    '''

    from dolo import dprint
    def vprint(t):
        if verbose:
            print(t)

    process = model.exogenous
    dprocess = process.discretize()

    n_ms = dprocess.n_nodes() # number of exogenous states
    n_mv = dprocess.n_inodes(0) # this assume number of integration nodes is constant

    x0 = model.calibration['controls']
    parms = model.calibration['parameters']
    n_x = len(x0)
    n_s = len(model.symbols['states'])

    endo_grid = model.get_grid(**grid)

    interp_type = 'cubic'

    exo_grid = dprocess.grid

    mdr = DecisionRule(exo_grid, endo_grid)

    grid = mdr.endo_grid.nodes()
    N = grid.shape[0]

    controls_0 = numpy.zeros((n_ms, N, n_x))
    if initial_guess is None:
        controls_0[:, :, :] = x0[None,None,:]
    else:
        for i_m in range(n_ms):
            controls_0[i_m, :, :] = initial_guess(i_m, grid)

    f = model.functions['arbitrage']
    g = model.functions['transition']

    if 'controls_lb' in model.functions and with_complementarities==True:
        lb_fun = model.functions['controls_lb']
        ub_fun = model.functions['controls_ub']
        lb = numpy.zeros_like(controls_0)*numpy.nan
        ub = numpy.zeros_like(controls_0)*numpy.nan
        for i_m in range(n_ms):
            m = dprocess.node(i_m)[None,:]
            p = parms[None,:]
            m = numpy.repeat(m, N, axis=0)
            p = numpy.repeat(p, N, axis=0)

            lb[i_m,:,:] = lb_fun(m, grid, p)
            ub[i_m,:,:] = ub_fun(m, grid, p)

    else:
        with_complementarities = False

    sh_c = controls_0.shape

    controls_0 = controls_0.reshape( (-1,n_x) )

    from dolo.numeric.optimize.newton import newton, SerialDifferentiableFunction
    from dolo.numeric.optimize.ncpsolve import ncpsolve

    err = 10
    it = 0

    if with_complementarities:
        vprint("Solving WITH complementarities.")
        lb = lb.reshape((-1,n_x))
        ub = ub.reshape((-1,n_x))


    if verbose:
        headline = '|{0:^4} | {1:10} | {2:8} | {3:8} | {4:3} |'.format( 'N',' Error', 'Gain','Time',  'nit' )
        stars = '-'*len(headline)
        print(stars)
        print(headline)
        print(stars)

    import time
    t1 = time.time()

    err_0 = numpy.nan

    verbit = (verbose == 'full')

    while err>tol and it<maxit:

        it += 1

        t_start = time.time()

        mdr.set_values(controls_0.reshape(sh_c))

        fn = lambda x: residuals_simple(f, g, grid, x.reshape(sh_c), mdr, dprocess, parms).reshape((-1,n_x))
        dfn = SerialDifferentiableFunction(fn)

        res = fn(controls_0)

        if hook:
            hook()

        if with_complementarities:
            [controls,nit] = ncpsolve(dfn, lb, ub, controls_0, verbose=verbit, maxit=inner_maxit)
        else:
            [controls, nit] = newton(dfn, controls_0, verbose=verbit, maxit=inner_maxit)

        err = abs(controls-controls_0).max()

        err_SA = err/err_0
        err_0 = err

        controls_0 = controls

        t_finish = time.time()
        elapsed = t_finish - t_start

        if verbose:
            print('|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} | {4:3} |'.format( it, err, err_SA, elapsed, nit  ))

    controls_0 = controls.reshape(sh_c)

    t2 = time.time()

    if verbose:
        print(stars)
        print("Elapsed: {} seconds.".format(t2-t1))
        print(stars)

    return mdr


if __name__ == '__main__':

    from dolo import *
    model = yaml_import("../../../examples/models/compat/rbc_mfga.yaml")
    print(model.calibration['states'])
    print(model.calibration_dict)
    print(model.exogenous)


    initial_guess_symbolic = [
        'i = delta*k',
        'n = 0.33'
    ]

    from dolo.compiler.function_compiler_ast import compile_function_ast
    from dolo.compiler.function_compiler import standard_function

    arg_names = [
        ['exogenous',0,'m'],
        ['states',0,'s'],
        ['parameters',0,'p']
    ]

    fun = compile_function_ast(initial_guess_symbolic, model.symbols, arg_names,'initial_guess')
    ff = standard_function(fun, len(initial_guess_symbolic))

    sol = time_iteration(model, initial_guess=ff)
