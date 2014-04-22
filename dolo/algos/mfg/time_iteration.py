import numpy

def residuals(f, g, s, x, dr, P, Q, parms):

    N = s.shape[0]
    n_s = s.shape[1]
    n_x = s.shape[1]

    n_ms = P.shape[0]   # number of markov states
    n_mv = P.shape[1] # number of markov variable

    res = numpy.zeros_like(x)



    for i_ms in range(n_ms):
        # solving on grid for markov index i_ms
        # m = P[i_ms,:][None,:]
        m = numpy.tile(P[i_ms,:],(N,1))
        xm = x[i_ms,:,:]

        for I_ms in range(n_ms):

            # M = P[I_ms,:][None,:]
            M = numpy.tile(P[I_ms,:], (N,1))
            prob = Q[i_ms, I_ms]

            S = g(m, s, xm, M, parms)
            XM = dr(I_ms, S)

            rr = f(m,s,xm,M,S,XM,parms)

            res[i_ms,:,:] += prob*rr

    return res







def solve_mfg_model(model, maxit=1000):

    assert(model.model_type == 'mfga')

    [P, Q] = model.markov_chain

    n_ms = P.shape[0]   # number of markov states
    n_mv = P.shape[1] # number of markov variables

    print(model.calibration['states'])
    x0 = model.calibration['controls']
    parms = model.calibration['parameters']
    n_x = len(x0)
    n_s = len(model.symbols['states'])

    approx = model.options['approximation_space']
    a = approx['a']
    b = approx['b']
    orders = approx['orders']

    from dolo.numeric.decision_rules_markov import MarkovDecisionRule

    mdr = MarkovDecisionRule(n_ms, a, b, orders)

    grid = mdr.grid
    N = grid.shape[0]

    controls = numpy.zeros((n_ms, N, n_x))
    controls[:,:,:] = x0[None,None,:]

    ff = model.functions['arbitrage']
    gg = model.functions['transition']
    aa = model.functions['auxiliary']

    f = lambda m,s,x,M,S,X,p: ff(m,s,x,aa(m,s,x,p),M,S,X,aa(M,S,X,p),p)
    g = lambda m,s,x,M,p: gg(m,s,x,aa(m,s,x,p),M,p)

    # mdr.set_values(controls)

    sh_c = controls.shape

    controls_0 = controls.reshape( (-1,n_x) )

    fn = lambda x: residuals(f, g, grid, x.reshape(sh_c), mdr, P, Q, parms).reshape((-1,n_x))

    from dolo.numeric.optimize.newton import newton, SerialDifferentiableFunction
    dfn = SerialDifferentiableFunction(fn)

    err = 10
    tol = 1e-8
    inner_maxit = 10
    it = 0

    while err>tol and it<maxit:

        it += 1

        mdr.set_values(controls_0.reshape(sh_c))

        [controls, nit] = newton(dfn, controls_0, verbose=False, maxit=inner_maxit)

        err = abs(controls-controls_0).max()

        controls_0 = controls

        print((it,err,nit))

    controls_0 = controls.reshape(sh_c)

    return None






if __name__ == '__main__':

    from dolo import *
    model = yaml_import("examples/global_models/rbc_mfga.yaml")

    sol = solve_mfg_model(model)


