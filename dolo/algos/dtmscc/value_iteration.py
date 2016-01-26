import numpy

def evaluate_policy(model, mdr, tol=1e-8,  maxit=2000, orders=None, verbose=True, initial_guess=None, hook=None, integration_orders=None):

    assert(model.model_type == 'dtmscc')

    [P, Q] = model.markov_chain

    n_ms = P.shape[0]   # number of markov states
    n_mv = P.shape[1] # number of markov variables

    x0 = model.calibration['controls']
    v0 = model.calibration['values']
    parms = model.calibration['parameters']
    n_x = len(x0)
    n_v = len(v0)
    n_s = len(model.symbols['states'])

    approx = model.options['approximation_space']
    a = approx['a']
    b = approx['b']

    if orders is None:
        orders = approx['orders']
    else:
        orders = numpy.array(orders,dtype=int)

    from dolo.numeric.decision_rules_markov import MarkovDecisionRule
    mdrv = MarkovDecisionRule(n_ms, a, b, orders) # values

    grid = mdrv.grid
    N = grid.shape[0]

    controls = numpy.zeros((n_ms, N, n_x))
    for i_m in range(n_ms):
        controls[i_m,:,:] = mdr(i_m,grid) #x0[None,:]

    values_0 = numpy.zeros((n_ms, N, n_v))
    if initial_guess is None:
        for i_m in range(n_ms):
            values_0[i_m,:,:] = v0[None,:]
    else:
        for i_m in range(n_ms):
            values_0[i_m,:,:] = initial_guess(i_m, grid)


    ff = model.functions['arbitrage']
    gg = model.functions['transition']
    aa = model.functions['auxiliary']
    vaval = model.functions['value']


    f = lambda m,s,x,M,S,X,p: ff(m,s,x,M,S,X,p)
    g = lambda m,s,x,M,p: gg(m,s,x,M,p)
    def val(m,s,x,v,M,S,X,V,p):
        return vaval(m,s,x,v,M,S,X,V,p)
    # val = lambda m,s,x,v,M,S,X,V,p: vaval(m,s,x,aa(m,s,x,p),v,M,S,X,aa(M,S,X,p),V,p)


    sh_v = values_0.shape

    err = 10
    inner_maxit = 50
    it = 0


    if verbose:
        headline = '|{0:^4} | {1:10} | {2:8} | {3:8} |'.format( 'N',' Error', 'Gain','Time')
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

        mdrv.set_values(values_0.reshape(sh_v))

        values = update_value(val, g, grid, controls, values_0, mdr, mdrv, P, Q, parms).reshape((-1,n_v))

        err = abs(values.reshape(sh_v)-values_0).max()

        err_SA = err/err_0
        err_0 = err

        values_0 = values.reshape(sh_v)

        t_finish = time.time()
        elapsed = t_finish - t_start

        if verbose:
            print('|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} |'.format( it, err, err_SA, elapsed  ))

    # values_0 = values.reshape(sh_v)

    t2 = time.time()

    if verbose:
        print(stars)
        print("Elapsed: {} seconds.".format(t2-t1))
        print(stars)

    return mdrv


def update_value(val, g, s, x, v, dr, drv, P, Q, parms):

    N = s.shape[0]
    n_s = s.shape[1]

    n_ms = P.shape[0]   # number of markov states

    res = numpy.zeros_like(v)

    for i_ms in range(n_ms):

        m = P[i_ms,:][None,:].repeat(N,axis=0)

        xm = x[i_ms,:,:]
        vm = v[i_ms,:,:]

        for I_ms in range(n_ms):

            # M = P[I_ms,:][None,:]
            M = P[I_ms,:][None,:].repeat(N,axis=0)
            prob = Q[i_ms, I_ms]

            S = g(m, s, xm, M, parms)
            XM = dr(I_ms, S)
            VM = drv(I_ms, S)

            rr = val(m,s,xm,vm,M,S,XM,VM,parms)

            res[i_ms,:,:] += prob*rr

    return res
