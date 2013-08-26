import numpy as np


def solver(fobj, x0, lb=None, ub=None, jac=None, method='lmmcp', infos=False, serial_problem=False, verbose=False, options={}):


    in_shape = x0.shape

    if serial_problem:
        ffobj = fobj
    else:
        ffobj = lambda x: fobj(x.reshape(in_shape)).flatten()


    # standardize jacobian
    if jac is not None:
        if not serial_problem:
            pp = np.prod(in_shape)
            def Dffobj(t):
                tt = t.reshape(in_shape)
                dval = jac(tt)
                return dval.reshape( (pp,pp) )
        else:
            Dffobj = jac
    elif serial_problem:
        Dffobj = MySerialJacobian(fobj, in_shape)
    else:
        Dffobj = MyJacobian(ffobj)


    if lb == None:
        lb = -np.inf*np.ones(len(x0.flatten()))
    if ub == None:
        ub = np.inf*np.ones(len(x0.flatten())).flatten()

    if not serial_problem:
         lb = lb.flatten()
         ub = ub.flatten()

    if not serial_problem:
        x0 = x0.flatten()


    if method == 'fsolve':
        import scipy.optimize as optimize
        factor = options.get('factor')
        factor = factor if factor else 1
        [sol,infodict,ier,msg] = optimize.fsolve(ffobj, x0, fprime=Dffobj, factor=factor, full_output=True, xtol=1e-10, epsfcn=1e-9)
        if ier != 1:
            print(msg)

    elif method == 'newton':
        from dolo.numeric.newton import newton_solver
        fun = lambda x: [ffobj(x), Dffobj(x) ]
        [sol,nit] = newton_solver(fun,x0, verbose=verbose, infos=True)

    elif method == 'lmmcp':
        from dolo.numeric.extern.lmmcp import lmmcp
        sol = lmmcp(lambda t: -ffobj(t), lambda u: -Dffobj(u),x0,lb,ub,verbose=verbose,options=options)

    elif method == 'ncpsolve':
        from dolo.numeric.ncpsolve import ncpsolve
        fun = lambda x: [ffobj(x), Dffobj(x) ]
        [sol,nit] = ncpsolve(fun,lb,ub,x0, verbose=verbose, infos=True, serial=serial_problem)

    else:
        raise Exception('Unknown method : '+str(method))
    sol = sol.reshape(in_shape)

    if infos:
        return [sol, nit]
    else:
        return sol


def MyJacobian(fun, eps=1e-6):

    def rfun(x):
        n = len(x)
        x0 = x.copy()
        y0 = fun(x)
        D = np.zeros( (len(y0),len(x0)) )
        for i in range(n):
            delta = np.zeros(len(x))
            delta[i] = eps
            y1 = fun(x+delta)
            y2 = fun(x-delta)
            D[:,i] = (y1 - y2)/eps/2
        return D
    return rfun



def MySerialJacobian(fun, shape, eps=1e-6):

    def rfun(x):

        x = x.reshape(shape)

        #        x0 = x.copy()
        p = x.shape[0]
        N = x.shape[1]

        y0 = fun(x)

        assert( y0.shape[0] == p)
        assert( y0.shape[1] == N)

        Dc = np.zeros( (p,p,N) )  # compressed jacobian
        for i in range(p):
            delta = np.zeros((p,N))
            delta[i,:] = eps
            y1 = fun(x+delta)
            y2 = fun(x-delta)
            Dc[i,:,:] = (y1 - y2)/eps/2

        return Dc.swapaxes(0,1)

#        D = np.zeros((p,N,p,N))
#        for n in range(N):
#            D[:,n,:,n] = Dc[:,:,n].T
#
#        return D.reshape(p*N, p*N)
#        #return D

    return rfun
