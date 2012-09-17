import numpy as np


def solver(fobj,x0,lb=None,ub=None,options={},method='lmmcp',jac='default',verbose=False):
    
    in_shape = x0.shape

    ffobj = lambda x: fobj(x.reshape(in_shape)).flatten()

    if not isinstance(jac,str):
        pp = np.prod(in_shape)
        def Dffobj(t):
            tt = t.reshape(in_shape)
            dval = jac(tt)
            return dval.reshape( (pp,pp) )
    elif jac=='precise':
        from numdifftools import Jacobian
        Dffobj = Jacobian(ffobj)
    else:
        Dffobj = MyJacobian(ffobj)

    if method == 'fsolve':
        import scipy.optimize as optimize
        factor = options.get('factor')
        factor = factor if factor else 1
        [sol,infodict,ier,msg] = optimize.fsolve(ffobj,x0.flatten(),fprime=Dffobj,factor=factor,full_output=True,xtol=1e-10,epsfcn=1e-9)
        if ier != 1:
            print msg

    elif method == 'anderson':
        import scipy.optimize as optimize
        sol = optimize.anderson(ffobj,x0.flatten())

    elif method == 'newton_krylov':
        import scipy.optimize as optimize
        sol = optimize.newton_krylov(ffobj,x0.flatten())

    elif method == 'lmmcp':

        from dolo.numeric.extern.lmmcp import lmmcp,Big
        if lb == None:
            lb = -Big*np.ones(len(x0.flatten()))
        else:
            lb = lb.flatten()
        if ub == None:
            ub = Big*np.ones(len(x0.flatten()))
        else:
            ub = ub.flatten()
        sol = lmmcp(lambda t: -ffobj(t), lambda u: -Dffobj(u),x0.flatten(),lb,ub,verbose=verbose,options=options)

    elif method == 'ncpsolve':

        from dolo.numeric.ncpsolve import ncpsolve
        if lb == None:
            lb = -np.inf*np.ones(len(x0.flatten()))
        else:
            lb = lb.flatten()
        if ub == None:
            ub = np.inf*np.ones(len(x0.flatten())).flatten()
        else:
            ub = ub.flatten()
        fun = lambda x: [ffobj(x),Dffobj(x)]
        [sol,fval] = ncpsolve(fun,lb,ub,x0.flatten(), verbose=verbose)

    return sol.reshape(in_shape)


def MyJacobian(fun):
    eps = 1e-10
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
