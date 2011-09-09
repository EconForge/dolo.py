import numpy as np


def solver(fobj,x0,options={},method='lmmcp',jac='default',verbose=False):
    in_shape = x0.shape

    ffobj = lambda x: fobj(x.reshape(in_shape)).flatten()

    if jac=='precise':
        from numdifftools import Jacobian
        Dffobj = Jacobian(ffobj)
    else:
        Dffobj = MyJacobian(ffobj)

    if method == 'fsolve':
        import scipy.optimize as optimize
        factor = options.get('factor')
        factor = factor if factor else 1
        [sol,infodict,ier,msg] = optimize.fsolve(ffobj,x0.flatten(),factor=factor,full_output=True,xtol=1e-10,epsfcn=1e-9)
        if ier != 1:
            print msg
    elif method == 'broyden1':
        import scipy.optimize as optimize
        sol = optimize.anderson(ffobj,x0.flatten())
    elif method == 'newton_krylov':
        import scipy.optimize as optimize
        sol = optimize.newton_krylov(ffobj,x0.flatten())
    elif method == 'lmmcp':
        from dolo.numeric.extern.lmmcp import lmmcp
        #lb = -np.inf*np.ones(len(x0.flatten()))
        #ub = np.inf*np.ones(len(x0.flatten()))
        lb = -np.inf*np.ones(len(x0.flatten()))
        ub = np.inf*np.ones(len(x0.flatten()))

        sol = lmmcp(ffobj,Dffobj,x0.flatten(),lb,ub,verbose=verbose,options=options)

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
