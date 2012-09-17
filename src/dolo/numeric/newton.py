import numpy

def newton_solver(f, x0, lb=None, ub=None, infos=False, backsteps=10, maxit=50, numdiff=False):
    '''Solves many independent systems f(x)=0 simultaneously using a simple gradient descent.
    :param f: objective function to be solved with values p x N . The second output argument represents the derivative with
    values in (p x p x N)
    :param x0: initial value ( p x N )
    :return: solution x such that f(x) = 0
    '''

    from dolo.numeric.serial_operations import serial_multiplication as stv, serial_solve
    err = 1
    tol = 1e-8
    eps = 1e-5
    it = 0
    while err > tol and it <= maxit:
        if not numdiff:
            [res,dres] = f(x0)
        else:
            res = f(x0)
            dres = numpy.zeros( (res.shape[0], x0.shape[0], x0.shape[1]) )
            for i in range(x0.shape[0]):
                xi = x0.copy()
                xi[i,:] += eps
                resi = f(xi)
                dres[:,i,:] = (resi - res)/eps

        try:
            dx = - serial_solve(dres,res)
        except:
            dx = - serial_solve(dres,res, debug=True)
        x = x0 + dx

        err = abs(res).max()

        x0 = x
        it += 1

    if not infos:
        return x
    else:
        return [x, it]

def newton_solver_comp(f, x0, lb, ub, infos=False, backsteps=10, maxit=50, numdiff=False):
    '''Solves many independent systems f(x)=0 simultaneously using a simple gradient descent.
    :param f: objective function to be solved with values p x N . The second output argument represents the derivative with
    values in (p x p x N)
    :param x0: initial value ( p x N )
    :param lb: bounds for first variable
    :param ub: bounds for second variable
    :return: solution x such that f(x) = 0
    '''

    from numpy import row_stack

    def fun_lc(xx):
        x = row_stack([lb, xx])
        res = f(x)
        return res[1:,:]

    def fun_uc(xx):
        x = row_stack([ub, xx])
        res = f(x)
        return res[1:,:]

    [sol_nc, nit0] = newton_solver(f, x0, numdiff=True, infos=True)
    lower_constrained = sol_nc[0,:] < lb
    upper_constrained = sol_nc[0,:] > ub
    not_constrained =  - ( lower_constrained + upper_constrained )


    sol = sol_nc.copy()

    sol[0,:] = lb * lower_constrained + ub * upper_constrained + sol_nc[0,:] * not_constrained
    nit = nit0


#    [sol_lc, nit1] = newton_solver(fun_lc, x0[1:,:], numdiff=True, infos=True)
#    [sol_uc, nit2] = newton_solver(fun_uc, x0[1:,:], numdiff=True, infos=True)
#
#    nit = nit0 + nit1 + nit2
#
#    sol_lc = row_stack([lb, sol_lc])
#    sol_uc = row_stack([ub, sol_uc])
#
#    lower_constrained = sol_nc[0,:] < lb
#    upper_constrained = sol_nc[0,:] > ub
#    not_constrained =  - ( lower_constrained + upper_constrained )
#
#    sol = sol_lc * lower_constrained + sol_uc * upper_constrained + sol_nc * not_constrained

    return [sol,nit]




from dolo.numeric.serial_operations import serial_inversion

if __name__ == '__main__':

    p = 5
    N = 500


    import numpy.random
    V = numpy.random.multivariate_normal([0]*p,numpy.eye(p),size=p)
    print V

    M = numpy.zeros((p,p,N))
    for i in range(N):
        M[:,:,i] = V

    from dolo.numeric.serial_operations import serial_multiplication as stm


    MM = numpy.zeros( (p,N) )



    import time
    t = time.time()
    for i in range(100):
        T = serial_inversion(M)
    s = time.time()
    print('Elapsed :' + str(s-t))


    tt = stm(M,T)
    for i in range(10):
        print tt[:,:,i]
