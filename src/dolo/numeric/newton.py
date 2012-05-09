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
#	res = f(x0)
#        dres = df(x0)
        fnorm = abs(res).max()  # suboptimal

        dx = - serial_solve(dres,res)
        x = x0 + dx

#        for i in range(backsteps):
#            xx = x0 + dx/(2**i)
#            if not ub==None:
#                xx = numpy.maximum(xx, lb)
#                xx = numpy.minimum(xx, ub)
#            new_res = f(xx)[0]
#            new_fnorm = abs(new_res).max()
#            if numpy.isfinite(new_fnorm) and new_fnorm < fnorm: # all right proceed to next iteration
#                x = xx
#                break
#            if i == backsteps -1:
#                if numpy.isfinite(new_fnorm):
#                    x = xx
#                else:
#                    raise Exception('Non finite value found')

        #err = abs(dx).max()
        err = abs(res).max()

        x0 = x
        it += 1
#    print (it <=maxit)
#    print(err)
    if not infos:
        return x
    else:
        return [x, it]

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
