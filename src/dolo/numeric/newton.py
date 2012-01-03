def newton_solver(f, x0,infos=False):
    '''Solves many independent systems f(x)=0 simultaneously using a simple gradient descent.
    :param f: objective function to be solved with values p x N . The second output argument represents the derivative with
    values in (p x p x N)
    :param x0: initial value ( p x N )
    :return: solution x such that f(x) = 0
    '''

    from dolo.numeric.serial_operations import strange_tensor_multiplication_vector as stv
    err = 1
    tol = 1e-8
    maxit = 10
    it = 0
    while err > tol and it <= maxit:
        [res,dres] = f(x0)
        dres_inv = serial_inversion(dres)
        x = x0 - stv(dres_inv, res)

        err = abs(x-x0).max()

        x0 = x
        it += 1
    if not infos:
        return x
    else:
        return [x,it]


def serial_inversion(M):
    '''

    :param M: a pxpxN array
    :return: a pxpxN array T such that T(:,:,i) is the inverse of M(:,:,i)
    '''

    import numpy
    from numpy.linalg import inv

    p = M.shape[0]
    assert(M.shape[1] == p)
    N = M.shape[2]
    T = numpy.zeros((p,p,N))


    for i in range(N):
        T[:,:,i] = inv(M[:,:,i])

    return T


if __name__ == '__main__':

    p = 5
    N = 500


    import numpy.random
    V = numpy.random.multivariate_normal([0]*p,numpy.eye(p),size=p)
    print V

    M = numpy.zeros((p,p,N))
    for i in range(N):
        M[:,:,i] = V

    from dolo.numeric.serial_operations import strange_tensor_multiplication as stm
    from dolo.numeric.serial_operations import strange_tensor_multiplication_vector as stv


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
