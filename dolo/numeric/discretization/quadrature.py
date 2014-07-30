

from __future__  import division

import numpy
from dolo.numeric.misc import cartesian

# Credits : both routines below are ported from the Compecon Toolbox
# by Paul L Fackler and Mario J. Miranda.
# It is downloadable at http://www4.ncsu.edu/~pfackler/compecon/toolbox.html
def hermgauss(n):

    from  numpy import pi, fix, zeros, sqrt
    maxit = 100
    pim4 = 1/pi**0.25
    m = int( fix( (n+1)/2 ) )

    x = zeros(n)
    w = zeros(n)
    # reasonable starting values
    for i in range(m):
        if i==0:
            z = sqrt(2*n+1)-1.85575*((2*n+1)**(-1/6))
        elif i==1:
            z = z-1.14*(n**0.426)/z
        elif i==2:
            z = 1.86*z+0.86*x[0]
        elif i==3:
            z = 1.91*z+0.91*x[1]
        else:
            z = 2*z+x[i-2]
    # root finding iterations 
        its = 0
        while its<maxit:
            its += 1
            p1 = pim4
            p2 = 0
            for j in range(n):
                p3 = p2
                p2 = p1
                p1 = z*sqrt(2/(j+1))*p2-sqrt(j/(j+1))*p3;
            pp = sqrt(2*n)*p2
            z1 = z
            z = z1-p1/pp
            if abs(z-z1)<1e-14:
                break
        if its >= maxit:
            raise Exception('Failure to converge')
        x[n-i-1] = z
        x[i] = -z
        w[i] = 2/pp**2
        w[n-i-1] = w[i]

    return [x,w]
 



def gauss_hermite_nodes(orders, sigma, mu=None):

    if isinstance(orders, int):
        orders = [orders]

    import numpy

    sigma = sigma.copy()

    if mu is None:
        mu = numpy.array( [0]*sigma.shape[0] )


    herms = [hermgauss(i) for i in orders]

    points = [ h[0]*numpy.sqrt(2) for h in herms]
    weights = [ h[1]/numpy.sqrt( numpy.pi) for h in herms]

    if len(orders) == 1:
        x = numpy.array(points)*sigma
        w = weights[0]
        return [x.T,w]

    else:
        x = cartesian( points).T

        from functools import reduce
        w = reduce( numpy.kron, weights)

        zero_columns = numpy.where(sigma.sum(axis=0)==0)[0]
        for i in zero_columns:
            sigma[i,i] = 1.0

        C = numpy.linalg.cholesky(sigma)

        x = numpy.dot(C, x) + mu[:,numpy.newaxis]

        x = numpy.ascontiguousarray(x.T)

        for i in zero_columns:
            x[:,i] =0

        return [x,w]

#from numpy.polynomial.hermite import hermgauss

           
if __name__ == '__main__':

    orders = [8,8]
    mu = numpy.array( [0.05,0.01] )
    sigma = numpy.array([
        [0.1,0.015],
        [0.015,0.1],
    ])

    from numpy.polynomial.hermite import hermgauss as  hermgauss_numpy

    [xg,wg] = hermgauss_numpy(10)
    [x,w] = hermgauss(10)
    print(w-wg)
    print(x-xg)
