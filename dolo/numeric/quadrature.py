import numpy

from misc import cartesian

def qnwnorm(orders, mu, sigma):

    import pytave
    pytave.addpath('/home/pablo/Programmation/compecon/CEtools')

    orders = numpy.array(orders,dtype=float)
    mu = numpy.array(mu,dtype=float)
    sigma = numpy.array(sigma)

    [x,w] = pytave.feval(2,'qnwnorm',orders, mu, sigma)

    w = numpy.ascontiguousarray(w.flatten())
    x = numpy.ascontiguousarray(x.T)

    return [x,w]

def gauss_hermite_nodes(orders, sigma, mu=None):

    import numpy
    from numpy.polynomial.hermite import hermgauss

    if mu is None:
        mu = numpy.array( [0]*sigma.shape[0] )

    herms = [hermgauss(i) for i in orders]

    points = [ h[0]*numpy.sqrt(2) for h in herms]
    weights = [ h[1]/numpy.sqrt( numpy.pi) for h in herms]


    x = cartesian( points).T
    w = reduce( numpy.kron, weights)

    C = numpy.linalg.cholesky(sigma)

    x = numpy.dot(C, x) + mu[:,numpy.newaxis]

    return [x,w]

if __name__ == '__main__':

    orders = [8,8]
    mu = numpy.array( [0.05,0.01] )
    sigma = numpy.array([
        [0.1,0.015],
        [0.015,0.1],
    ])

    def f(P):
        return P[1,:]**4 - (P[0,:]-1)*P[0,:]**2 + 2

    [x,w] = qnwnorm(orders, mu, sigma)

    [x_numpy, w_numpy] = gauss_hermite_nodes(orders, mu, sigma)

    int_1 = ( w*f(x) ).sum()
    int_2 = ( w_numpy*f(x_numpy) ).sum()


    print('Integrals')
    print(int_1)
    print(int_2)


    assert( abs(int_1-int_2) < 1e-15 )