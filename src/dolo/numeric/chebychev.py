import numpy as np


def cheb_extrema(n):
    jj = np.arange(1.0,n+1.0)
    zeta =  np.cos( np.pi * (jj-1) / (n-1 ) )
    return zeta

def chebychev(x, n):
    # computes the chebychev polynomials of the first kind
    dim = x.shape
    results = np.zeros((n+1,) + dim)
    results[0,...] = np.ones(dim)
    results[1,...] = x
    for i in range(2,n+1):
        results[i,...] = 2 * x * results[i-1,...] - results[i-2,...]
    return results

def chebychev2(x, n):
    # computes the chebychev polynomials of the second kind
    dim = x.shape
    results = np.zeros((n+1,) + dim)   
    results[0,...] = np.ones(dim)
    results[1,...] = 2*x
    for i in range(2,n+1):
        results[i,:] = 2 * x * results[i-1,:] - results[i-2,:]
    return results

if __name__ == '__main__':
    points = np.linspace(-1,1,100)
    from matplotlib import pyplot
    cheb = chebychev(points,5)
    cheb2 = chebychev2(points,5)

    def T4(x):
        return ( 8*np.power(x,4) - 8*np.power(x,2) + 1 )
    def U4(x):
        return 4*( 16*np.power(x,4) - 12*np.power(x,2) + 1 )

    true_values_T = np.array([T4(i) for i in points])
    true_values = np.array([U4(i) for i in points])
    #pyplot.plot(points, cheb[4,:])
    pyplot.plot(points, cheb[4,:])
    pyplot.plot(points, true_values_T)
    pyplot.figure()
    pyplot.plot(points, cheb2[4,:])
    pyplot.plot(points, true_values)
    pyplot.show()