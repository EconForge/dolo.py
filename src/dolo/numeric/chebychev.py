import numpy as np


def cheb_extrema(n):
    jj = np.arange(1.0,n+1.0)
    zeta =  np.cos( np.pi * (jj-1) / (n-1 ) )
    return zeta

def chebychev(x, n):
    dim = x.shape
    results = np.zeros((n+1,) + dim)   
    results[0,:] = np.ones(dim)
    results[1,:] = x
    for i in range(2,n+1):
        results[i,:] = 2 * x * results[i-1,:] - results[i-2,:]
    return results

def chebychev2(x, n):
    dim = x.shape
    results = np.zeros((n+1,) + dim)   
    results[0,:] = np.ones(dim)
    results[1,:] = 2*x
    for i in range(2,n+1):
        results[i,:] = 2 * x * results[i-1,:] - results[i-2,:]
    return results
