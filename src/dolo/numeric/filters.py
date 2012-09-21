import numpy as np
from numpy import sin
from scipy.signal import fftconvolve


def hpfilter( x, lam=1600 ):
    T = x.shape[-1]
    Au = np.array([
        [1, -2],
        [-2, 5]
    ])
    Ad = np.array([
        [5, -2],
        [-2, 1]
    ])
    a = np.diag( np.ones( T )*6 )
    b = np.diag( np.ones( T-1 )*(-4), 1 )
    c = np.diag( np.ones( T-2 ), 2 )
    d = np.diag( np.ones( T-1 )*(-4), -1 )
    e = np.diag( np.ones( T-2 ), -2 )
    M = a + b + c + d + e
    M[0:2,0:2] = Au
    M[-2:,-2:] = Ad
    M *= lam
    M += np.eye(T)

    if x.ndim == 1:
        return np.linalg.solve(M,x)
    elif x.ndim > 3:
        raise Exception('HP filter is not defined for dimension >= 3.')
    else:
        return np.linalg.solve(M,x.T).T


def bandpass_filter(data, k, w1, w2):
    """
    This funciton will apply a bandpass filter to data. It will be kth
    order and will select the band bewtween w1 and w2.

    Parameters
    ----------
        data: array, dtype=float
            The data you wish to filter
        k: number, int
            The order of approximation for the filter. A max value for
            this isdata.size/2
        w1: number, float
            This is the lower bound for which frecuecies will pass
            through.
        w2: number, float
            This is the upper bound for which frecuecies will pass
            through.

    Returns
    -------
        y: array, dtype=float
            The filtered data.
    """
    data = np.asarray(data)
    low_w = np.pi * 2 / w2
    high_w = np.pi * 2 / w1
    bweights = np.zeros(2 * k + 1)
    bweights[k] = (high_w - low_w) / np.pi
    j = np.arange(1, int(k) + 1)
    weights = 1 / (np.pi * j) * (sin(high_w * j) - sin(low_w * j))
    bweights[k + j] = weights
    bweights[:k] = weights[::-1]

    bweights -= bweights.mean()

    return fftconvolve(bweights, data, mode='valid')
