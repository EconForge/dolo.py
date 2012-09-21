import numpy as np
from numpy import sin
import scipy as sp
import scipy.sparse.linalg as spla
from scipy.signal import fftconvolve


def hp_filter(data, lamb=1600):
    """
    This function will apply a Hodrick-Psomething filter to a dataset.
    The return value is the filtered data-set found according to:
        min sum((X[t] - T[t])**2 + lamb*((T[t+1] - T[t]) - (T[t] - T[t-1]))**2)
          T   t

    T = Lambda**-1 * Y

    Parameters
    ----------
        data: array, dtype=float
            The data set for which you want to apply the HP_filter.
            This mustbe a numpy array.
        lamb: array, dtype=float
            This is the value for lambda as used in the equation.

    Returns
    -------
        T: array, dtype=float
            The solution to the minimization equation above (the trend).
        Cycle: array, dtype=float
            This is the 'stationary data' found by Y - T.

    Notes
    -----
        This function implements sparse methods to be efficient enough to handle
        very large data sets.
    """
    Y = np.asarray(data)
    if Y.ndim > 1:
        Y = Y.squeeze()
    lil_t = len(Y)
    big_Lambda = sp.sparse.eye(lil_t, lil_t)
    big_Lambda = sp.sparse.lil_matrix(big_Lambda)

    # Use FOC's to build rows by group. The first and last rows are similar.
    # As are the second-second to last. Then all the ones in the middle...
    first_last = np.array([1 + lamb, -2 * lamb, lamb])
    second = np.array([-2 * lamb, (1 + 5 * lamb), -4 * lamb, lamb])
    middle_stuff = np.array([lamb, -4. * lamb, 1 + 6 * lamb, -4 * lamb, lamb])

    #--------------------------- Putting it together --------------------------#

    # First two rows
    big_Lambda[0, 0:3] = first_last
    big_Lambda[1, 0:4] = second

    # Last two rows. Second to last first
    big_Lambda[lil_t - 2, -4:] = second
    big_Lambda[lil_t - 1, -3:] = first_last

    # Middle rows
    for i in range(2, lil_t - 2):
        big_Lambda[i, i - 2:i + 3] = middle_stuff

    # spla.spsolve requires csr or csc matrix. I choose csr for fun.
    big_Lambda = sp.sparse.csr_matrix(big_Lambda)

    T = spla.spsolve(big_Lambda, Y)

    Cycle = Y - T

    return T, Cycle


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
