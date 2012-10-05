import numpy as np
from numpy import sin
import scipy as sp
from scipy.signal import fftconvolve
from scipy.sparse import linalg as spla

def hp_filter(data, lam=1600):
    """
    This function will apply a Hodrick-Prescott filter to a dataset.
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

    if Y.ndim == 2:
        resp = [hp_filter(e) for e in data]
        T = np.row_stack( [e[0] for e in resp] )
        Cycle = np.row_stack( [e[1] for e in resp] )
        return [T,Cycle]

    elif Y.ndim > 2:
        raise Exception('HP filter is not defined for dimension >= 3.')

    lil_t = len(Y)
    big_Lambda = sp.sparse.eye(lil_t, lil_t)
    big_Lambda = sp.sparse.lil_matrix(big_Lambda)

    # Use FOC's to build rows by group. The first and last rows are similar.
    # As are the second-second to last. Then all the ones in the middle...
    first_last = np.array([1 + lam, -2 * lam, lam])
    second = np.array([-2 * lam, (1 + 5 * lam), -4 * lam, lam])
    middle_stuff = np.array([lam, -4. * lam, 1 + 6 * lam, -4 * lam, lam])

    #--------------------------- Putting it together --------------------------#

    # First two rows
    big_Lambda[0, 0:3] = first_last
    big_Lambda[1, 0:4] = second

    # Last two rows. Second to last first : we have to reverse arrays
    big_Lambda[lil_t - 2, -4:] = second[::-1]
    big_Lambda[lil_t - 1, -3:] = first_last[::-1]

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
    This function will apply a bandpass filter to data. It will be kth
    order and will select the band between w1 and w2.

    Parameters
    ----------
        data: array, dtype=float
            The data you wish to filter
        k: number, int
            The order of approximation for the filter. A max value for
            this isdata.size/2
        w1: number, float
            This is the lower bound for which frequencies will pass
            through.
        w2: number, float
            This is the upper bound for which frequencies will pass
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
