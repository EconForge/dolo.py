import unittest



def hpfilter_dense( x, lam=1600 ):

    import numpy as np
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



class  FiltersTest(unittest.TestCase):

    def test_hpfilter(self):

        from dolo.numeric.filters import hp_filter

        import numpy
        import numpy.random

        sig = numpy.eye( 2 )  # number of series
        N = 100  # number of observations

        dumb_series = numpy.random.multivariate_normal([0,0], sig, N).T

        trend_test = hpfilter_dense(dumb_series)

        [T, cycle] = hp_filter(dumb_series)

        from numpy.testing import assert_almost_equal

        assert_almost_equal(trend_test, T)


if __name__ == "__main__":

    unittest.main()