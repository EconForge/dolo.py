import unittest

import numpy as np


def josephy(x):
    #   Computes the function value F(x) of the NCP-example by Josephy.
    n=len(x)
    Fx=np.zeros(n)
    Fx[0]=3*x[0]**2+2*x[0]*x[1]+2*x[1]**2+x[2]+3*x[3]-6
    Fx[1]=2*x[0]**2+x[0]+x[1]**2+3*x[2]+2*x[3]-2
    Fx[2]=3*x[0]**2+x[0]*x[1]+2*x[1]**2+2*x[2]+3*x[3]-1
    Fx[3]=x[0]**2+3*x[1]**2+2*x[2]+3*x[3]-3;
    return Fx

def Djosephy(x):
    # Local Variables: x, DFx, n
    # Function calls: Djosephy, zeros, length
    #%
    #%   Computes the Jacobian DF(x) of the NCP-example by Josephy
    #%
    n = len(x)
    DFx = np.zeros( (n, n) )
    DFx[0,0] = 6.*x[0]+2.*x[1]
    DFx[0,1] = 2.*x[0]+4.*x[1]
    DFx[0,2] = 1.
    DFx[0,3] = 3.
    DFx[1,0] = 4.*x[0]+1.
    DFx[1,1] = 2.*x[1]
    DFx[1,2] = 3.
    DFx[1,3] = 2.
    DFx[2,0] = 6.*x[0]+x[1]
    DFx[2,1] = x[0]+4.*x[1]
    DFx[2,2] = 2.
    DFx[2,3] = 3.
    DFx[3,0] = 2.*x[0]
    DFx[3,1] = 6.*x[1]
    DFx[3,2] = 2.
    DFx[3,3] = 3.
    return DFx


class ComplementaritySolve(unittest.TestCase):

    #
    # # TODO: the two first tests fail because ncpsolve assumes a vectorize function
    # def test_infinite_bounds(self):
    #
    #     import numpy
    #     f = lambda x: [-x**3 + 1.2, -numpy.atleast_2d(3*x**2)]
    #     lb = numpy.array([-numpy.inf])
    #     ub = numpy.array([numpy.inf])
    #     x0 = numpy.array([0.3])
    #     res = ncpsolve(f, lb, ub, x0)
    #
    # def test_complementarities(self):
    #
    #     import numpy
    #     from numpy.testing import assert_almost_equal
    #     f = lambda x: [-x**3 + 1.2, -numpy.atleast_2d(3*x**2)]
    #     lb = numpy.array([-1])
    #     ub = numpy.array([1])
    #     x0 = numpy.array([0.3])
    #     res = ncpsolve(f, lb, ub, x0)
    #     assert_almost_equal( res, 1.0)
    #
    # def test_josephy(self):
    #
    #     import numpy
    #
    #     fun = lambda x: [-josephy(x), -Djosephy(x)]
    #
    #     x0 = np.array( [1.25, 0.01, 0.01, 0.50] )
    #
    #     lb = np.array( [0.00, 0.00, 0.00, 0.00] )
    #     ub = np.array( [inf, inf, inf, inf] )
    #
    #     resp = ncpsolve(fun,  lb, ub, x0, tol=1e-15)
    #
    #     sol = numpy.array( [ 1.22474487e+00, 0.00000000e+00, 3.60543164e-17, 5.00000000e-01])
    #
    #     from numpy.testing import assert_almost_equal
    #
    #     assert_almost_equal(sol,  resp)

    def test_lmmcp(self):

        from dolo.numeric.extern import lmmcp

        x0=np.array( [1.25, 0.00, 0.00, 0.50] )
        lb=np.array( [0.00, 0.00, 0.00, 0.00] )
        ub=np.array( [1e20, 1e20, 1e20, 1e20] )

        resp = lmmcp.lmmcp(josephy, Djosephy, x0, lb, ub)

        sol = np.array([1.22474487, -0.0000, 0.0000, 0.5000])
        print(sol)
        print(resp)
        assert( abs(sol - resp).max()<1e-5 )

if __name__ == '__main__':

    unittest.main()