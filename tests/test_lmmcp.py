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

class  Test_LMMCPTestCase(unittest.TestCase):

    def test_josephy(self):

        from dolo.numeric.extern import lmmcp

        x0=np.array( [1.25, 0.00, 0.00, 0.50] )
        lb=np.array( [0.00, 0.00, 0.00, 0.00] )
        ub=np.array( [1e20, 1e20, 1e20, 1e20] )

        resp = lmmcp.lmmcp(josephy, Djosephy, x0, lb, ubz)

        sol = np.array([1.224746243, -0.0000, 0.0000, 0.5000])

        assert( abs(sol - resp).max()<1e-6 )

    def test_solver(self):

        from dolo.numeric.solver import solver

        x0=np.array( [1.25, 0.00, 0.00, 0.50] )
        lb=np.array( [0.00, 0.00, 0.00, 0.00] )
        ub=np.array( [1e20, 1e20, 1e20, 1e20] )

        resp = solver(josephy, x0, lb, ub, verbose=True, jac=Djosephy)

        sol = np.array([1.224746243, -0.0000, 0.0000, 0.5000])
        assert( abs(sol - resp).max()<1e-6 )



if __name__ == '__main__':
    unittest.main()

