import unittest

import numpy as np

from dolo.numeric.optimize.ncpsolve import ncpsolve


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

class SerialSolve(unittest.TestCase):
    #
    # def test_serial_smooth(self):
    #
    #     x0 = np.array([0.5,0.5,0.5,0.5])
    #
    #
    #     lb = np.array([0.0,0.6,0.0,0.0])
    #     ub = np.array([1.0,1.0,1.0,0.4])
    #
    #     fval = np.array([ 0.5, 0.5, 0.1,0.5 ])
    #
    #     jac = np.array([
    #         [1.0,0.2,0.1,0.0],
    #         [1.0,0.2,0.1,0.0],
    #         [0.0,1.0,0.2,0.0],
    #         [0.1,1.0,0.2,0.1]
    #     ])
    #
    #     N = 10
    #     d = len(fval)
    #
    #     s_x0 = np.row_stack([x0]*N)
    #     s_lb = np.row_stack([lb]*N)
    #     s_ub = np.row_stack([ub]*N)
    #     s_fval = np.row_stack([fval]*N)
    #
    #
    #     s_jac = np.zeros( (N,d,d) )
    #     for i in range(N):
    #         s_jac[i,:,:] = jac
    #
    #     from dolo.numeric.ncpsolve import smooth
    #
    #     [fnew, Jnew] = smooth(x0, lb, ub, fval, J=jac)
    #
    #     serial_fnew_true = np.zeros( (N,d) )
    #     serial_Jnew_true = np.zeros( (N,d,d) )
    #     for n in range(N):
    #         serial_fnew_true[n,:] = fnew
    #         serial_Jnew_true[n,:,:] = Jnew
    #
    #
    #     [serial_fnew, serial_Jnew] = smooth(s_x0, s_lb, s_ub, s_fval, J=s_jac, serial=True)
    #
    #     from numpy.testing import assert_equal
    #
    #     assert_equal(serial_fnew, serial_fnew_true)
    #     assert_equal(serial_Jnew, serial_Jnew_true)


    def test_serial_solve(self):

        fun = lambda x: [-josephy(x), -Djosephy(x)]

        x0=np.array( [1.25, 0.01, 0.01, 0.50] )
        lb=np.array( [0.00, 0.00, 0.00, 0.00] )
        # ub=np.array( [inf, inf, inf, inf] )
        ub=np.array( [1e20, 1e20, 1e20, 1e20] )


        # resp = ncpsolve(fun,  lb, ub, x0, tol=1e-15)

        sol = np.array( [ 1.22474487e+00, 0.00000000e+00, 3.60543164e-17, 5.00000000e-01])

        # assert_almost_equal(sol,  resp)

        N = 10
        d = len(x0)

        s_x0 = np.row_stack([x0]*N)
        s_lb = np.row_stack([lb]*N)
        s_ub = np.row_stack([ub]*N)

        def serial_fun(xvec):

            resp = np.zeros( (N,d) )
            dresp = np.zeros( (N,d,d) )
            for n in range(N):
                [v, dv] = fun(xvec[n,:])
                resp[n,:] = v
                dresp[n,:,:] = dv
            return [resp, dresp]

        res = serial_fun(s_x0)[0]

        serial_sol = ncpsolve( serial_fun, s_lb, s_ub, s_x0, jactype='serial', verbose=True)

        print(serial_sol)









if __name__ == '__main__':
    from numpy import inf

    fun = lambda x: [-josephy(x), -Djosephy(x)]

    x0=np.array( [1.25, 0.01, 0.01, 0.50] )
    lb=np.array( [0.00, 0.00, 0.00, 0.00] )
    ub=np.array( [inf, inf, inf, inf] )

    resp = ncpsolve(fun,  lb, ub, x0, tol=1e-15)

    sol = np.array( [ 1.22474487e+00, 0.00000000e+00, 3.60543164e-17, 5.00000000e-01])

    from numpy.testing import assert_almost_equal

    assert_almost_equal(sol,  resp[0])


    N = 2
    d = len(x0)


    serial_sol_check = np.zeros((d,N))
    for n in range(N):
        serial_sol_check[:,n] = resp[0]

    s_x0 = np.column_stack([x0]*N)
    s_lb = np.column_stack([lb]*N)
    s_ub = np.column_stack([ub]*N)

    def serial_fun(xvec):

        resp = np.zeros( (d,N) )
        dresp = np.zeros( (d,d,N) )
        for n in range(N):
            [v, dv] = fun(xvec[:,n])
            resp[:,n] = v
            dresp[:,:,n] = dv
        return [resp, dresp]

    serial_sol = ncpsolve( serial_fun, s_lb, s_ub, s_x0, serial=True)


    err = abs(serial_sol[0] - serial_sol_check).max()

    assert_almost_equal( serial_sol[0],  serial_sol_check )

