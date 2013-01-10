import unittest

import numpy as np

from dolo.numeric.ncpsolve import ncpsolve, smooth


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

    def test_simple_solve(self):

        x0 = np.array([0.5,0.5,0.5,0.5])


        lb = np.array([0.0,0.6,0.0,0.0])
        ub = np.array([1.0,1.0,1.0,0.4])

        fval = np.array([ 0.5, 0.5, 0.1,0.5 ])

        jac = np.array([
            [1.0,0.2,0.1,0.0],
            [1.0,0.2,0.1,0.0],
            [0.0,1.0,0.2,0.0],
            [0.1,1.0,0.2,0.1]
        ])

        N = 10
        d = len(fval)

        from dolo.numeric.solver import solver

        sol_fsolve = solver(josephy, x0, method='fsolve')

        sol_lmmcp = solver(josephy, x0, method='lmmcp')

        from numpy.testing import assert_almost_equal

        assert_almost_equal(sol_fsolve, sol_lmmcp)


    def test_serial_problems(self):

        from numpy import inf
        import numpy

        fun = lambda x: [-josephy(x), -Djosephy(x)]

        x0=np.array( [1.25, 0.01, 0.01, 0.50] )
        lb=np.array( [0.00, 0.00, 0.00, 0.00] )
        ub=np.array( [inf, inf, inf, inf] )

        resp = ncpsolve(fun,  lb, ub, x0, tol=1e-15)

        sol = np.array( [ 1.22474487e+00, 0.00000000e+00, 3.60543164e-17, 5.00000000e-01])

        from numpy.testing import assert_almost_equal, assert_equal

        assert_almost_equal(sol,  resp)


        N = 10
        d = len(x0)


        serial_sol_check = np.zeros((d,N))
        for n in range(N):
            serial_sol_check[:,n] = resp[0]

        s_x0 = np.column_stack([x0]*N)
        s_lb = np.column_stack([lb]*N)
        s_ub = np.column_stack([ub]*N)

        def serial_fun(xvec, deriv=None):

            resp = np.zeros( (d,N) )
            if deriv=='serial':
                dresp = np.zeros( (d,d,N) )
            elif deriv=='full':
                dresp = np.zeros( (d,N,d,N) )
            for n in range(N):
                [v, dv] = fun(xvec[:,n])
                resp[:,n] = v
                if deriv=='serial':
                    dresp[:,:,n] = dv
                elif deriv=='full':
                    dresp[:,n,:,n] = dv
#            if deriv=='full':
#                dresp = dresp.swapaxes(0,2).swapaxes(1,3)
            if deriv is None:
                return resp
            else:
                return [resp, dresp]


        serial_fun_val = lambda x: serial_fun(x)
        serial_fun_serial_jac = lambda x: serial_fun(x,deriv='serial')[1]
        serial_fun_full_jac = lambda x: serial_fun(x,deriv='full')[1]

        from dolo.numeric.solver import solver


        print("Serial Bounded solution : ncpsolve")
        serial_sol_with_bounds_without_jac = solver( serial_fun_val, s_x0, lb=s_lb, ub=s_ub, method='ncpsolve', serial_problem=True)

        print("Serial Bounded solution (with jacobian) : ncpsolve")
        serial_sol_with_bounds_with_jac = solver( serial_fun_val, s_x0, s_lb, s_ub, jac=serial_fun_serial_jac, method='ncpsolve', serial_problem=True)


        print("Bounded solution : ncpsolve")
        sol_with_bounds_without_jac = solver( serial_fun_val, s_x0, s_lb, s_ub, method='ncpsolve', serial_problem=False)

        print("Bounded solution (with jacobian) : ncpsolve")
        sol_with_bounds_with_jac = solver( serial_fun_val, s_x0, s_lb, s_ub, jac=serial_fun_full_jac, method='ncpsolve', serial_problem=False)


        print("Serial Unbounded  solution : ncpsolve")
        serial_sol_without_bounds_without_jac = solver( serial_fun_val, s_x0, method='newton', serial_problem=True)

        print("Unbounded solution : fsolve")
        sol_without_bounds_without_jac = solver( serial_fun_val, s_x0, method='fsolve', serial_problem=False)



        print("Unbounded solution (with jacobian) : fsolve")
        sol_without_bounds = solver( serial_fun_val, s_x0, jac=serial_fun_full_jac, method='fsolve', serial_problem=False)


        print("Unbounded solution : lmmcp")
        sol_without_bounds = solver( serial_fun_val, s_x0, jac=serial_fun_full_jac, method='lmmcp', serial_problem=False)

        # TODO : check that results are equal to the benchmark


if __name__ == '__main__':

    unittest.main()