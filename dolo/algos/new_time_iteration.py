from dolo.compiler.model import Model

import numpy
from dolo import dprint
from dolo.compiler.model import Model
from dolo.numeric.processes import DiscretizedIIDProcess, DiscretizedProcess
from dolo.numeric.decision_rule import DecisionRule, Linear
from dolo.numeric.grids import CartesianGrid
from dolo.numeric.serial_operations import serial_multiplication_vector, serial_solve
from dolo.numeric.optimize.newton import SerialDifferentiableFunction


class CVector:

    data: object # vector


    def __init__(self, x0):

        assert(x0.ndim>=3)

        self.x0 = x0

        self.datashape = x0.shape[2:]
        self.shape = x0.shape[:2]

        self.data = x0.reshape((-1,)+self.datashape)

        # self.views = [x0[i,:,:] for i in range(x0.shape[0])]

    def __str__(self):

        s = f"CVector(shape={self.shape}, datashape={self.datashape})"
        return s

    def zeros_like(self):

        x1 = self.x0*0
        return CVector(x1)

    def __getitem__(self, args):
        if len(args)!=3 or (args[1]!=slice(None,None)) or (args[2]!=slice(None,None)):
            raise Exception("Unsupported Subscripts.")
        else:
            i = args[0]
            return self.x0[i,:,:]


    def __setitem__(self, args, value):
        if len(args)!=3 or (args[1]!=slice(None,None)) or (args[2]!=slice(None,None)):
            raise Exception("Unsupported Subscripts.")
        else:
            i = args[0]
            self.x0[i,:,:] = value


    def __mul__(self, x):

        if isinstance(x,CVector):
            assert(self.shape == x.shape)
            x = x.x0
        return CVector(self.x0*x)

    def __rmul__(self, x):
        if isinstance(x,CVector):
            assert(self.shape == x.shape)
            x = x.x0
        return CVector(self.x0*x)

    def __add__(self, x):
        if isinstance(x,CVector):
            assert(self.shape == x.shape)
            x = x.x0
        return CVector(self.x0+x)

    def __radd__(self, x):
        if isinstance(x,CVector):
            assert(self.shape == x.shape)
            x = x.x0
        return CVector(self.x0+x)

    def __sub__(self, x):
        if isinstance(x,CVector):
            assert(self.shape == x.shape)
            x = x.x0
        return CVector(self.x0-x)

    def __truediv__(self, x):
        if isinstance(x,CVector):
            assert(self.shape == x.shape)
            x = x.x0
        return CVector(self.x0/x)


    def __matmul__(self, x):

        assert(isinstance(x, CVector))

        A = self.data
        B = x.data
        from dolo.numeric.serial_operations import serial_multiplication, serial_multiplication_vector
        
        if len(A.shape) == len(B.shape) == 3:
            C = serial_multiplication(A,B)
        elif len(A.shape)==3 and len(B.shape)==2:
            C = serial_multiplication_vector(A,B)

        return CVector(C.reshape(self.shape+C.shape[1:]))


    def norm(self):
        return abs(self.data).max()

    def solve(self, x):

        assert(isinstance(x, CVector))

        A = self.data
        B = x.data
        from dolo.numeric.serial_operations import serial_solve_numba
        
        # if len(A.shape) == len(B.shape) == 3:
        #     C = serial_multiplication(A,B)
        # elif len(A.shape)==3 and len(B.shape)==2:
        #     C = serial_multiplication_vector(A,B)

        C = serial_solve(A,B)

        return CVector(C.reshape(self.shape+C.shape[1:]))




class Euler:

    model: Model
    grid: object
    dprocess: DiscretizedProcess
    dr: DecisionRule
    x0: object   # N_e . N . n_x array
    p: object   # n_p vector

    def __init__(self, model, grid=dict(), interp_method='cubic'):

        self.model = model
        grid, dprocess = model.discretize(**grid)
        self.grid = grid
        self.dprocess = dprocess

        self.dr = DecisionRule(grid['exo'], grid['endo'], dprocess=dprocess, interp_method=interp_method)

        n_x = len(model.symbols["controls"])
        N_e = max(1,grid['exo'].n_nodes)
        N = grid['endo'].n_nodes

        x0 = numpy.zeros((N_e, N, n_x))
        x0[:, :, :] = model.calibration["controls"][None, None, :]
        self.x0 = CVector(x0)
        
        self.p = model.calibration['parameters']
        self.sh_C = (N_e*N, n_x)

    def __call__(self, x0, x1=None):


        f = self.model.functions['arbitrage']
        g = self.model.functions['transition']

        s = self.grid["endo"].nodes      
        dp = self.dprocess
        p = self.p


        if x1 is not None:
            self.dr.set_values(x1.data.reshape(x1.x0.shape))
        

        return residuals_simple(f, g, s, x0, self.dr, dp, p)

    def d_A(self, x0, x1=None):

        sh = x0.x0.shape

        def fun(x):
            return self( CVector( x.reshape( sh )) ).data.copy()

        z0 = x0.data.copy()
        
        from dolo.numeric.optimize.newton import serial_newton

        dfun = SerialDifferentiableFunction(fun)
    
        r, J = dfun(z0)

        R = CVector(r.reshape(x0.x0.shape))
        J = CVector(J.reshape(x0.shape + J.shape[1:]))

        return R,J
    
    def d_B(self, x0, x1=None):





        f = self.model.functions['arbitrage']
        g = self.model.functions['transition']

        s = self.grid["endo"].nodes      
        dp = self.dprocess
        p = self.p
        dr = self.dr

        ## 
        x = x0.x0


        res, jres, S_ij = euler_residuals(f, g, s, x, dr, dp, p, diff=True, with_jres=True, set_dr=False)


        import copy
        cdr = copy.deepcopy(dr)

        R = CVector(res.reshape(x.shape))
        L = Jacobian(jres, S_ij, cdr)

        return R, L


class Jacobian:

    def __init__(self, M_ij, S_ij, cdr):

        self.M_ij = M_ij
        self.S_ij = S_ij
        self.cdr = cdr

    def __matmul__(self, A, inplace=False):

        π = A.x0

        M_ij = self.M_ij
        S_ij = self.S_ij

        if not inplace:
            π = π.copy()

        from dolo.algos.improved_time_iteration import d_filt_dx

        d_filt_dx(π, M_ij, S_ij, self.cdr)

        return CVector(π)

    def ldiv(self, A): # performs the computation in place

        from .invert import solve_tensor

        M_ij = self.M_ij
        n_m, n_im = self.M_ij.shape[:2]

        V = A.x0

        for i_m in range(n_m):
            for j_m in range(n_im):
                M = M_ij[i_m, j_m, :, :, :]
                X = V[i_m, :, :, :].copy()
                rhs = solve_tensor(X, M)

    def __imul__(self, other):

        self.M_ij[...] *= other
        return self

def residuals_simple(f, g, s, x, dr, dprocess, parms)->CVector:

    N = s.shape[0]
    n_s = s.shape[1]

    res = x.zeros_like()

    for i_ms in range(dprocess.n_nodes):

        # solving on grid for markov index i_ms
        m = numpy.tile(dprocess.node(i_ms), (N, 1))
        xm = x[i_ms, :, :]

        for I_ms in range(dprocess.n_inodes(i_ms)):
            M = numpy.tile(dprocess.inode(i_ms, I_ms), (N, 1))
            prob = dprocess.iweight(i_ms, I_ms)
            S = g(m, s, xm, M, parms)
            XM = dr.eval_ijs(i_ms, I_ms, S)
            rr = f(m, s, xm, M, S, XM, parms)
            res[i_ms, :, :] += prob * rr

    return res


import time

def euler_residuals(
    f,
    g,
    s,
    x,
    dr,
    dp,
    p_,
    diff=True,
    with_jres=False,
    set_dr=True,
    jres=None,
    S_ij=None,
):


    if set_dr:
        dr.set_values(x)

    N = s.shape[0]
    n_s = s.shape[1]
    n_x = x.shape[2]

    n_ms = max(dp.n_nodes, 1)  # number of markov states
    n_im = dp.n_inodes(0)

    res = numpy.zeros_like(x)

    if with_jres:
        if jres is None:
            jres = numpy.zeros((n_ms, n_im, N, n_x, n_x))
        if S_ij is None:
            S_ij = numpy.zeros((n_ms, n_im, N, n_s))

    for i_ms in range(n_ms):
        m_ = dp.node(i_ms)
        xm = x[i_ms, :, :]
        for I_ms in range(n_im):
            M_ = dp.inode(i_ms, I_ms)
            w = dp.iweight(i_ms, I_ms)
            S = g(m_, s, xm, M_, p_, diff=False)
            XM = dr.eval_ijs(i_ms, I_ms, S)
            if with_jres:
                ff = SerialDifferentiableFunction(
                    lambda u: f(m_, s, xm, M_, S, u, p_, diff=False)
                )
                rr, rr_XM = ff(XM)

                rr = f(m_, s, xm, M_, S, XM, p_, diff=False)
                jres[i_ms, I_ms, :, :, :] = w * rr_XM
                S_ij[i_ms, I_ms, :, :] = S
            else:
                rr = f(m_, s, xm, M_, S, XM, p_, diff=False)
            res[i_ms, :, :] += w * rr

    t2 = time.time()

    if with_jres:
        return res, jres, S_ij
    else:
        return res



def solve(model, verbose=True, maxit=500):

    F = Euler(model)

    x0 = F.x0


    for  i in range(maxit):

        r = ( F(x0, x0) )

        r, J = ( F.d_A(x0) )

        dx = J.solve(r)

        if verbose:
            print(f"{i} | {r.norm()}")

        x0 = x0 - dx



def improved_time_iteration(model, tol_ε=1e-8):

    F = Euler(model)

    x0 = F.x0


    for k in range(10):


        r = F(x0, x0)

        err_ε = r.norm()

        if err_ε<tol_ε:
            print(f"Error {k} = {err_ε} |")
            break


        R, J = F.d_A(x0)
        R, L = F.d_B(x0)

        r = J.solve(r)

        L.ldiv(J)
        L *= -1


        tol_i = 1e-10

        dx = r
        du = r
        for n in range(1000):
            du = L@du
            dx = dx + du
            if du.norm()<tol_i:
                break

        print(f"Error {k} = {err_ε} | {n}")

        x0 = x0 - dx

    print(x0.shape)
    return x0
