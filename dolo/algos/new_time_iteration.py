from dolo.compiler.model import Model

import numpy
from dolo import dprint
from dolo.compiler.model import Model
from dolo.numeric.processes import DiscretizedIIDProcess, DiscretizedProcess
from dolo.numeric.decision_rule import DecisionRule
from dolo.numeric.grids import CartesianGrid
from dolo.numeric.serial_operations import serial_multiplication_vector


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
            x = x.x0
        return CVector(self.x0*x)

    def __mul__(self, x):
        if isinstance(x,CVector):
            x = x.x0
        return CVector(self.x0-x)

    def __add__(self, x):
        if isinstance(x,CVector):
            x = x.x0
        return CVector(self.x0+x)

    def __radd__(self, x):
        if isinstance(x,CVector):
            x = x.x0
        return CVector(self.x0+x)

    def __sub__(self, x):
        if isinstance(x,CVector):
            x = x.x0
        return CVector(self.x0-x)

    def __truediv__(self, x):
        if isinstance(x,CVector):
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

        C = serial_solve_numba(A,B)

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

        def fun(x):
            return self( CVector( x.reshape( x0.x0.shape)) ).data.copy()

        z0 = x0.data.copy()
        
        from dolo.numeric.optimize.newton import SerialDifferentiableFunction

        dfun = SerialDifferentiableFunction(fun)
    
        r, J = dfun(z0)

        return CVector(J.reshape(x0.shape + J.shape[1:]))

    
    def d_B(self, x0, x1=None):
        pass


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

