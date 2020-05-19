from dolo import *
from dolo.numeric.decision_rule import DecisionRule
from dolo.numeric.optimize.newton import SerialDifferentiableFunction

model = yaml_import("/home/pablo/dolo.py/examples/models/rbc.yaml")


dr = time_iteration(model, verbose=False, interp_method='cubic')
epsilon = 1e-8

class Residuals:

    def __init__(self, model, interp_method='cubic'):


        self.f = model.functions['arbitrage']
        self.g = model.functions['transition']
        x_lb = model.functions['controls_lb']
        x_ub = model.functions['controls_ub']

        self.p = model.calibration['parameters']

        self.dp = dp = model.exogenous.discretize()

        n_m = max(dp.n_nodes,1)
        n_s = len(model.symbols['states'])

        self.gridsize = (n_m, n_s)

        grid = model.get_endo_grid()
        self.s = grid.nodes

        self.dr = DecisionRule(dp.grid, grid, dprocess=dp, interp_method=interp_method)
        # ddr_filt = DecisionRule(dp.grid, grid, dprocess=dp, interp_method=interp_method)

        # s = ddr.endo_grid
        s = grid.nodes
        N = s.shape[0]
        n_x = len(model.symbols['controls'])

        self.x0 = model.calibration['controls'][None,None,].repeat(n_m, axis=0).repeat(N,axis=1)

    def __call__(self, x0, x1, diff_A=True, diff_B=True, set_dr=True):

        f = self.f
        g = self.g
        p_ = self.p
        dr = self.dr
        dp = self.dp

        if set_dr:
            dr.set_values(x1)

        s = self.s
        x = x0

        N = s.shape[0]
        n_s = s.shape[1]
        n_x = x.shape[2]

        n_ms = max(dp.n_nodes,1)   # number of markov states
        n_im = dp.n_inodes(0)

        res = numpy.zeros_like(x)

        if diff_B:
            J_ij = numpy.zeros((n_ms,n_im,N,n_x,n_x))
            S_ij = numpy.zeros((n_ms,n_im,N,n_s))

        for i_ms in range(n_ms):
            m_ = dp.node(i_ms)
            xm = x[i_ms,:,:]
            for I_ms in range(n_im):
                M_ = dp.inode(i_ms, I_ms)
                w = dp.iweight(i_ms, I_ms)
                S = g(m_, s, xm, M_, p_, diff=False)
                XM = dr.eval_ijs(i_ms, I_ms, S)
                if diff_B:
                    ff = SerialDifferentiableFunction(lambda u: f(m_,s,xm,M_,S,u,p_,diff=False),
                        epsilon=epsilon)
                    rr, rr_XM = ff(XM)

                    # rr = f(m_,s,xm,M_,S,XM,p_,diff=False)
                    J_ij[i_ms,I_ms,:,:,:] = w*rr_XM
                    S_ij[i_ms,I_ms,:,:] = S
                else:
                    rr = f(m_,s,xm,M_,S,XM,p_,diff=False)
                res[i_ms,:,:] += w*rr

        if (not diff_A) and (not diff_B):
            return res
        
        out = [res]
        if diff_A:
            fun = SerialDifferentiableFunction(
                lambda u: self(u.reshape(x1.shape), x1, diff_A=False, diff_B=False, set_dr=False).reshape((-1,x1.shape[2])),
                epsilon=epsilon
            )
            res, dres = fun( x0.reshape((-1,x1.shape[2]) ) )
            out.append( dres.reshape( (x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[2])) )
        if diff_B:
            import copy
            L = Operator(J_ij, S_ij, copy.deepcopy(self.dr))
            out.append(L)

        return out


import numpy as np
from scipy.sparse.linalg import LinearOperator

class Operator(LinearOperator):

    def __init__(self, M_ij, S_ij, dumdr):
        self.M_ij = M_ij
        self.S_ij = S_ij
        self.n_m = M_ij.shape[0]
        self.N = M_ij.shape[2]
        self.n_x = M_ij.shape[3]
        self.dumdr = dumdr
        self.dtype = numpy.dtype('float64')
        self.counter = 0
        self.addid = False

    @property
    def shape(self):
        nn = self.n_m*self.N*self.n_x
        return (nn,nn)

    def _matvec(self, x):
        self.counter += 1
        xx = x.reshape((self.n_m, self.N, self. n_x))
        yy = self.apply(xx)
        if self.addid:
            yy = xx-yy # preconditioned system
        return yy.ravel()

    def apply(self, π, inplace=False):
        M_ij = self.M_ij
        S_ij = self.S_ij
        n_m = self.n_m
        N = self.N
        n_x = self.n_x
        dumdr = self.dumdr
        if not inplace:
            π = π.copy()
        from dolo.algos.improved_time_iteration import d_filt_dx
        return d_filt_dx(π,M_ij,S_ij,n_m,N,n_x,dumdr)

    def as_matrix(self):

        arg = np.zeros((self.n_m,self.N,self.n_x))
        larg = arg.ravel()
        N = len(larg)
        J = numpy.zeros((N,N))
        for i in range(N):
            if i>0:
                larg[i-1] = 0.0
            larg[i] = 1.0
            J[:,i] = self.apply(arg).ravel()
        return J


from numpy.linalg import solve
import copy

def linsolve(A,B,inplace=True):

    if isinstance(B, Operator):
        
        if inplace:
            L = B
        else:
            L = Operator(B.M_ij, B.S_ij, copy.deepcopy(B.dumdr))
            
        J_ij = L.M_ij
            # out = copy.copy(J_ij) 
        for i in range(J_ij.shape[0]):
            for j in range(J_ij.shape[1]):
                for n in range(J_ij.shape[2]):
                    J_ij[i,j,n,:,:] = solve(A[i,n,:,:], J_ij[i,j,n,:,:])
        return L

    elif isinstance(B, np.ndarray) and B.ndim==3:
        if inplace:
            out = B
        else:
            out = copy.copy(B)
        for i in range(A.shape[0]):
            for n in range(A.shape[1]):
                out[i,n,:] = solve(A[i,n,:,:], B[i,n,:])
        return out




F = Residuals(model)

x0 = F.x0

import scipy.sparse.linalg


[R, A, L] = F(x0,x0, diff_A=True, diff_B=True)

T = linsolve(A,L,inplace=False)
R = linsolve(A,R,inplace=False)

from dolo.numeric.optimize import newton
import matplotlib.pyplot as plt


def show_evs(T):

    evs = scipy.sparse.linalg.eigs(T)[0]

    fig,ax = plt.subplots()
    plt.grid()
    circle = plt.Circle((0,0),1, fill=False)
    ax.add_artist(circle)
    plt.plot(evs.real, evs.imag,'x', color='red')

    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)


def iti(F, x0, maxit=100, switch=20, tol=1e-6, improvements=10):

    x1 = copy.copy(x0)

    for it in range(maxit):

        def fobj(u): 
            res, dres = F(u.reshape(x1.shape),x1, diff_A=True, diff_B=False)
            return [
                res.reshape((-1, x1.shape[2])),
                dres.reshape((-1, x1.shape[2],x1.shape[2])),
            ]

        x0, nit = newton.newton(fobj, x1.reshape((-1, x1.shape[2])))
        x0 = x0.reshape(x1.shape)

        Delta = x0-x1

        err = abs(Delta).max()

        if (it>switch) and (improvements>0):

            [R, A, B] = F(x0,x1,diff_A=True, diff_B=True)

            L = linsolve(-A,B,inplace=False)

            DD = Delta.copy()
            for im in range(improvements):
                DD = L.apply(DD)
                Delta += DD

        print((it,err,nit))
        
        x1 += Delta
        
        if err<tol:
            break
    
    return x1

linsolve
%time sol = iti(F, F.x0, 100,  improvements=20)
# %time sol = iti(F, x0, 100, improvements=100)


# from dolo.algos.improved_time_iteration import improved_time_iteration

# sol = improved_time_iteration(model, verbose=True, invmethod='gmres')


# x0 = np.concatenate(    [sol.dr(i, F.s)[None,:,:] for i in range(F.dp.n_nodes)],
#     axis=0
# )

# res, A, B = F(x0, x0)

# L = linsolve(-A,B)
# show_evs(L)

# show_evs(sol.L)


# abs(res).max()


# show_evs(sol.L)
