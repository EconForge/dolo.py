from dolo.numeric.tensor import sdot,mdot

import numpy as np

TOL = 1e-10

# credits : second_order_solver is adapted from Sven Schreiber's port of Uhlig's Toolkit.

def second_order_solver(FF,GG,HH):
    from extern.qz import qz,qzdiv
    
    from numpy import mat,c_,r_,eye,zeros,real_if_close,diag,allclose,where,diagflat
    from numpy.linalg import solve
    
    Psi_mat = mat(FF)
    Gamma_mat = mat(-GG)
    Theta_mat = mat(-HH)
    m_states = FF.shape[0]
        
    Xi_mat = r_[c_[Gamma_mat, Theta_mat],
                c_[eye(m_states), zeros((m_states, m_states))]]

        
    Delta_mat = r_[c_[Psi_mat, zeros((m_states, m_states))], 
                   c_[zeros((m_states, m_states)), eye(m_states)]]

    AAA,BBB,Q,Z = qz(Delta_mat, Xi_mat)
        
    Delta_up,Xi_up,UUU,VVV = [real_if_close(mm) for mm in (AAA,BBB,Q,Z)]
       
    Xi_eigval = diag(Xi_up)/where(diag(Delta_up)>TOL, diag(Delta_up), TOL)

    Xi_sortindex = abs(Xi_eigval).argsort()
    # (Xi_sortabs doesn't really seem to be needed)
    
    Xi_sortval = Xi_eigval[Xi_sortindex]
    
    Xi_select = slice(0, m_states)
    
    stake = (abs(Xi_sortval[Xi_select])).max() + TOL
    
    Delta_up,Xi_up,UUU,VVV = qzdiv(stake,Delta_up,Xi_up,UUU,VVV)

    
    try:
        # check that all unused roots are unstable
        assert abs(Xi_sortval[m_states]) > (1-TOL)
        # check that all used roots are stable
        assert abs(Xi_sortval[Xi_select]).max() < 1+TOL
    except:
        raise BKError('generic')


    
    # check for unit roots anywhere

#    assert (abs((abs(Xi_sortval) - 1)) > TOL).all()

    Lambda_mat = diagflat(Xi_sortval[Xi_select])
    VVVH = VVV.H
    VVV_2_1 = VVVH[m_states:2*m_states, :m_states]
    VVV_2_2 = VVVH[m_states:2*m_states, m_states:2*m_states]
    UUU_2_1 = UUU[m_states:2*m_states, :m_states]

    PP = - solve(VVV_2_1, VVV_2_2)
    
    # slightly different check than in the original toolkit:
    assert allclose(real_if_close(PP), PP.real)
    PP = PP.real
    ## end of solve_qz!

    return [Xi_sortval[Xi_select],PP.A]
    
def solve_sylvester(A,B,C,D,Ainv = None):
    # Solves equation : A X + B X [C,...,C] + D = 0
    # where X is a multilinear function whose dimension is determined by D
    # inverse of A can be optionally specified as an argument

    import slycot
        
    n_d = D.ndim - 1
    n_v = C.shape[1]

    n_c = D.size/n_v**n_d


    import dolo.config
    opts = dolo.config.use_engine

    if opts['sylvester']:
        DD = D.flatten().reshape( n_c, n_v**n_d)
        [err,XX] = dolo.config.engine.engine.feval(2,'gensylv',n_d,A,B,C,-DD)
        X = XX.reshape( (n_c,)+(n_v,)*(n_d))

    else:
        DD = D.reshape( n_c, n_v**n_d )

        if n_d == 1:
            CC = C
        else:
            CC = np.kron(C,C)
        for i in range(n_d-2):
            CC = np.kron(CC,C)
    
        if Ainv != None:
            Q = sdot(Ainv,B)
            S = sdot(Ainv,DD)
        else:
            Q = np.linalg.solve(A,B)
            S = np.linalg.solve(A,DD)

        n = n_c
        m = n_v**n_d

        XX = slycot.sb04qd(n,m,Q,CC,-S)

        X = XX.reshape( (n_c,)+(n_v,)*(n_d) )

    return X
    
class BKError(Exception):
    def __init__(self,type):
        self.type = type
    def __str__(self):
        return 'Blanchard-Kahn error ({0})'.format(self.type)
