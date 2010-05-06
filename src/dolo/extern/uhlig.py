# This port was initially done by Sven Schreiber

"""
start porting the toolkit to numpy/scipy
"""
from helpers import rank, unvec, null
from toolkithelpers import qzswitch, qzdiv, octave4numpy
from qz import qz
from numpy.linalg import pinv, solve, det
from numpy.matlib import (eye, ones, zeros, diag, diagflat,
        allclose, real_if_close, kron, r_, c_, isinf, mat, where)
#from scipy import *

class UhligToolkit:
    def __init__(self, FF, GG, HH, LL, MM, NN,  varnames, TOL = 1e-10,bruteforce=True,print_matrices=True):
        
        if bruteforce == False:
            raise "Only bruteforce is implemented. Don't think so much."
    
        self.NN = NN
        if print_matrices:
            print('FF',FF)
            print('GG',GG)
            print('HH',HH)
            print('LL',LL)
            print('MM',MM)
            print('NN',NN)
#        l_equ = 0
        m_states = FF.shape[0]
        n_endog = 0
        k_exog = MM.shape[1]
#        
#        # message assignment in do_it
#        # warnings assignment in do_it
#        # options.m call in do_it; try to get by without first
#
#        ## the solve call:
#        q_expect_equ = FF.shape[0]
#        assert m_states == FF.shape[1]
#        k_exog = min(NN.shape)
#        # (skip the complex number stuff)

        m_states = FF.shape[1]
        
        Psi_mat = mat(FF)
        Gamma_mat = mat(-GG)
        Theta_mat = mat(-HH)

        m1 = (c_[Gamma_mat, Theta_mat])
        m2 = (c_[eye(m_states), zeros((m_states, m_states))])
        m1 = mat(m1)
        m2 = mat(m2)
        
        # there is an error in the initial code : zeros((m_states,1)) instead of zeros((m_states, m_states))

        #Xi_mat = r_[c_[Gamma_mat, Theta_mat], # why is it transposed ?
        #            c_[eye(m_states), zeros((m_states, m_states))]]

        #print('Gamma_mat :')
        #print(Gamma_mat)
        
        #print('Psi_mat :')
        #print(Psi_mat)
        
        #print('Theta_mat :')
        #print(Theta_mat)
    
        Xi_mat = r_[c_[Gamma_mat, Theta_mat],
                    c_[eye(m_states), zeros((m_states, m_states))]]
        #print('Xi_mat :')
        #print(Xi_mat)
        
        Delta_mat = r_[c_[Psi_mat, zeros((m_states, m_states))], 
                       c_[zeros((m_states, m_states)), eye(m_states)]]
        #print('Delta_mat :')
        #print(Delta_mat)
      
        ## end general solve call

        ## proceed with solve_qz call
        # Matlab and octave seem to return triangular AA and BB (aka S and T).
        # Octave in addition returns the eigenvalues as 7th output (which may
        #  be theoretically infinite?).
        # Definitions of outputs differ slightly, so need to (conjugate-)
        #  transpose Q:
        #  matlab: AA = Q*A*Z,      BB = Q*B*Z,     (A*V = B*V*diag(lda))
        #  octave: AA = Q.H*A*Z,    BB = Q.H*B*Z,   (dito)
        # (note that in this script AA is sth. different...)
        
        AAA,BBB,Q,Z = qz(Delta_mat, Xi_mat)
        
        # convert to toolkit notation
        # old line:
        # Delta_up,Xi_up,UUU,VVV,Xi_eigvec = A2, B2, QH.H, Z, V
        Delta_up,Xi_up,UUU,VVV = map(real_if_close, (AAA,BBB,Q,Z))
        
        # to do: treatment of complex eigenvalues

        #assert allclose(real_if_close(Delta_up), Delta_up.real)
        #assert allclose(real_if_close(Xi_up), Xi_up.real)
        
        Xi_eigval = diag(Xi_up)/where(diag(Delta_up)>TOL, diag(Delta_up), TOL)
        #print Xi_eigval, m_states
        Xi_sortindex = abs(Xi_eigval).argsort()
        # (Xi_sortabs doesn't really seem to be needed)
        Xi_sortval = Xi_eigval[Xi_sortindex]
        Xi_select = slice(0, m_states)
        stake = (abs(Xi_sortval[Xi_select])).max() + TOL
        Delta_up,Xi_up,UUU,VVV = qzdiv(stake,Delta_up,Xi_up,UUU,VVV)

        # check that all unused roots are unstable
        assert abs(Xi_sortval[m_states]) > 1-TOL
        # check that all used roots are stable
        assert abs(Xi_sortval[Xi_select]).max() < 1+TOL
        # check for unit roots anywhere
        assert (abs((abs(Xi_sortval) - 1)) > TOL).all()

        Lambda_mat = diagflat(Xi_sortval[Xi_select])
        VVVH = VVV.H
        VVV_2_1 = VVVH[m_states:2*m_states, :m_states]
        VVV_2_2 = VVVH[m_states:2*m_states, m_states:2*m_states]
        UUU_2_1 = UUU[m_states:2*m_states, :m_states]
    
        assert abs(det(UUU_2_1)) > TOL
        assert abs(det(VVV_2_1)) > TOL

        PP = - solve(VVV_2_1, VVV_2_2)
        # slightly different check than in the original toolkit:
        assert allclose(real_if_close(PP), PP.real)
        PP = PP.real
        ## end of solve_qz!
        
        #print "solution for PP :"
        #print PP
        
        #print "checking if PP is really a solution"

        a1 = Psi_mat * (PP * PP)
        a2 = -(Gamma_mat * PP)
        a3 = -(Theta_mat)

        eq = a1 + a2 + a3
        
        RR = mat(0)
        VV = kron(NN.T, FF) + kron(eye(k_exog), FF*PP + GG)

        assert rank(VV) >= k_exog * (m_states + n_endog)
        LLNN_plus_MM = LL*NN + MM
        # fixme: what does (:) in matlab source mean?:
        QQ = - solve(VV, LLNN_plus_MM)
        assert not isinf(QQ).any()
        RR = None
        SS = None
        WW = None

# what is W ?
#        WW = r_[c_[eye(m_states),              zeros((m_states, k_exog))],
#                c_[RR * pinv(PP),              SS - RR * pinv(PP) * QQ], 
#                c_[zeros((k_exog, m_states)),  eye(k_exog)] ]
        ## end calc_qrs!

        ## sol_out just prints stuff, we leave that for the wrapper
        self.Xi_sortval = Xi_sortval
        self.m_states, self.n_endog, self.k_exog = m_states, n_endog, k_exog
        self.PP = PP; self.RR = RR; self.QQ = QQ; self.SS = SS; self.WW = WW ; self.VV = VV
        