 # -*- coding: utf-8 -*-

### does this code really depends on Sage ?

import dolo
import numpy as np
from  scipy.linalg.decomp import schur
import time
from dolo.compiler.compiler_dynare import DynareCompiler
sys.path.append( '/home/pablo/Sources/src')
from TensorLib.tensor import tensor

#from mlabwrap import mlab
#mlab.addpath('/home/pablo/Programmation/mini_dynare/')
# all calls to mlabwrap should have been removed now


import scipy.io as matio
import os
import time
#

def compute_steadystate_values(model):
    from dolo.misc.calculus import solve_triangular_system
            
    dvars = dict()
    dvars.update(model.parameters_values)
    dvars.update(model.init_values)
    for v in model.variables:
        if v not in dvars:
            dvars[v] = 0
    undeclared_parameters = []
    for p in model.parameters:
        if p not in dvars:
            undeclared_parameters.append(p)
            dvars[p] = 0
            raise Warning('No initial value for parameters : ' + str.join(', ', [p.name for p in undeclared_parameters]) )

    values = solve_triangular_system(dvars)[0]

    y = [values[v] for v in model.variables]
    x = [0 for s in model.shocks]
    params = [values[v] for v in model.parameters]
    return [y,x,params]

TOL = 1e-10

def second_order_solver(FF,GG,HH):
    from dolo.extern.toolkithelpers import qzdiv
    from dolo.extern.qz import qz
    
    from numpy import mat,c_,r_,eye,zeros,real_if_close,diag,allclose,where,diagflat
    from numpy.linalg import solve
    Psi_mat = mat(FF)
    Gamma_mat = mat(-GG)
    Theta_mat = mat(-HH)
    m_states = FF.shape[0]

    m1 = (np.c_[Gamma_mat, Theta_mat])
    m2 = (np.c_[eye(m_states), np.zeros((m_states, m_states))])
    m1 = np.mat(m1)
    m2 = np.mat(m2)
        
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

    # check that all unused roots are unstable
    assert abs(Xi_sortval[m_states]) > 1-TOL
    
    # check that all used roots are stable
    assert abs(Xi_sortval[Xi_select]).max() < 1+TOL
    
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
        
    #print "solution for PP :"
    #print PP

    return [Xi_sortval[Xi_select],PP.A]

def solve_decision_rule( model, order=2, derivs=None, use_dynare=False):
    
    if order>3:
        raise Exception('Order > 3 not implemented yet')
    
    if derivs == None:
        comp = DynareCompiler(model)
        # let compile the function computing derivatives and residuals
        #f_static = comp.compute_static_pfile(max_order=2)
        f_dynamic = comp.compute_dynamic_pfile(max_order=order)
        # compute steadystate values of the model
        [y,x,parms] = compute_steadystate_values(model)
        y = np.array(y) # y seems to be a list !
        x = np.array(x) # y seems to be a list !
        parms = np.array(parms) # y seems to be a list !
        # compute steady derivatives
        # [gs_0,gs_1,gs_2] = f_static(y,x,parms)
        # compute dynamic derivatives
        yy = [y[i] for i in [model.variables.index(v.P) for v in model.dyn_var_order]]
        xx = np.zeros((1,len(model.shocks)))
        if order == 1:
            [f_0,f_1] = f_dynamic(yy,xx,parms)
        elif order ==2:
            [f_0,f_1,f_2] = f_dynamic(yy,xx,parms)
        if abs(f_0).max() > TOL:
            print("Supplied initial values don't solve the steady state")
    else:
        if order == 1:
            [f_0,f_1] = derivs
        elif order ==2:
            [f_0,f_1,f_2] = derivs
    
    
    if use_dynare:
        #comp = DynareCompiler(model)
        #lli = comp.lead_lag_incidence_matrix()
        #from mlabwrap import mlab
        #mlab.addpath('/home/pablo/Programmation/mini_dynare/')
        #resp = mlab.dr3(f1,f2,0,lli,nout=1)
        print 'Solve for decision rule using Dynare'
        engine.set('f1',f_1)
        engine.set('f2',f_2)
        engine.execute('clear dr')
        engine.eval('''
        dr = struct;
        dr.ys = oo_.steady_state; %dummy
        dr.jacobia_ = f1;
        ss = size(f2);
        f2 = reshape(f2, [ss(1) (ss(2)*ss(3))]);
        f2 = sparse(f2);
        dr.hessian = f2;
        ''')
        print  engine.execute('[dr,info] = drs(dr,0,M_,options_,oo_);')
        info = engine.get('info')
        if info[0].max() != 0:
            print dr1_info_code[float(info[0,0])]
            print info
        engine.eval('''
        dr = reorder_dr(dr);
        n_s = length( find( M_.lead_lag_incidence(1,:) ) );
        dr.ghxx = reshape(dr.ghxx, M_.endo_nbr, n_s, n_s);
        dr.ghxu = reshape(dr.ghxu, M_.endo_nbr, M_.exo_nbr, n_s);
        dr.ghuu = reshape(dr.ghuu, M_.endo_nbr, M_.exo_nbr, M_.exo_nbr);
        ''')

        dr = engine.get('dr')
        g_y = dr['ghx'][0,0]
        g_u = dr['ghu'][0,0]
        g_yy = dr['ghxx'][0,0]
        g_yu = dr['ghxu'][0,0]
        # this is done because of Fortran/C order conflicts
        g_yu = g_yu.swapaxes(1,2)

        g_uu = dr['ghuu'][0,0]
        g_ss = dr['ghs2'][0,0]
        return [[g_y,g_u],[g_yy,g_yu,g_uu],g_ss]
        
   
    
    n_v = model.info['n_variables']
    n_s = model.info['n_shocks']
    dvo = model.dyn_var_order
    n_dvo = len(dvo)
    
    ## solve for first order using uhligs toolkit
    
    # Construction of f_d, f_a, f_h, f_u
    f_d = np.zeros((n_v,n_v))
    f_a = np.zeros((n_v,n_v))
    f_h = np.zeros((n_v,n_v))
    f_u = np.zeros((n_v,n_s))
    O = np.zeros((n_v,n_v))
    I = np.eye(n_v)
    for i in range(n_v):
        v = model.variables[i]
        if v(-1) in model.dyn_var_order:
            j = dvo.index( v(-1) )
            f_h[:,i] = f_1[:,j]
        if v in model.dyn_var_order:
            j = dvo.index( v )
            f_a[:,i] = f_1[:,j]
        if v(1) in model.dyn_var_order:
            j = dvo.index( v(1) )
            f_d[:,i] = f_1[:,j]
    for i in range( n_s ):
        f_u[:,i] = f_1[:,n_dvo + i]
        
    fut_variables = [v for v in model.variables if v(1) in model.dyn_var_order]
    cur_variables = [v for v in model.variables if v in model.dyn_var_order]
    pred_variables = [v for v in model.variables if v(-1) in model.dyn_var_order]
    
    fut_ind = [model.variables.index(i) for i in fut_variables]
    cur_ind = [model.variables.index(i) for i in cur_variables]
    pred_ind = [model.variables.index(i) for i in pred_variables]
    fut_ind_d = [model.dyn_var_order.index(i(1)) for i in fut_variables]
    cur_ind_d = [model.dyn_var_order.index(i) for i in cur_variables]
    pred_ind_d = [model.dyn_var_order.index(i(-1)) for i in pred_variables]
    
    #from dolo.extern.toolkithelpers import qzdiv
    #from dolo.extern.qz import qz
    [ev,g_y] = second_order_solver(f_d,f_a,f_h)

    if ( max( np.linalg.eigvals(g_y) )  > 1 + TOL):
        raise Exception( 'BK conditions not satisfied' )
    
    mm = np.dot(f_d, g_y) + f_a
    g_u = - np.linalg.solve( mm , f_u )

    if order == 1:
        return [g_y,g_u]
    
    S = np.concatenate( [np.dot(g_y,g_y),g_y,np.eye(n_v)] )
    gg_1 = np.zeros( (n_v , n_v*3))
    gg_2 = np.zeros( (n_v , n_v*3, n_v*3) )
    full_dvo_i = [n_v*2 + i for i in pred_ind] + [i + n_v for i in cur_ind]  + [i for i in fut_ind]
    
    for i in range(n_dvo):
        gg_1[:,full_dvo_i[i]] = f_1[:, i]
        for j in range(n_dvo):
            gg_2[:, full_dvo_i[i], full_dvo_i[j]] = f_2[:,i,j]
            
    # let compute the constant term of the sylvester equations
    Sm = S
    gg_2_t = tensor(gg_2)
    D_t = gg_2_t.ttm([Sm,Sm],dims=(1,2),option='t')
    D = D_t.tondarray()

    A = np.dot(f_d,g_y) + f_a
    B = f_d
    C = g_y

    CC = np.kron(C,C)
    DD = -D.reshape( (n_v,n_v*n_v))

    g_yy = solve_sylvester( A,B,C,D, insist=False)
    
    # it is now easy to 

    F = np.zeros( ( len(fut_ind),n_v) )

    for i in range(len(fut_ind)):
        F[i,fut_ind[i]] = 1
    P = np.zeros( ( len(pred_ind),n_v ) )
    for i in range(len(pred_ind)):
        P[i,pred_ind[i]] = 1

    n_pred_v = len( pred_ind )
    S_y = np.concatenate( [P, g_y,  np.dot(F,np.dot(g_y,g_y)),  np.zeros((n_s,n_v))] )
    S_u = np.concatenate( [np.zeros((n_pred_v,n_s)), g_u, np.dot(F,np.dot(g_y,g_u)), np.eye(n_s)] )

    A = np.dot( f_d, g_y ) + f_a
    A_inv = -np.linalg.inv( A )

    res_yu = np.tensordot( f_d, tensor(g_yy).ttm([g_y,g_u],dims=(1,2),option='t').tondarray() , axes=(1,0) )
    K_yu = tensor(f_2).ttm([S_y,S_u],dims=(1,2),option='t').tondarray()
    g_yu = np.tensordot( A_inv , res_yu + K_yu, axes=(1,0))

    res_uu = np.tensordot( f_d, tensor(g_yy).ttm([g_u,g_u],dims=(1,2),option='t').tondarray() , axes=(1,0) )
    K_uu = tensor(f_2).ttm([S_u,S_u],dims=(1,2),option='t').tondarray()
    g_uu = np.tensordot( A_inv , res_uu + K_uu, axes=(1,0))

    As = A + f_d
    As_inv = - np.linalg.inv(As)
    S_u_s = np.concatenate( [ np.zeros((n_pred_v,n_s)), np.zeros((n_v,n_s)), np.dot(F,g_u) , np.eye(n_s)] )
    K_ss = tensor(f_2).ttm([S_u_s,S_u_s],dims=(1,2),option='t').tondarray()
    K_ss = K_ss + np.tensordot( f_d, g_uu, axes=(1,0) )
    g_ss = np.tensordot( As_inv , K_ss, axes=(1,0) )


    svi = [model.variables.index(v) for v in  model.state_variables]
    g_y = g_y[:,svi]
    g_yy = g_yy[:,svi,:]
    g_yy = g_yy[:,:,svi]
    g_yu = g_yu[:,svi,:]

    if derivs == None:
        return DDR( [y,[g_y,g_u],[g_yy,g_yu,g_uu]], correc_s = g_ss )
    else:
        return [[g_y,g_u],[g_yy,g_yu,g_uu],g_ss]
        


    #return DR( [,[g_y,g_u],[g_yy,g_yu,g_uu]] , correc_s = g_ss)

def multidot(ten,mats):
    '''
    Implements tensor operation : tensor-times-matrices.
    If last dimensions of ten represent multilinear operations of the type : [X1,...,Xk]->B[X1,...,Xk]
    and mats contains matrices or vectors [A1,...Ak] the function returns an array representing operators : 
    [X1,...,Xk]->B[A1 X1,...,Ak Xk]
    '''
    resp = ten
    n_d = ten.ndim
    n_m = len(mats)
    for i in range(n_m):
        print n_d -1 -i
        resp = np.tensordot( resp, mats[i], (n_d-n_m+i-1,0) )
    return resp

def solve_sylvester(A,B,C,D,insist=False):
    # Solves equation : A X + B X [C,...,C] + D = 0
    # where X is a multilinear function whose dimension is determined by D
    n_d = D.ndim - 1
    n_v = C.shape[1]
    CC = np.kron(C,C)
    for i in range(n_d-2):
        CC = np.kron(CC,C)
    DD = D.reshape( n_v, n_v**n_d )

    Q = np.linalg.solve(A,B)
    R = CC
    S = np.linalg.solve(A,DD)
    
    # X must be solution of :  X + Q X R = S
    tt = np.kron( R.T , Q )
    I = np.eye(tt.shape[0])
    vec_S = S.flatten(1)
    vec_X = np.linalg.solve( I + tt, - vec_S )
    XX = vec_X.reshape( n_v, n_v**n_d, order='F' )
    
    # we can check against dynare routines
    #XX = mlab.sylvester3(A,B,CC,-DD)
    #XX = mlab.sylvester3a(XX,A,B,CC,-DD)
    #print abs(my_XX - XX).max()

    return XX.reshape( (n_v,)*(n_d+1) )
    

class DDR():
    # this class represent a dynare decision rule
    def __init__(self,g,ghs2=None,correc_s=None):
        self.g = g
        if correc_s !=None:
            self.correc_s = correc_s
        if ghs2 !=None:
            self.ghs2 = ghs2

    @property
    def ys(self):
        return self.g[0]

    @property
    def ghx(self):
        return self.g[1][0]

    @property
    def ghu(self):
        return self.g[1][1]

    @property        
    def ghxx(self):
        return self.g[2][0]

    @property        
    def ghxu(self):
        return self.g[2][1]

    @property        
    def ghuu(self):
        return self.g[2][2]
        
    @property        
    def ghs2(self):
        return self.ghs2
     
     
    def ys_c(self,Sigma_e):
        return self.g[0] + 0.5*np.tensordot( self.correc_s , Sigma_e, axes = ((1,2),(0,1)) )

    def __call__(self, x, u, Sigma_e):
        # evaluates y_t, given y_{t-1} and e_t
        resp = self.ys + np.dot( self.ghx, x ).flatten() +  np.dot( self.ghu, u ).flatten()
        resp += 0.5*np.tensordot( self.ghxx, np.outer(x,x) ) 
        resp += 0.5*np.tensordot( self.ghxu, np.outer(x,u) )
        resp += 0.5*np.tensordot( self.ghuu, np.outer(u,u) )
        resp += 0.5*self.ghs2
        return resp
          
    def __str__(self):
        return 'Decision rule'


def retrieve_DDR_from_matlab(name,mlab):
    mlab.execute( 'drn = reorder_dr({0});'.format(name) )
    rdr = retrieve_from_matlab('drn',mlab)
    ys =  rdr['ys'][0,0].flatten()
    ghx = rdr['ghx'][0,0]
    ghu = rdr['ghu'][0,0]
    [n_v,n_states] = ghx.shape
    n_shocks = ghu.shape[1]
    ghxx = rdr['ghxx'][0,0].reshape( (n_v,n_states,n_states) )
    ghxu = rdr['ghxu'][0,0].reshape( (n_v,n_states,n_shocks) )
    ghuu = rdr['ghuu'][0,0].reshape( (n_v,n_shocks,n_shocks) )
    ghs2 = rdr['ghs2'][0,0].flatten()
    ddr = DDR( [ ys,[ghx,ghu],[ghxx,ghxu,ghuu] ] , ghs2 = ghs2 )
    return [ddr,rdr]


dr1_info_code = {
1: "the model doesn't define current variables uniquely.",
2: "problem in mjdgges.dll info(2) contains error code.",
3: "BK order condition not satisfied info(2) contains 'distance' absence of stable trajectory.",
4: "BK order condition not satisfied info(2) contains 'distance' indeterminacy.",
5: "BK rank condition not satisfied.",
6: "The jacobian matrix evaluated at the steady state is complex."
}