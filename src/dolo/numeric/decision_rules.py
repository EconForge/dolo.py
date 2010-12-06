"""
This module contains classes representing decision rules
"""

from dolo.misc.decorators import memoized

import numpy as np

class TaylorExpansion(dict):

    @property
    @memoized
    def order(self):
        order = 0
        if self.get('g_a'):
            order = 1
        if self.get('g_aa'):
            order = 2
        if self.get('g_aa'):
            order = 3
        return order



class DynareDecisionRule(TaylorExpansion):


    def __init__(self,d,model):
        super(DynareDecisionRule,self).__init__(d)
        self.model = model
        self.dr_var_order_i = [model.variables.index(v) for v in model.dr_var_order]
        self.dr_states_order = [v for v in model.dr_var_order if v in model.state_variables]
        self.dr_states_i = [model.variables.index(v) for v in self.dr_states_order]

    @property
    @memoized
    def ghx(self):
        ghx = self.get('g_a')
        ghx = ghx[self.dr_var_order_i,:]
        ghx = ghx[:,self.dr_states_i]
        return ghx

    @property
    @memoized
    def ghu(self):
        ghu = self.get('g_e')
        ghu = ghu[self.dr_var_order_i,:]
        return ghu

    @property
    @memoized
    def ghxx(self):
        ghxx = self.get('g_aa')
        ghxx = ghxx[self.dr_var_order_i,:]
        ghxx = ghxx[:,self.dr_states_i,:]
        ghxx = ghxx[:,:,self.dr_states_i]
        n_v = ghxx.shape[0]
        n_s =  ghxx.shape[1]
        return ghxx.reshape( (n_v,n_s*n_s) )

    @property
    @memoized
    def ghxu(self):
        ghxu = self.get('g_ae')
        ghxu = ghxu[self.dr_var_order_i,:,:]
        ghxu = ghxu[:,self.dr_states_i,:]
        n_v = ghxu.shape[0]
        n_s = ghxu.shape[1]
        n_e = ghxu.shape[2]
        return ghxu.reshape( (n_v,n_s*n_e ))

    @property
    @memoized
    def ghuu(self):
        ghuu = self.get('g_ee')
        ghuu = ghuu[self.dr_var_order_i,:]
        n_v = ghuu.shape[0]
        n_e = ghuu.shape[1]
        return ghuu.reshape( (n_v,n_e*n_e) )

    @property
    @memoized
    def g_1(self):
        return np.column_stack([self.ghx,self.ghu])

    @property
    @memoized
    def g_2(self):
        n_v = self['g_a'].shape[0]
        n_s = len(self.dr_states_order)
        n_e = self['g_e'].shape[1]
        g_2 = np.zeros( (n_v,n_s+n_e,n_s+n_e) )


        g_2[:,n_s:,n_s:] = self.ghuu.reshape((n_v,n_e,n_e))
        g_2[:,:n_s,n_s:] = self.ghxu.reshape((n_v,n_s,n_e))
        g_2[:,n_s:,:n_s] = self.ghxu.reshape((n_v,n_s,n_e)).swapaxes(1,2)
        g_2[:,:n_s,:n_s] = self.ghxx.reshape((n_v,n_s,n_s))
        return fold( g_2 ) / 2

    @property
    @memoized
    def ghxxx(self):
        ghxxx = self['g_aaa'][self.dr_var_order_i,...]
        ghxxx = ghxxx[:,self.dr_states_i,:,:]
        ghxxx = ghxxx[:,:,self.dr_states_i,:]
        ghxxx = ghxxx[:,:,:,self.dr_states_i]
        return ghxxx

    @property
    @memoized
    def ghxxu(self):
        ghxxu = self['g_aae'][self.dr_var_order_i,...]
        ghxxu = ghxxu[:,self.dr_states_i,:,:]
        ghxxu = ghxxu[:,:,self.dr_states_i,:]
        return ghxxu

    @property
    @memoized
    def ghxuu(self):
        ghxuu = self['g_aee'][self.dr_var_order_i,...]
        ghxuu = ghxuu[:,self.dr_states_i,:,:]
        return ghxuu

    @property
    @memoized
    def ghuuu(self):
        ghuuu = self['g_eee'][self.dr_var_order_i,...]
        return ghuuu

    @property
    @memoized
    def g_3(self):
        n_v = self['g_a'].shape[0]
        n_s = len(self.dr_states_order)
        n_e = self['g_e'].shape[1]

        ghxxx = self.ghxxx
        ghxxu = self.ghxxu
        ghxuu = self.ghxuu
        ghuuu = self.ghuuu
        
        g_3 = np.zeros( (n_v,n_s+n_e,n_s+n_e,n_s+n_e))

        g_3[:,:n_s,:n_s,:n_s] = ghxxx

        g_3[:,:n_s,:n_s,n_s:] = ghxxu
        g_3[:,:n_s,n_s:,:n_s] = ghxxu.swapaxes(2,3)
        g_3[:,n_s:,:n_s,:n_s] = ghxxu.swapaxes(1,3)

        g_3[:, :n_s, n_s:, n_s:] = ghxuu
        g_3[:, n_s:, :n_s, n_s:] = ghxuu.swapaxes(1,2)
        g_3[:, n_s:, n_s:, :n_s] = ghxuu.swapaxes(1,3)

        g_3[:,n_s:,n_s:,n_s:] = ghuuu

        g_3 = symmetrize(g_3)

        return fold( g_3 ) / 2 / 3


def symmetrize(tens):
    return (tens + tens.swapaxes(3,2) + tens.swapaxes(1,2) + tens.swapaxes(1,2).swapaxes(2,3) + tens.swapaxes(1,3) + tens.swapaxes(1,3).swapaxes(2,3) )/6

def fold(tens):
    from itertools import product

    if tens.ndim == 3:
        n = tens.shape[0]
        q = tens.shape[1]
        assert( tens.shape[2] == q )

        non_decreasing_pairs = [ (i,j) for (i,j) in product(range(q),range(q)) if i<=j ]
        result = np.zeros( (n,len(non_decreasing_pairs) ) )
        for k,(i,j) in enumerate(non_decreasing_pairs):
            result[:,k] = tens[:,i,j]
        return result

    elif tens.ndim == 4:
        n = tens.shape[0]
        q = tens.shape[1]
        assert( tens.shape[2] == q )
        assert( tens.shape[3] == q)

        non_decreasing_tuples = [ (i,j,k) for (i,j,k) in product(range(q),range(q),range(q)) if i<=j<=k ]
        result = np.zeros( (n,len(non_decreasing_tuples) ) )
        for l,(i,j,k) in enumerate(non_decreasing_tuples):
            result[:,l] = tens[:,i,j,k]
        return result

class DDR():
# this class represent a dynare decision rule
    def __init__(self,g,ghs2=None):
        self.g = g
        if ghs2 !=None:
            self.ghs2 = ghs2
        # I should do something with Sigma_e

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



    #def ghs2(self,Sigma_e):
    #    return np.tensordot( self.correc_s , Sigma_e )/2

    def ys_c(self,Sigma_e):
        return self.g[0] + 0.5*self.ghs2

    def __call__(self, x, u, Sigma_e):
    # evaluates y_t, given y_{t-1} and e_t
        resp = self.ys + np.dot( self.ghx, x ).flatten() +  np.dot( self.ghu, u ).flatten()
        resp += 0.5*np.tensordot( self.ghxx, np.outer(x,x) )
        resp += 0.5*np.tensordot( self.ghxu, np.outer(x,u) )
        resp += 0.5*np.tensordot( self.ghuu, np.outer(u,u) )
        resp += 0.5*self.ghs2(Sigma_e)
        return resp

    def __str__(self):
        return 'Decision rule'

