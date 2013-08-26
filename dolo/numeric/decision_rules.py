"""
This module contains classes representing decision rules
"""

import dolo.config

from dolo.misc.caching import memoized

import numpy as np

class TaylorExpansion(dict):

    @property
    @memoized
    def order(self):
        order = 0
        if self.get('g_a') is not None:
            order = 1
        if self.get('g_aa') is not None:
            order = 2
        if self.get('g_aaa') is not None:
            order = 3
        return order



class DynareDecisionRule(TaylorExpansion):


    def __init__(self,d,model):
        super(DynareDecisionRule,self).__init__(d)
        self.model = model
        self.dr_var_order_i = [model.variables.index(v) for v in self.dr_var_order]
        self.dr_states_order = [v for v in self.dr_var_order if v in self.state_variables]
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
    def ghs2(self):
        ghs2 = self.get('g_ss')
        ghs2 = ghs2[self.dr_var_order_i]
        return ghs2


    @property
    @memoized
    def g_1(self):
        
        g_1x = self.ghx
        g_1u = self.ghu

        if 'g_ass' in self:
            correc_x_ss =  self['g_ass'][self.dr_var_order_i,:][:,self.dr_states_i]/2
            correc_u_ss =  self['g_ess'][self.dr_var_order_i,:]/2
            g_1x += correc_x_ss
            g_1u += correc_u_ss

        return np.column_stack([g_1x,g_1u])

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


    def __str__(self):
        txt = '''
Decision rule (order {order}) :
{msg}
    - States : {states}
    
    - Endogenous variables : {endo}

    - First order coefficients :

{foc}
'''
        mat = np.concatenate([self.ghx,self.ghu],axis=1)
        if self.order > 1:
            msg = '\n    (Only first order derivatives are printed)\n'
        else:
            msg = ''
        col_names = [str(v(-1)) for v in self.model.dr_var_order if v in self.model.state_variables] + [str(s) for s in self.model.shocks]
        row_names = [str(v) for v in self.model.dr_var_order]
        txt = txt.format(
            msg = msg,
            order=self.order,
            states=str.join(' ', col_names),
            endo=str.join(' ', row_names),
            foc=mat
        )
        return txt

    @memoized
    def risky_ss(self):
        oo = self.order
        if oo == 1:
            return self['ys']
        elif oo in (2,3):
            #TODO: RSS for order 3 should be computed differently
            import numpy.linalg      
            A = self['g_a']
            I = np.eye(A.shape[0])
            D = self['g_ss']/2
            dx = numpy.linalg.solve( I - A, D)
            return self['ys'] + dx
        
    def gap_to_risky_steady_state(self,x):
        from dolo.numeric.tensor import mdot
        d = x - self['ys']
        res = self['ys'] - x
        res += np.dot( self['g_a'],d )
        res += mdot(self['g_aa'],[d,d])
        res += mdot(self['g_aaa'],[d,d,d])
        res += self['g_ss']/2
        res += np.dot(self['g_ass'],d)/2
        return res

    def __call__(self,x,e,order=None):
        from dolo.numeric.tensor import mdot
        if order is None:
            order = self.order
        d = x - self['ys']
        res = self['ys'] + np.dot( self['g_a'], d )
        res += np.dot( self['g_e'], e )
        if ('g_aa' in self) or (order >=2):
            res += mdot(self['g_aa'],[d,d])/2
            res += mdot(self['g_ae'],[d,e])
            res += mdot(self['g_ee'],[e,e])/2
            res += self['g_ss']/2
        if ('g_aaa' in self) or (order >=3):
            res += mdot(self['g_aaa'],[d,d,d])/6
            res += mdot(self['g_aae'],[d,d,e])/3
            res += mdot(self['g_aee'],[d,e,e])/3
            res += mdot(self['g_eee'],[e,e,e])/6
            res += np.dot(self['g_ass'],d)/2
            res += np.dot(self['g_ess'],e)/2
        return res

    @property
    def dyn_var_order(this):
        self = this.model
        # returns a list of dynamic variables ordered as in Dynare's dynamic function
        if hasattr(self,'__dyn_var_order__') :
            return self.__dyn_var_order__
        d = dict()
        for eq in self.equations:
            all_vars = eq.variables
            for v in all_vars:
                if not v.lag in d:
                    d[v.lag] = set()
                d[v.lag].add(v)
        maximum = max(d.keys())
        minimum = min(d.keys())
        ord = []
        for i in range(minimum,maximum+1):
            if i in d.keys():
                ord += [v(i) for v in self.variables if v(i) in d[i]]
        self.__dyn_var_order__ = ord
        return ord

    @property
    def state_variables(self):
        return [v for v in self.model.variables if v(-1) in self.dyn_var_order ]

    @property
    def dr_var_order(self):
        dvo = self.dyn_var_order
        purely_backward_vars = [v for v in self.model.variables if (v(1) not in dvo) and (v(-1) in dvo)]
        purely_forward_vars = [v for v in self.model.variables if (v(-1) not in dvo) and (v(1) in dvo)]
        static_vars =  [v for v in self.model.variables if (v(-1) not in dvo) and (v(1) not in dvo) ]
        mixed_vars = [v for v in self.model.variables if not v in purely_backward_vars+purely_forward_vars+static_vars ]
        dr_order = static_vars + purely_backward_vars + mixed_vars + purely_forward_vars
        return dr_order

DecisionRule = DynareDecisionRule

def theoretical_moments(dr,with_correlations=True):
    maxit = 1000
    tol = 0.00001
    A = dr['g_a']
    B = dr['g_e']
    Sigma = dr['Sigma']
    M0 = np.dot(B,np.dot(Sigma,B.T))
    M1 = M0
    for i in range( maxit ):
        M = M0 + np.dot( A, np.dot( M1, A.T ) )
        if abs( M - M1).max() < tol:
            break
        M1 = M
    if not with_correlations:
        return M
    else:
        cov = M
        d = np.diag( 1/np.sqrt( np.diag(cov) ) )
        correl = np.dot(d, np.dot(cov,d.T) )
        return [M,correl]



    

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



def impulse_response_function(decision_rule, shock, variables = None, horizon=40, order=1, output='deviations', plot=True):
    
    if order > 1:
        raise Exception('irfs, for order > 1 not implemented')
    
    dr = decision_rule
    A = dr['g_a']
    B = dr['g_e']
    Sigma = dr['Sigma']

    [n_v, n_s] = [ len(dr.model.variables), len(dr.model.shocks) ]


    if isinstance(shock,int):
        i_s = shock
    elif isinstance(shock,str):
        from dolo.symbolic.symbolic import Shock
        shock =  Shock(shock)
        i_s = dr.model.shocks.index( shock )
    else:
        i_s = shock

    E0 = np.zeros(  n_s )
    E0[i_s] = np.sqrt(dr['Sigma'][i_s,i_s])

    E = E0*0

    RSS = dr.risky_ss()

    simul = np.zeros( (n_v, horizon+1) )

    start = dr( RSS, E0 )
    simul[:,0] = start
    for i in range(horizon):
        simul[:,i+1] = dr( simul[:,i], E )

    # TODO: change the correction so that it corresponds to the risky steady-state
    constant = np.tile(RSS, ( horizon+1, 1) ).T
    if output == 'deviations':
        simul = simul - constant
    elif output == 'logs':
        simul = np.log((simul-constant)/constant)
    elif output == 'percentages':
        simul = (simul-constant)/constant*100
    elif output == 'levels':
        pass
    else:
        raise Exception("Unknown output type")


    if variables:
        from dolo.symbolic.symbolic import Variable
        if not isinstance(variables,list):
            variables = [variables]
        variables =  [Variable(str(v)) for v in variables]
        ind_vars = [dr.model.variables.index( v ) for v in variables]
        simul = simul[ind_vars, :]

    x = np.linspace(0,horizon,horizon+1)

    if plot:
        from matplotlib import pylab
        pylab.clf()
        pylab.title('Impulse-Response Function for ${var}$'.format(var=shock.__latex__()))
        for k in range(len(variables)):
            pylab.plot(x[1:], simul[k,1:],label='$'+variables[k]._latex_()+'$' )
        pylab.plot(x,x*0,'--',color='black')
        pylab.xlabel('$t$')
        if output == 'percentages':
            pylab.ylabel('% deviations from the steady-state')
        elif output == 'deviations':
            pylab.ylabel('Deviations from the steady-state')
        elif output == 'levels':
            pylab.ylabel('Levels')
        pylab.legend()
        if dolo.config.save_plots:
            filename = 'irf_' + str(shock) + '__' + '_' + str.join('_',[str(v) for v in variables])
            pylab.savefig(filename) # not good...
        else:
            pylab.show()

    return simul

def stoch_simul(decision_rule, variables = None,  horizon=40, order=None, start=None, output='deviations', plot=True, seed=None):

#    if order > 1:
#        raise Exception('irfs, for order > 1 not implemented')

    dr = decision_rule

    [n_v, n_s] = [ len(dr.model.variables), len(dr.model.shocks) ]

    Sigma = dr['Sigma']
    if seed:
        np.random.seed(seed)

    E = np.random.multivariate_normal((0,)*n_s,Sigma,horizon)
    E = E.T

    simul = np.zeros( (n_v, horizon+1) )
    RSS = dr.risky_ss()
    if start is None:
        start = RSS
        
    if not order:
        order = dr.order

    simul[:,0] = start
    for i in range(horizon):
        simul[:,i+1] = dr( simul[:,i], E[:,i] )

    # TODO: change the correction so that it corresponds to the risky steady-state
    constant = np.tile(RSS, ( horizon+1, 1) ).T
    if output == 'deviations':
        simul = simul - constant
    elif output == 'logs':
        simul = np.log((simul-constant)/constant)
    elif output == 'percentages':
        simul = (simul-constant)/constant*100
    elif output == 'levels':
        pass
    else:
        raise Exception("Unknown output type")

    if variables:
        from dolo.symbolic.symbolic import Variable
        if not isinstance(variables,list):
            variables = [variables]
        variables =  [Variable(str(v)) for v in variables]
        ind_vars = [dr.model.variables.index( v ) for v in variables]
        simul = simul[ind_vars, :]

    x = np.linspace(0,horizon,horizon+1)

    if plot:
        from matplotlib import pylab
        pylab.clf()
        pylab.title('Stochastic simulation')
        for k in range(len(variables)):
            pylab.plot(x[1:], simul[k,1:],label='$'+variables[k]._latex_()+'$' )
        pylab.plot(x,x*0,'--',color='black')
        pylab.xlabel('$t$')
        if output == 'percentages':
            pylab.ylabel('% deviations from the steady-state')
        elif output == 'deviations':
            pylab.ylabel('Deviations from the steady-state')
        elif output == 'levels':
            pylab.ylabel('Levels')
        pylab.legend()
        if dolo.config.save_plots:
            filename = 'simul_' + '_' + str.join('_',[str(v) for v in variables])
            pylab.savefig(filename) # not good...
        else:
            pylab.show()

    return simul
