from dolo.compiler.compiler import *
from sympy import Matrix

class UhligCompiler(Compiler):
    '''
    Uhlig toolkit assumes that model is written in the linear form :
       E_t  F x_{t+1} + G x_t + H x_{t-1} + L z_{t+1} + M z_t
       z_{t+1} = N z_t + eps_{t+1}
    Dolo uses a slighlty different convention in order to make it a particular
    case of Dynare formulation :
       E_t  F x_{t+1} + G x_t + H x_{t-1} + L z_{t+1} + M z_t
       z_{t} = N z_{t-1} + eps_{t}
    An Uhlig model can either be directly specified in python or converted from
    a Dynare model with command dynare_to_uhlig(model)
    One doesn't need to specify which are the exogenous variables or exogenous
    equations as they are defined by the appearance of shocks.
    Calling model.check() will ensure an UhligModel permits to define the
    matrices F,G,H,L,M,N,eps.
    '''
    
    def write_mfile_for_matrices(self):
        model = self.model
        d = self.compute_formal_matrices()
        mfile = 'function [F,G,H,L,M,N] = %s(y, x, parms)\n'
        mfile = mfile %( model.fname + '_uhlig')
        for k in d.keys():
            p = d[k].shape[0]
            q = d[k].shape[1]
            mfile += '%s = zeros(%s,%s);\n' %(k,p,q)
            for i in range(p):
                for j in range(q):
                    expr = d[k][i,j]
                    expr_txt = str(self.tabify_expression(expr,for_matlab=True))
                    expr_txt = expr_txt.replace('**','^')
                    mfile += '%s(%s,%s) = %s;\n' %(k,i+1,j+1,expr_txt)
        f = file(model.fname + '_uhlig.m','w')
        f.write(mfile)
        f.close()

    def compute_formal_matrices(self):
        model = self.model
        exo_eqs = [eq for eq in model.equations if eq.info.get('exogenous') == 'true']
        non_exo_eqs = [eq for eq in model.equations if not eq in exo_eqs]
        exo_vars = [eq.lhs for eq in exo_eqs]
        non_exo_vars = [v for v in model.variables if not v in exo_vars]
        model.info['exo_vars'] = exo_vars
        model.info['non_exo_vars'] = non_exo_vars

        mat_exo_vars_f = Matrix([v(+1) for v in exo_vars]).T
        mat_exo_vars_c = Matrix([v for v in exo_vars]).T
        mat_exo_vars_p = Matrix([v(-1) for v in exo_vars]).T

        mat_non_exo_vars_f = Matrix( [v(+1) for v in non_exo_vars] ).T
        mat_non_exo_vars_c = Matrix( [v for v in non_exo_vars] ).T
        mat_non_exo_vars_p = Matrix( [v(-1) for v in non_exo_vars] ).T

        # Compute matrix for exogenous equations
        mat_exo_rhs = Matrix([eq.rhs for eq in exo_eqs]).T
        N = mat_exo_rhs.jacobian(mat_exo_vars_p).T

        # Compute matrices for non exogenous equations

        mat_non_exo_eqs = Matrix( [ eq.gap for eq in non_exo_eqs ] ).T
        F = mat_non_exo_eqs.jacobian(mat_non_exo_vars_f).T
        G = mat_non_exo_eqs.jacobian(mat_non_exo_vars_c).T
        H = mat_non_exo_eqs.jacobian(mat_non_exo_vars_p).T
        L = mat_non_exo_eqs.jacobian(mat_exo_vars_f).T
        M = mat_non_exo_eqs.jacobian(mat_exo_vars_c).T

        def steady_state_ify(m):
            # replaces all variables in m by steady state value
            for v in model.variables + model.shocks: # slow and inefficient
                m = m.subs(v(+1),v.P)
                m = m.subs(v(-1),v.P)
            return m

        d = dict()
        d['F'] = steady_state_ify(F)
        d['G'] = steady_state_ify(G)
        d['H'] = steady_state_ify(H)
        d['L'] = steady_state_ify(L)
        d['M'] = steady_state_ify(M)
        d['N'] = steady_state_ify(N)
        return d