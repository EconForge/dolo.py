from __future__ import division

import numpy

def compile_source(source):
    exec(source)
    return test()

class CModel:

    model_type = 'fga'

    def as_type(self, tt):

        if tt == 'fga':
            return self
        elif tt == 'fg':
            cc = CModel_fg(self)
            return cc

    def __init__(self,smodel):

        self.model = smodel

        eq_groups = smodel['equations_groups']
        v_states = smodel['variables_groups']['states']
        v_controls = smodel['variables_groups']['controls']
        v_auxiliaries = smodel['variables_groups']['auxiliary']
        v_shocks = smodel.shocks
        v_parameters = smodel.parameters


        eq_g = [eq.rhs for eq in smodel['equations_groups']['transition']]
        eq_f = [eq.gap for eq in smodel['equations_groups']['arbitrage']]
        eq_a = [eq.rhs for eq in smodel['equations_groups']['auxiliary']]


        if 'auxiliary_2' in eq_groups:
            from dolo.misc.misc import timeshift
            from dolo.misc.triangular_solver import solve_triangular_system as simple_triangular_solve
            aux2_eqs = eq_groups['auxiliary_2']
            dd = {eq.lhs: eq.rhs for eq in aux2_eqs}
            dd.update( { eq.lhs(1): timeshift(eq.rhs,1) for eq in aux2_eqs } )
            dd.update( { eq.lhs(-1): timeshift(eq.rhs,-1) for eq in aux2_eqs } )
            ds = simple_triangular_solve(dd)

            eq_f =  [eq.subs(ds) for eq in eq_f]
            eq_a =  [eq.subs(ds) for eq in eq_a]
            eq_g =  [eq.subs(ds) for eq in eq_g]

        v_states_p = [v(-1) for v in v_states]
        v_controls_p = [v(-1) for v in v_controls]
        v_auxiliaries_p = [v(-1) for v in v_auxiliaries]

        v_states_f = [v(1) for v in v_states]
        v_controls_f = [v(1) for v in v_controls]
        v_auxiliaries_f = [v(1) for v in v_auxiliaries]
        v_shocks_f = [v(1) for v in v_shocks]

        from dolo.compiler.compiling_very_fast import compile_theano, compile_theano_2


        source = compile_theano_2( eq_f, [v_states, v_controls, v_states_f, v_controls_f, v_auxiliaries, v_auxiliaries_f, v_shocks_f], ['s','x','S','X','a','A','E'], v_parameters)
        f = compile_source(source)

        # here I should solve the triangular system
        source = compile_theano( v_states, eq_g, [v_states_p, v_controls_p, v_auxiliaries_p, v_shocks], ['s','x','a','e'], v_parameters)
        g = compile_source(source)

        source = compile_theano( v_auxiliaries, eq_a, [v_states, v_controls], ['s','x'], v_parameters)
        a = compile_source(source)

        self.__a_theano__ = a
        self.__g_theano__ = g
        self.__f_theano__ = f


    def f(self,s,x,S,X,a,A,E,p, derivs=False):
        res = self.__f_theano__(s,x,S,X,a,A,E,p)
        res = numpy.row_stack(res)
        return res

    def g(self,s,x,a,e,p, derivs=False):
        res = self.__g_theano__(s,x,a,e,p)
        res = numpy.row_stack(res)
        return res

    def a(self,s,x,p, derivs=False):
        res = self.__a_theano__(s,x,p)
        res = numpy.row_stack(res)
        return res


class CModel_fg:

    model_type = 'fg'

    def __init__(self, cmodel_fga):
        self.model = cmodel_fga.model
        self.model_fga = cmodel_fga

    def g(self, s, x, e, p, derivs=False):
        if not derivs:
            a = self.model_fga.a(s,x,p)
            S = self.model_fga.g(s,x,a,e,p)
            return S
        else:
            g0 = self.g(s,x,e,p)
            g_s = numdiff( lambda l: self.g(l,x,e,p), s, g0)
            g_x = numdiff( lambda l: self.g(s,l,e,p), x, g0)
            g_e = numdiff( lambda l: self.g(s,x,l,p), e, g0)
            return [g0, g_s, g_x, g_e]



    def f(self, s, x, S, X, E, p, derivs=False):
        if not derivs:
            a = self.model_fga.a(s,x,p)
            A = self.model_fga.a(S,X,p)
            rr = self.model_fga.f(s, x, S, X, a, A, E, p)
            return rr
        else:
            f0 = self.f(s,x,S,X,E,p)
            f_s = numdiff(lambda l: self.f(l,x,S,X,E,p), s, f0)
            f_x = numdiff(lambda l: self.f(s,l,S,X,E,p), x, f0)
            f_S = numdiff(lambda l: self.f(s,x,l,X,E,p), S, f0)
            f_X = numdiff(lambda l: self.f(s,x,S,l,E,p), X, f0)
            f_E = numdiff(lambda l: self.f(s,x,S,X,l,p), X, f0)

        return([f0, f_s, f_x, f_S, f_X, f_E])


def numdiff(f,x0,f0=None):

    eps = 1E-6

    if f0 == None:
        f0 = f(x0)

    p = f0.shape[0]
    q = x0.shape[0]
    N = x0.shape[1]

    df = numpy.zeros( (p,q,N) )
    for i in range(q):
        x = x0.copy()
        x[i,:] += eps
        ff = f(x)
        df[:,i,:] = (ff - f0)/eps

    # could be made more efficiently by making only one call to f
    # problem: the following doesn't work if f expects a specific dimension for x
    #    x = numpy.column_stack( [x0]*q )
    #    for i in range(q):
    #        x[i, i*N:(i+1)*N] += eps
    #
    #    ff = f(x)
    #    ff = ff.reshape(p,q,N)
    #    for i in range(q):
    #        df[:, i, :] = (ff[:, i, :] - f0)/eps


    return df




if __name__ == '__main__':

    from dolo import yaml_import, global_solve
#    model = yaml_import( '../../../examples/global_models/rbc.yaml')
    model = yaml_import('/home/pablo/Documents/Research/Thesis/chapter_5/code/models/integration_A.yaml')

#    cmodel = CModel(model)
    maxit = 5
    so = 4


    dr_smol_1 = global_solve(model, pert_order=1, n_s=1, smolyak_order=so, maxit=maxit, polish=False, numdiff=True)
    dr_smol_1b = global_solve(model, pert_order=1, n_s=1, smolyak_order=so, maxit=maxit, polish=False, numdiff=False)
    dr_smol_2 = global_solve(model, pert_order=1, n_s=1, smolyak_order=so, maxit=maxit, polish=False, numdiff=True, compiler='theano')
    dr_smol_2b = global_solve(model, pert_order=1, n_s=1, smolyak_order=so, maxit=maxit, polish=False, numdiff=False, compiler='theano')
#
#    from dolo.compiler.compiler_global import GlobalCompiler2
#    gc = GlobalCompiler2(model)
#    gc_theano = CModel(model)
#
#    exit()
#    from dolo.numeric.simulations import simulate
#    dr = approximate_controls(model)
#    s0 = dr.S_bar
#    sigma = model.read_covariances()
#
#    sim_1 = simulate(gc, dr, s0, sigma, n_exp=10000)
#    sim_2 = simulate(gc_theano, dr, s0, sigma, n_exp=10000)
