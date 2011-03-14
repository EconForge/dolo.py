from dolo.symbolic.model import *
from dolo.symbolic.symbolic import *
from dolo.compiler.compiler import Compiler
import numpy as np
import numpy.linalg

class MirFacCompiler(Compiler):
    
    def __init__(self,model):
        self.model = model
        # we assume model has already been checked

    def read_model(self):

        dmodel = self.model
        model = dmodel

        def_eqs = [eq for eq in dmodel.equations if 'def' == eq.tags['eq_type']]


        from dolo.misc.misc import map_function_to_expression
        from dolo.symbolic.symbolic import Variable
        def timeshift(v,n):
            if isinstance(v,Variable):
                return v(n)
            else:
                return v

        import sympy

        #### built substitution dict
        def_dict = {}
        for eq in def_eqs:
            v = eq.lhs
            rhs = sympy.sympify( eq.rhs )
            def_dict[v] = rhs
            def_dict[v(1)] = map_function_to_expression( lambda x: timeshift(x,1), rhs)

        new_equations = []
        tbr = []
        for i,eq in enumerate(dmodel.equations) :
            if not ('def' == eq.tags['eq_type']):
                lhs = sympy.sympify( eq.lhs ).subs(def_dict)
                rhs = sympy.sympify( eq.rhs ).subs(def_dict)
                neq = Equation(lhs,rhs).tag(**eq.tags)
                new_equations.append( neq )

        dmodel['equations'] = new_equations
        dmodel.check_consistency()



        f_eqs = [eq for eq in dmodel.equations if eq.tags['eq_type']=='f']
        g_eqs = [eq for eq in dmodel.equations if eq.tags['eq_type']=='g']
        h_eqs = [eq for eq in dmodel.equations if 'h' == eq.tags['eq_type']]            


        
        states_vars = [eq.lhs for eq in g_eqs]
        exp_vars =  [eq.lhs for eq in h_eqs]
        controls = set(dmodel.variables) - set(states_vars + exp_vars)
        controls = list(controls)

        states_vars = [v for v in dmodel.variables if v in states_vars]
        exp_vars = [v for v in dmodel.variables if v in exp_vars]
        controls = [v for v in dmodel.variables if v in controls]


        # now we remove the left side of equations
        f_eqs = [eq.gap for eq in f_eqs]
        g_eqs = [eq.rhs for eq in g_eqs]
        h_eqs = [eq.rhs for eq in h_eqs]




        g_eqs = [map_function_to_expression(lambda x: timeshift(x,1),eq) for eq in g_eqs]
        h_eqs = [map_function_to_expression(lambda x: timeshift(x,-1),eq) for eq in h_eqs]


    #    sub_list[v] = v.name

        # read complementarity conditions
        compcond = {}
        of_eqs = [eq for eq in dmodel.equations if eq.tags['eq_type']=='f']
        locals = {}
        import sympy
        locals['inf'] = sympy.Symbol('inf')
        for v in model.variables + model.parameters:
            locals[v.name] = v
        import re
        compregex = re.compile('(.*)<=(.*)<=(.*)')
        for eq in of_eqs:
            tg = eq.tags['complementarity']
            [lhs,mhs,rhs] = compregex.match(tg).groups()
            [lhs,mhs,rhs] = [eval(x,locals) for x in [lhs,mhs,rhs] ]
            compcond[mhs] = (lhs,rhs)
        
        complementarities = [compcond[v] for v in controls]

        inf_bounds = [c[0] for c in complementarities]
        sup_bounds = [c[1] for c in complementarities]
        
        data = {
            'f_eqs': f_eqs,
            'g_eqs': g_eqs,
            'h_eqs': h_eqs,
            'controls': controls,
            'states_vars': states_vars,
            'exp_vars': exp_vars,
            'inf_bounds': inf_bounds,
            'sup_bounds': sup_bounds
        }
        return data


    def perturbation_solution(self):
        """
        Returns perturbation solution around the steady-state
        Result is [X,Y,Z] representing :
        s_t = X s_{t-1} + Y e_t
        x_t = Z s_t
        """

        model = self.model
        data = self.read_model()
        
        controls_i = [model.variables.index(v) for v in data['controls'] ]
        states_i = [model.variables.index(v) for v in data['states_vars'] ]

        from dolo.numeric.perturbations import solve_decision_rule
        dr = solve_decision_rule(model)


        A = dr['g_a'][states_i,:]
        B = dr['g_a'][controls_i,:]

        [Z,err,rank,junk] = np.linalg.lstsq(A.T,B.T)
        Z = Z.T

        X = A[:,states_i] + np.dot( A[:,controls_i], Z )
        Y = dr.ghu[states_i,:]

        # We also compute bounds for distribution at a given probability interval

        M = np.linalg.solve( 1-X, Y)

        Sigma = np.array(model.covariances).astype(np.float64)
        [V,P] = np.linalg.eigh(Sigma)
        # we have Sigma == P * diag(V) * P.T
        # unconditional distribution is ( P*diag(V) ) * N
        # where N is the normal distribution associated
        H = np.dot(Y,np.dot(P,np.diag(V)))
        n_s = Sigma.shape[0]
        I = np.eye(n_s)
        points = np.concatenate([H,-H],axis=1)
        lam = 2 # this coefficient should be more cleverly defined
        p_infs = np.min(points,axis = 1) * lam
        p_max  = np.max(points,axis = 1) * lam

        # steady_state values
        s_ss = dr['ys'][states_i,]
        x_ss = dr['ys'][controls_i,]

        bounds = np.row_stack([
            s_ss + p_infs,
            s_ss + p_max
        ])
        return [[s_ss,x_ss],[X,Y,Z],bounds]



    def process_output_python(self):
        data = self.read_model()
        dmodel = self.model
        model = dmodel

        f_eqs = data['f_eqs']
        g_eqs = data['g_eqs']
        h_eqs = data['h_eqs']
        states_vars = data['states_vars']
        controls = data['controls']
        exp_vars = data['exp_vars']
        inf_bounds = data['inf_bounds']
        sup_bounds = data['sup_bounds']

        sub_list = dict()
        for i,v in enumerate(exp_vars):
            sub_list[v] = 'ep[{0},:]'.format(i)

        for i,v in enumerate(controls):
            sub_list[v] = 'x[{0},:]'.format(i)

        for i,v in enumerate(states_vars):
            sub_list[v] = 's[{0},:]'.format(i)

        for i,v in enumerate(dmodel.shocks):
            sub_list[v] = 'e[{0},:]'.format(i)

        for i,v in enumerate(dmodel.parameters):
            sub_list[v] = 'p[{0}]'.format(i)



        text = '''
from __future__ import division
import numpy as np
inf = np.inf

def model(flag,s,x,ep,e,{param_names}):


    n = s.shape[-1]

    if flag == 'b':
{eq_bounds_block}
        return [out1,out2]

    elif flag == 'f':
{eq_fun_block}
        return [out1,out2,out3]

    elif flag == 'g':
{state_trans_block}
        return [out1,out2]

    elif flag == 'h':
{exp_fun_block}
        return [out1,out2,out3]

        '''

        from dolo.compiler.compiler import DicPrinter

        dp = DicPrinter(sub_list)

        def write_eqs(eq_l,outname='out1'):
            eq_block = '        {0} = np.zeros( ({1},n) )\n'.format(outname, len(eq_l))
            for i,eq in enumerate(eq_l):
                eq_block += '        {0}[{1},:] = {2}\n'.format(outname, i,  dp.doprint_numpy(eq,vectorize=True))
            return eq_block

        def write_der_eqs(eq_l,v_l,lhs):
            eq_block = '        {lhs} = np.zeros( ({0},{1},n) )\n'.format(len(eq_l),len(v_l),lhs=lhs)
            eq_l_d = eqdiff(eq_l,v_l)
            for i,eqq in enumerate(eq_l_d):
                for j,eq in enumerate(eqq):
                    s = dp.doprint_numpy( eq, vectorize=True )
                    eq_block += '        {lhs}[{0},{1},:] = {2}\n'.format(i,j,s,lhs=lhs)
            return eq_block

        eq_bounds_block = write_eqs(inf_bounds)
        eq_bounds_block += write_eqs(sup_bounds,'out2')

        eq_f_block = write_eqs(f_eqs)
        eq_f_block += write_der_eqs(f_eqs,controls,'out2')
        eq_f_block += write_der_eqs(f_eqs,exp_vars,'out3')

        eq_g_block = write_eqs(g_eqs)
        eq_g_block += write_der_eqs(g_eqs,controls,'out2')

        eq_h_block = write_eqs(h_eqs)
        eq_h_block += write_der_eqs(h_eqs,controls,'out2')
        eq_h_block += write_der_eqs(h_eqs,states_vars,'out3')

        text = text.format(
                eq_bounds_block = eq_bounds_block,
                mfname =  model.fname,
                eq_fun_block=eq_f_block,
                state_trans_block=eq_g_block,
                exp_fun_block=eq_h_block,
                #    param_names=str.join(',',[p.name for p in dmodel.parameters])
                param_names = 'p'
                )

        return text

        #f = file('pf/optimal_growth_model.py','w')

        #f.write( text )
        #f.close()

    def process_output_matlab(self,with_param_names=False):
        data = self.read_model()
        dmodel = self.model
        model = dmodel

        f_eqs = data['f_eqs']
        g_eqs = data['g_eqs']
        h_eqs = data['h_eqs']
        states_vars = data['states_vars']
        controls = data['controls']
        exp_vars = data['exp_vars']
        inf_bounds = data['inf_bounds']
        sup_bounds = data['sup_bounds']

        sub_list = dict()
        for i,v in enumerate(exp_vars):
            sub_list[v] = 'ep(:,{0})'.format(i+1)

        for i,v in enumerate(controls):
            sub_list[v] = 'x(:,{0})'.format(i+1)

        for i,v in enumerate(states_vars):
            sub_list[v] = 's(:,{0})'.format(i+1)

        for i,v in enumerate(dmodel.shocks):
            sub_list[v] = 'e(:,{0})'.format(i+1)

        for i,v in enumerate(dmodel.parameters):
            sub_list[v] = 'p({0})'.format(i+1)




        text = '''
function [out1,out2,out3,out4] = {mfname}(flag,s,x,ep,e,{param_names});

% p is the vector of parameters
{param_def}

n = size(s,1);
switch flag

case 'b';
{eq_bounds_block}

case 'f';
{eq_fun_block}

case 'g';
{state_trans_block}

case 'h';
{exp_fun_block}

end;
'''

        from dolo.compiler.compiler import DicPrinter

        dp = DicPrinter(sub_list)

        def write_eqs(eq_l,outname='out1'):
            eq_block = '    {0} = zeros(n,{1});\n'.format(outname,len(eq_l))
            for i,eq in enumerate(eq_l):
                eq_block += '    {0}(:,{1}) = {2};\n'.format( outname,  i+1,  dp.doprint_matlab(eq,vectorize=True) )
            return eq_block

        def write_der_eqs(eq_l,v_l,lhs):
            eq_block = '    {lhs} = zeros(n,{0},{1});\n'.format(len(eq_l),len(v_l),lhs=lhs)
            eq_l_d = eqdiff(eq_l,v_l)
            for i,eqq in enumerate(eq_l_d):
                for j,eq in enumerate(eqq):
                    s = dp.doprint_matlab( eq, vectorize=True )
                    eq_block += '    {lhs}(:,{0},{1}) = {2}; %d eq {eq_n} w.r.t. {vname}\n'.format(i+1,j+1,s,lhs=lhs,eq_n=i+1,vname=str(v_l[j]) )
            return eq_block

        eq_bounds_block = write_eqs(inf_bounds)
        eq_bounds_block += write_eqs(sup_bounds,'out2')

        eq_f_block = write_eqs(f_eqs)
        eq_f_block += write_der_eqs(f_eqs,controls,'out2')
        eq_f_block += write_der_eqs(f_eqs,exp_vars,'out3')
        eq_f_block += write_der_eqs(f_eqs,states_vars,'out4')

        eq_g_block = write_eqs(g_eqs)
        eq_g_block += write_der_eqs(g_eqs,controls,'out2')
        eq_g_block += write_der_eqs(g_eqs,states_vars,'out3')


        eq_h_block = write_eqs(h_eqs)
        eq_h_block += write_der_eqs(h_eqs,controls,'out2')
        eq_h_block += write_der_eqs(h_eqs,states_vars,'out3')

        if with_param_names:
            eq_h_block = 's=snext;\nx=xnext;\n'+eq_h_block

        param_def = 'p = [ ' + str.join(',',[p.name for p in dmodel.parameters])  + '];'


        text = text.format(
            eq_bounds_block = eq_bounds_block,
            mfname = 'mf_' + model.fname,
            eq_fun_block=eq_f_block,
            state_trans_block=eq_g_block,
            exp_fun_block=eq_h_block,
            param_names= 'snext,xnext,' + (str.join(',',[p.name for p in dmodel.parameters]) if with_param_names  else 'p'),
            param_def= param_def if with_param_names else ''
        )

        return text

def eqdiff(leq,lvars):
    resp = []
    for eq in leq:
        el = [ eq.diff(v) for v in lvars]
        resp += [el]
    return resp


if __name__ == "__main__":
    from dolo import dynare_import
    model = dynare_import('../../../examples/global_models/optimal_growth.mod')
    model.check()
    mfcp = MirFacCompiler(model)
    print mfcp.process_output_matlab()
    #print mfcp.process_output_python()