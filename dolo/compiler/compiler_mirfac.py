from __future__ import division

from dolo.symbolic.model import *
from dolo.symbolic.symbolic import *
from dolo.compiler.compiler import Compiler
import numpy as np
import numpy.linalg

class MirFacCompiler(Compiler):
    
    def __init__(self,model):
        self.model = model
        self.__transformed_model__ = None
        # we assume model has already been checked

    def read_model(self):
    
        if self.__transformed_model__:
            return self.__transformed_model__


        dmodel = Model(**self.model) # copy the model
        dmodel.check_consistency(auto_remove_variables=False)

        def_eqs = [eq for eq in dmodel.equations if eq.tags['eq_type'] in ('def', 'auxiliary')]

        from dolo.misc.misc import map_function_to_expression
        from dolo.symbolic.symbolic import Variable
        def timeshift(v,n):
            if isinstance(v,Variable):
                return v(n)
            else:
                return v

        import sympy

        #### build substitution dict
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



        f_eqs = [eq for eq in dmodel.equations if eq.tags['eq_type'] in ('f','arbitrage')]
        g_eqs = [eq for eq in dmodel.equations if eq.tags['eq_type'] in ('g','transition')]
        h_eqs = [eq for eq in dmodel.equations if eq.tags['eq_type'] in ('h','expectation')]


        
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
        #h_eqs = [map_function_to_expression(lambda x: timeshift(x,-1),eq) for eq in h_eqs] #no


    #    sub_list[v] = v.name

        # read complementarity conditions
        compcond = {}
        of_eqs = [eq for eq in dmodel.equations if eq.tags['eq_type'] in ('f','arbitrage')]
        locals = {}
        import sympy
        locals['inf'] = sympy.Symbol('inf')
        locals['log'] = sympy.log # this should be more generic
        locals['exp'] = sympy.exp
        
        for v in dmodel.variables + dmodel.parameters:
            locals[v.name] = v
        import re
        compregex = re.compile('(.*)<=(.*)<=(.*)')
        for eq in of_eqs:
            tg = eq.tags['complementarity']
            [lhs,mhs,rhs] = compregex.match(tg).groups()
            [lhs,mhs,rhs] = [dmodel.eval_string(x) for x in [lhs,mhs,rhs] ]
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
        
        self.__transformed_model__ = data # cache computation

        return data


    def perturbation_solution(self,with_bounds=False):
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
        dr = solve_decision_rule(model,order=1)


        A = dr['g_a'][states_i,:]
        B = dr['g_a'][controls_i,:]

        [Z,err,rank,junk] = np.linalg.lstsq(A.T,B.T)
        Z = Z.T

        X = A[:,states_i] + np.dot( A[:,controls_i], Z )
        Y = dr.ghu[states_i,:]

        # steady_state values
        s_ss = dr['ys'][states_i,]
        x_ss = dr['ys'][controls_i,]

        # We also compute bounds for distribution at a given probability interval
        if with_bounds:
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

            bounds = np.row_stack([
                s_ss + p_infs,
                s_ss + p_max
            ])
        else:
            bounds = None
            
        return [[s_ss,x_ss],[X,Y,Z],bounds]



    def process_output_recs(self, solution_order=False, fname=None):

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


        controls_f = [v(1) for v in controls]
        states_f = [v(1) for v in states_vars]

        sub_list = dict()
        for i,v in enumerate(exp_vars):
            sub_list[v] = 'z(:,{0})'.format(i+1)

        for i,v in enumerate(controls):
            sub_list[v] = 'x(:,{0})'.format(i+1)
            sub_list[v(1)] = 'xnext(:,{0})'.format(i+1)

        for i,v in enumerate(states_vars):
            sub_list[v] = 's(:,{0})'.format(i+1)
            sub_list[v(1)] = 'snext(:,{0})'.format(i+1)

        for i,v in enumerate(dmodel.shocks):
            sub_list[v] = 'e(:,{0})'.format(i+1)

        for i,v in enumerate(dmodel.parameters):
            sub_list[v] = 'p({0})'.format(i+1)


#        sub_list[sympy.inf] = 'Inf'


        text = '''function [out1,out2,out3,out4,out5] = {mfname}(flag,s,x,z,e,snext,xnext,p,out);

output = struct('F',1,'Js',0,'Jx',0,'Jsn',0,'Jxn',0,'Jz',0,'hmult',0);

if nargin == 9
  output                     = catstruct(output,out);
  voidcell                   = cell(1,5);
  [out1,out2,out3,out4,out5] = voidcell{{:}};
else
  if nargout >= 2, output.Js = 1; end
  if nargout >= 3, output.Jx = 1; end
  if nargout >= 4
    if strcmpi(flag, 'f')
      output.Jz = 1;
    else
      output.Jsn = 1;
    end
  end
  if nargout >= 5, output.Jxn = 1; end
end


switch flag

  case 'b';
    n = size(s,1);
{eq_bounds_block}

  case 'f';
    n = size(s,1);
{eq_fun_block}
  case 'g';
    n = size(s,1);
{state_trans_block}
  case 'h';
    n = size(snext,1);
{exp_fun_block}
  case 'e';
    out1 = [];

  case 'model'; % informations about the model
{model_info}

end
'''

        from dolo.compiler.compiler import DicPrinter

        dp = DicPrinter(sub_list)

        def write_eqs(eq_l,outname='out1',ntabs=0):
            eq_block = '  ' * ntabs + '{0} = zeros(n,{1});'.format(outname,len(eq_l))
            for i,eq in enumerate(eq_l):
                eq_block += '\n' + '  ' * ntabs + '{0}(:,{1}) = {2};'.format( outname,  i+1,  dp.doprint_matlab(eq,vectorize=True) )
            return eq_block

        def write_der_eqs(eq_l,v_l,lhs,ntabs=0):
            eq_block = '  ' * ntabs + '{lhs} = zeros(n,{0},{1});'.format(len(eq_l),len(v_l),lhs=lhs)
            eq_l_d = eqdiff(eq_l,v_l)
            for i,eqq in enumerate(eq_l_d):
                for j,eq in enumerate(eqq):
                    s = dp.doprint_matlab( eq, vectorize=True )
                    eq_block += '\n' + '  ' * ntabs + '{lhs}(:,{0},{1}) = {2}; % d eq_{eq_n} w.r.t. {vname}'.format(i+1,j+1,s,lhs=lhs,eq_n=i+1,vname=str(v_l[j]) )
            return eq_block

        eq_bounds_block = write_eqs(inf_bounds,ntabs=2)
        eq_bounds_block += '\n'
        eq_bounds_block += write_eqs(sup_bounds,'out2',ntabs=2)

        eq_f_block = '''
    % f
    if output.F
{0}
    end

    % df/ds
    if output.Js
{1}
    end

    % df/dx
    if output.Jx
{2}
    end

    % df/dz
    if output.Jz
{3}
    end
        '''.format( write_eqs(f_eqs,'out1',3),
                    write_der_eqs(f_eqs,states_vars,'out2',3),
                    write_der_eqs(f_eqs,controls,'out3',3),
                    write_der_eqs(f_eqs,exp_vars,'out4',3)
            )

        eq_g_block = '''
    % g
    if output.F
{0}      
    end

    if output.Js
{1}
    end

    if output.Jx
{2}
    end
        '''.format( write_eqs(g_eqs,'out1',3),
                    write_der_eqs(g_eqs,states_vars,'out2',3),
                    write_der_eqs(g_eqs,controls,'out3',3)
            )
        
        eq_h_block = '''
    %h
    if output.F
{0}
    end

    if output.Js
{1}
    end

    if output.Jx
{2}
    end

    if output.Jsn
{3}
    end

    if output.Jxn
{4}
    end
        '''.format(
             write_eqs(h_eqs,'out1',3),
             write_der_eqs(h_eqs,states_vars,'out2',3),
             write_der_eqs(h_eqs,controls,'out3',3),
             write_der_eqs(h_eqs,states_f,'out4',3),
             write_der_eqs(h_eqs,controls_f,'out5',3)
        )

        # if not with_param_names:
        #    eq_h_block = 's=snext;\nx=xnext;\n'+eq_h_block

        # param_def = 'p = [ ' + str.join(',',[p.name for p in dmodel.parameters])  + '];'

        from dolo.misc.matlab import value_to_mat
        
        # read model informations
        [y,x,params_values] = model.read_calibration()
        #params_values = '[' + str.join(  ',', [ str( p ) for p in params] ) + '];'
        vvs = model.variables
        s_ss = [ y[vvs.index(v)] for v in model['variables_groups']['states'] ]
        x_ss = [ y[vvs.index(v)] for v in model['variables_groups']['controls'] ]

        model_info = '''
    mod = struct;
    mod.s_ss = {s_ss};
    mod.x_ss = {x_ss};
    mod.params = {params_values};
'''.format(
    s_ss = value_to_mat(s_ss),
    x_ss = value_to_mat(x_ss),
    params_values = value_to_mat(params_values)
)
        if solution_order:
            from dolo.numeric.perturbations_to_states import approximate_controls

            ZZ = approximate_controls(self.model,order=solution_order)
            n_c = len(controls)
	    
	    ZZ = [np.array(e) for e in ZZ]
            ZZ = [e[:n_c,...] for e in ZZ] # keep only control vars. (x) not expectations (h)

            solution = "    mod.X = cell({0},1);\n".format(len(ZZ))
            for i,zz in enumerate(ZZ):
                solution += "    mod.X{{{0}}} = {1};\n".format(i+1,value_to_mat(zz))
            model_info += solution
        model_info += '    out1 = mod;\n'

        text = text.format(
            eq_bounds_block = eq_bounds_block,
            mfname = fname if fname else 'mf_' + model.fname,
            eq_fun_block=eq_f_block,
            state_trans_block=eq_g_block,
            exp_fun_block=eq_h_block,
#            solution = solution,
            model_info = model_info
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
    print mfcp.process_output_recs()
    #print mfcp.process_output_python()
