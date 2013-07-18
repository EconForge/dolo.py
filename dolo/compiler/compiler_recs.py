from __future__ import division
from dolo.symbolic.model import SModel
from dolo.symbolic.symbolic import Equation

class RecsCompiler(object):

    def __init__(self,model):
        self.model = model
        self.__transformed_model__ = None
        # we assume model has already been checked

    def read_model(self):

        import re
        import sympy
        from dolo.symbolic.symbolic import map_function_to_expression
        from dolo.symbolic.symbolic import Variable

        if self.__transformed_model__:
            return self.__transformed_model__

        dmodel = SModel(**self.model) # copy the model
        dmodel.check_consistency(auto_remove_variables=False)

        def_eqs = [eq for eq in dmodel.equations if eq.tags['eq_type'] in ('def', 'auxiliary')]

        def timeshift(v,n):
            if isinstance(v,Variable):
                return v(n)
            else:
                return v

        # Build substitution dict
        def_dict = {}
        for eq in def_eqs:
            v = eq.lhs
            rhs = sympy.sympify(eq.rhs)
            def_dict[v] = rhs
            def_dict[v(1)] = map_function_to_expression(lambda x: timeshift(x,1), rhs)

        new_equations = []
        tbr = []
        for i,eq in enumerate(dmodel.equations) :
            if not ('def' == eq.tags['eq_type']):
                lhs = sympy.sympify( eq.lhs ).subs(def_dict)
                rhs = sympy.sympify( eq.rhs ).subs(def_dict)
                neq = Equation(lhs,rhs).tag(**eq.tags)
                new_equations.append(neq)

        dmodel['equations'] = new_equations
        dmodel.check_consistency()

        f_eqs = [eq for eq in dmodel.equations if eq.tags['eq_type'] in ('f','arbitrage','equilibrium')]
        g_eqs = [eq for eq in dmodel.equations if eq.tags['eq_type'] in ('g','transition')]
        h_eqs = [eq for eq in dmodel.equations if eq.tags['eq_type'] in ('h','expectation')]
        hm_eqs = [eq for eq in dmodel.equations if eq.tags['eq_type'] in ('h','expectation_mult')] # Need to understand the need for 'h'
        e_eqs = [eq for eq in dmodel.equations if eq.tags['eq_type'] in ('e','equation_error')]

        states_vars = [eq.lhs for eq in g_eqs]
        exp_vars =  [eq.lhs for eq in h_eqs]
        controls = set(dmodel.variables) - set(states_vars + exp_vars)
        controls = list(controls)

        states_vars = [v for v in dmodel.variables if v in states_vars]
        exp_vars = [v for v in dmodel.variables if v in exp_vars]
        controls = [v for v in dmodel.variables if v in controls]

        # Remove the left-hand side of equations
        f_eqs = [eq.gap for eq in f_eqs]
        g_eqs = [eq.rhs for eq in g_eqs]
        h_eqs = [eq.rhs for eq in h_eqs]
        hm_eqs = [eq.rhs for eq in hm_eqs]
        e_eqs = [eq.lhs for eq in e_eqs]

        g_eqs = [map_function_to_expression(lambda x: timeshift(x,1),eq) for eq in g_eqs]
        #h_eqs = [map_function_to_expression(lambda x: timeshift(x,-1),eq) for eq in h_eqs] #no

    #    sub_list[v] = v.name

        # Read complementarity conditions
        compcond = {}
        of_eqs = [eq for eq in dmodel.equations if eq.tags['eq_type'] in ('f','arbitrage','equilibrium')]
        locals = {}
        locals['inf'] = sympy.Symbol('inf')
        locals['log'] = sympy.log
        locals['exp'] = sympy.exp

        for v in dmodel.variables + dmodel.parameters:
            locals[v.name] = v
        compregex = re.compile('(.*)<=(.*)<=(.*)')
        for eq in of_eqs:
            tg = eq.tags['complementarity']
            [lhs,mhs,rhs] = compregex.match(tg).groups()
            [lhs,mhs,rhs] = [dmodel.eval_string(x) for x in [lhs,mhs,rhs]]
            compcond[mhs] = (lhs,rhs)

        complementarities = [compcond[v] for v in controls]

        inf_bounds = [c[0] for c in complementarities]
        sup_bounds = [c[1] for c in complementarities]

        data = {'f_eqs': f_eqs,
                'g_eqs': g_eqs,
                'h_eqs': h_eqs,
                'hm_eqs': hm_eqs,
                'e_eqs':e_eqs,
                'controls': controls,
                'states_vars': states_vars,
                'exp_vars': exp_vars,
                'inf_bounds': inf_bounds,
                'sup_bounds': sup_bounds}

        self.__transformed_model__ = data # cache computation

        return data

    def process_output_recs(self):
        '''Main function that formats the model in recs format'''
        import sympy
        from dolo.compiler.common import DicPrinter
        from dolo.misc.matlab import value_to_mat

        data = self.read_model()
        dmodel = self.model
        model = dmodel

        f_eqs = data['f_eqs']
        g_eqs = data['g_eqs']
        h_eqs = data['h_eqs']
        hm_eqs = data['hm_eqs']
        e_eqs = data['e_eqs']
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

        sub_list[sympy.Symbol('inf')] = 'inf'

        # Case h(s,x,e,sn,xn)
        text = '''function [out1,out2,out3,out4,out5] = {filename}(flag,s,x,z,e,snext,xnext,p,output)

voidcell                   = cell(1,5);
[out1,out2,out3,out4,out5] = voidcell{{:}};

switch flag

  case 'b'
    n = size(s,1);
{eq_bounds_block}

  case 'f'
    n = size(s,1);
{eq_fun_block}

  case 'g'
    n = size(s,1);
{state_trans_block}

  case 'h'
    n = size(snext,1);
{exp_fun_block}

  case 'e'
    n = size(s,1);
{equation_error_block}

  case 'params'
    out1 = {model_params};

  case 'ss'
{model_ss}

  case 'J'
{jac_struc}
end'''

        # Case h(.,.,.,sn,xn)*hmult(e)
        textmult = '''function [out1,out2,out3,out4,out5,out6] = {filename}(flag,s,x,z,e,snext,xnext,p,output)

voidcell                        = cell(1,6);
[out1,out2,out3,out4,out5,out6] = voidcell{{:}};

switch flag

  case 'b'
    n = size(s,1);
{eq_bounds_block}

  case 'f'
    n = size(s,1);
{eq_fun_block}

  case 'g'
    n = size(s,1);
{state_trans_block}

  case 'h'
    n = size(snext,1);
{exp_fun_block}
{exp_exp_mult_block}

  case 'e'
    n = size(s,1);
{equation_error_block}

  case 'params'
    out1 = {model_params};

  case 'ss'
{model_ss}

  case 'J'
{jac_struc}
end'''

        dp = DicPrinter(sub_list)

        def write_eqs(eq_l, outname='out1', ntabs=0, default=None):
            '''Format equations and bounds'''
            if default:
                eq_block = '  ' * ntabs + '{0} = ' + default + '(n,{1});'
                eq_block = eq_block.format(outname,len(eq_l))
            else:
                eq_block = '  ' * ntabs + '{0} = zeros(n,{1});'.format(outname,len(eq_l))
            eq_template = '\n' + '  ' * ntabs + '{0}(:,{1}) = {2};'
            for i,eq in enumerate(eq_l):
                eq_txt = dp.doprint_matlab(eq,vectorize=True)
                if eq_txt != default:
                    eq_block += eq_template.format(outname, i+1, eq_txt)
            return eq_block

        def write_der_eqs(eq_l, v_l, lhs, ntabs=0):
            '''Format Jacobians'''
            eq_block = '  ' * ntabs + '{lhs} = zeros(n,{0},{1});'
            eq_block = eq_block.format(len(eq_l), len(v_l), lhs=lhs)
            eq_l_d = eqdiff(eq_l,v_l)
            eq_template = '\n' + '  ' * ntabs + '{lhs}(:,{0},{1}) = {2}; % d eq_{eq_n} w.r.t. {vname}'
            jac_struc = [[0 for i in range(len(v_l))] for j in range(len(eq_l))]
            for i,eqq in enumerate(eq_l_d):
                for j,eq in enumerate(eqq):
                    s = dp.doprint_matlab(eq, vectorize=True)
                    if s != '0':
                        eq_block += eq_template.format(i+1, j+1, s, lhs=lhs, eq_n=i+1, vname=str(v_l[j]))
                        jac_struc[i][j] = 1
            return [eq_block,jac_struc]

        eq_bounds_block = '''
    % b
{0}

    % db/ds
    if nargout==4
{1}
{2}
    end'''
        eq_bounds_values = write_eqs(inf_bounds, ntabs=2, default='-inf')
        eq_bounds_values += '\n'
        eq_bounds_values += write_eqs(sup_bounds, 'out2', ntabs=2, default='inf')

        eq_bounds_jac_inf = write_der_eqs(inf_bounds, states_vars, 'out3', 3)
        eq_bounds_jac_sup = write_der_eqs(sup_bounds, states_vars, 'out4', 3)
        eq_bounds_block = eq_bounds_block.format(eq_bounds_values,
                                                 eq_bounds_jac_inf[0],
                                                 eq_bounds_jac_sup[0])

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
    end'''
        # eq_f_block = eq_f_block.format(write_eqs(f_eqs, 'out1', 3),
        #                                write_der_eqs(f_eqs, states_vars, 'out2', 3),
        #                                write_der_eqs(f_eqs, controls, 'out3', 3),
        #                                write_der_eqs(f_eqs, exp_vars, 'out4', 3))
        df_ds = write_der_eqs(f_eqs, states_vars, 'out2', 3)
        df_dx = write_der_eqs(f_eqs, controls, 'out3', 3)
        df_dz = write_der_eqs(f_eqs, exp_vars, 'out4', 3)
        eq_f_block = eq_f_block.format(write_eqs(f_eqs, 'out1', 3),
                                       df_ds[0],
                                       df_dx[0],
                                       df_dz[0])
        jac_struc = '    out1.fs = '+list_to_mat(df_ds[1])+';\n'
        jac_struc += '    out1.fx = '+list_to_mat(df_dx[1])+';\n'
        jac_struc += '    out1.fz = '+list_to_mat(df_dz[1])+';\n'

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
    end'''
        # eq_g_block = eq_g_block.format(write_eqs(g_eqs, 'out1', 3),
        #                                write_der_eqs(g_eqs, states_vars, 'out2', 3),
        #                                write_der_eqs(g_eqs, controls, 'out3', 3))
        dg_ds = write_der_eqs(g_eqs, states_vars, 'out2', 3)
        dg_dx = write_der_eqs(g_eqs, controls, 'out3', 3)
        eq_g_block = eq_g_block.format(write_eqs(g_eqs, 'out1', 3),
                                       dg_ds[0],
                                       dg_dx[0])
        jac_struc += '    out1.gs = '+list_to_mat(dg_ds[1])+';\n'
        jac_struc += '    out1.gx = '+list_to_mat(dg_dx[1])+';\n'

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
    end'''
        # eq_h_block = eq_h_block.format(write_eqs(h_eqs, 'out1', 3),
        #                                write_der_eqs(h_eqs, states_vars, 'out2', 3),
        #                                write_der_eqs(h_eqs, controls, 'out3', 3),
        #                                write_der_eqs(h_eqs, states_f, 'out4', 3),
        #                                write_der_eqs(h_eqs, controls_f, 'out5', 3))
        dh_ds = write_der_eqs(h_eqs, states_vars, 'out2', 3)
        dh_dx = write_der_eqs(h_eqs, controls, 'out3', 3)
        dh_ds_f = write_der_eqs(h_eqs, states_f, 'out4', 3)
        dh_dx_f = write_der_eqs(h_eqs, controls_f, 'out5', 3)
        eq_h_block = eq_h_block.format(write_eqs(h_eqs, 'out1', 3),
                                       dh_ds[0],
                                       dh_dx[0],
                                       dh_ds_f[0],
                                       dh_dx_f[0])
        jac_struc += '    out1.hs = '+list_to_mat(dh_ds[1])+';\n'
        jac_struc += '    out1.hx = '+list_to_mat(dh_dx[1])+';\n'
        jac_struc += '    out1.hsnext = '+list_to_mat(dh_ds_f[1])+';\n'
        jac_struc += '    out1.hxnext = '+list_to_mat(dh_dx_f[1])+';\n'

        eq_hm_block = '''
    % hmult
    if output.hmult
{0}
    end'''
        eq_hm_block = eq_hm_block.format(write_eqs(hm_eqs, 'out6', 3))

        if e_eqs:
            equation_error_block = write_eqs(e_eqs, 'out1', 3)
        else:
            equation_error_block ='''    out1 = [];'''

        # Model informations
        [y,x,params_values] = model.read_calibration()
        vvs = model.variables
        s_ss = [y[vvs.index(v)] for v in model['variables_groups']['states']]
        x_ss = [y[vvs.index(v)] for v in model['variables_groups']['controls']]

        model_ss = '''    out1 = {s_ss};
    out2 = {x_ss};'''
        model_ss = model_ss.format(s_ss = value_to_mat(s_ss).replace(';',''),
                                   x_ss = value_to_mat(x_ss).replace(';',''))

        if hm_eqs:
            text = textmult.format(eq_bounds_block = eq_bounds_block,
                                   filename = model.fname,
                                   eq_fun_block = eq_f_block,
                                   state_trans_block = eq_g_block,
                                   exp_fun_block = eq_h_block,
                                   exp_exp_mult_block = eq_hm_block,
                                   equation_error_block = equation_error_block,
                                   model_params = value_to_mat(params_values),
                                   model_ss = model_ss,
                                   jac_struc = jac_struc)
        else:
            text = text.format(eq_bounds_block = eq_bounds_block,
                               filename = model.fname,
                               eq_fun_block = eq_f_block,
                               state_trans_block = eq_g_block,
                               exp_fun_block = eq_h_block,
                               equation_error_block = equation_error_block,
                               model_params = value_to_mat(params_values),
                               model_ss = model_ss,
                               jac_struc = jac_struc)

        return text


def eqdiff(leq,lvars):
    '''Calculate the Jacobian of the system of equations with respect to a set of variables.'''
    resp = []
    for eq in leq:
        el = [eq.diff(v) for v in lvars]
        resp += [el]
    return resp

def list_to_mat(l):
    mat = str(l)
    mat = mat.replace('[[','[')
    mat = mat.replace(']]',']')
    mat = mat.replace('],',';')
    mat = mat.replace(' [',' ')
    mat = mat.replace(',','')
    return mat

if __name__ == "__main__":

    from dolo import *

    model = yaml_import('examples/global_models/sto1.yaml')
    comp  = RecsCompiler(model)
    print comp.process_output_recs()

