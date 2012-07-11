# pickle
# hash
from dolo.misc.misc import map_function_to_expression
from dolo.symbolic.symbolic import Variable
import numpy as np
def timeshift(v,n):
    if isinstance(v,Variable):
        return v(n)
    else:
        return v

def eqdiff(leq,lvars):
    resp = []
    for eq in leq:
        el = [ eq.diff(v) for v in lvars]
        resp += [el]
    return resp


class CompilerMatlab:

    def __init__(self,model):
        self.model = model
        #self.sgm = simple_global_representation(model, substitute_auxiliary=True, solve_systems=True)

    def process_output(self, solution_order=False, fname=None):

        from dolo.numeric.perturbations_to_states import simple_global_representation
        data = simple_global_representation(self.model, substitute_auxiliary=True, keep_auxiliary=True, solve_systems=True)

#        print data['a_eqs']
#        print data['f_eqs']
        dmodel = self.model
        model = dmodel

        f_eqs = data['f_eqs']
        g_eqs = data['g_eqs']

        g_eqs = [map_function_to_expression(lambda x: timeshift(x,1),eq) for eq in g_eqs]

#        h_eqs = data['h_eqs']
        auxiliaries = data['auxiliaries']
        states = data['states']
        controls = data['controls']

        shocks = dmodel.shocks
#        exp_vars = data['exp_vars']
        #inf_bounds = data['inf_bounds']
        #sup_bounds = data['sup_bounds']


        controls_f = [v(1) for v in controls]
        states_f = [v(1) for v in states]
        shocks_f = [v(1) for v in shocks]

        sub_list = dict()
#        for i,v in enumerate(exp_vars):
#            sub_list[v] = 'z(:,{0})'.format(i+1)

        for i,v in enumerate(controls):
            sub_list[v] = 'x(:,{0})'.format(i+1)
            sub_list[v(1)] = 'xnext(:,{0})'.format(i+1)

        for i,v in enumerate(states):
            sub_list[v] = 's(:,{0})'.format(i+1)
            sub_list[v(1)] = 'snext(:,{0})'.format(i+1)

        for i,v in enumerate(dmodel.shocks):
            sub_list[v] = 'e(:,{0})'.format(i+1)
            sub_list[v(1)] = 'enext(:,{0})'.format(i+1)

        for i,v in enumerate(dmodel.parameters):
            sub_list[v] = 'p({0})'.format(i+1)




        text = '''function [model] = get_model()
    model = model_info;
    model.f = @f;
    model.g = @g;
    model.a = @a;
end

function [out1,out2,out3,out4,out5] = f(s,x,snext,xnext,enext,p)
    n = size(s,1);
{eq_fun_block}
end

function [out1,out2,out3] = g(s,x,e,p)
    n = size(s,1);
{state_trans_block}
end

function [out1,out2,out3] = a(s,x,p)
    n = size(s,1);
{aux_block}
end

function [out1] = model_info() % informations about the model
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

#        eq_bounds_block = write_eqs(inf_bounds,ntabs=2)
#        eq_bounds_block += '\n'
#        eq_bounds_block += write_eqs(sup_bounds,'out2',ntabs=2)

        eq_f_block = '''
    % f
{0}

if nargout >= 2

    % df/ds
{1}

    % df/dx
{2}

    % df/dsnext
{3}

    % df/dxnext
{4}

end

        '''.format( write_eqs(f_eqs,'out1',3),
                    write_der_eqs(f_eqs,states,'out2',3),
                    write_der_eqs(f_eqs,controls,'out3',3),
                    write_der_eqs(f_eqs,states_f,'out4',3),
                    write_der_eqs(f_eqs,controls_f,'out5',3),
                    write_der_eqs(f_eqs, shocks_f, 'out6',3)
#                    write_der_eqs(f_eqs,exp_vars,'out4',3)
            )

        eq_g_block = '''
    % g

{0}

if nargout >=2
    % dg/ds
    {1}
    % dg/dx
    {2}
end
        '''.format( write_eqs(g_eqs,'out1',3),
                    write_der_eqs(g_eqs,states,'out2',3),
                    write_der_eqs(g_eqs,controls,'out3',3)
            )

        if 'a_eqs' in data:
            a_eqs = data['a_eqs']
            eq_a_block =  '''
    % a

{0}

if nargout >=2
    % da/ds
    {1}
    % da/dx
    {2}
end
                    '''.format( write_eqs(a_eqs,'out1',3),
                                write_der_eqs(a_eqs,states,'out2',3),
                                write_der_eqs(a_eqs,controls,'out3',3)
                        )
        else:
            eq_a_block = ''

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
    mod.states = {states};
    mod.controls = {controls};
    mod.auxiliaries = {auxiliaries};
    mod.parameters = {parameters};
    mod.shocks = {shocks};
    mod.s_ss = {s_ss};
    mod.x_ss = {x_ss};
    mod.params = {params_values};
'''.format(
    states = '{{ {} }}'.format(str.join(',', ["'{}'".format(v) for v in states])),
    controls = '{{ {} }}'.format(str.join(',', ["'{}'".format(v) for v in controls])),
    auxiliaries = '{{ {} }}'.format(str.join(',', ["'{}'".format(v) for v in auxiliaries])),
    parameters = '{{ {} }}'.format(str.join(',', ["'{}'".format(v) for v in model.parameters])),
    shocks = '{{ {} }}'.format(str.join(',', ["'{}'".format(v) for v in model.shocks])),
    s_ss = value_to_mat(s_ss),
    x_ss = value_to_mat(x_ss),
    params_values = value_to_mat(params_values)
)
        if solution_order:
            from dolo.numeric.perturbations_to_states import approximate_controls

            ZZ = approximate_controls(self.model,order=solution_order, return_dr=False)
            n_c = len(controls)

            ZZ = [np.array(e) for e in ZZ]
            ZZ = [e[:n_c,...] for e in ZZ] # keep only control vars. (x) not expectations (h)

            solution = "    mod.X = cell({0},1);\n".format(len(ZZ))
            for i,zz in enumerate(ZZ):
                solution += "    mod.X{{{0}}} = {1};\n".format(i+1,value_to_mat(zz.real))
            model_info += solution
        model_info += '    out1 = mod;\n'

        text = text.format(
#            eq_bounds_block = eq_bounds_block,
            mfname = fname if fname else 'mf_' + model.fname,
            eq_fun_block=eq_f_block,
            state_trans_block=eq_g_block,
            aux_block=eq_a_block,
#            exp_fun_block=eq_h_block,
#            solution = solution,
            model_info = model_info
        )

        return text



if __name__ == '__main__':

    from dolo import *

    model = yaml_import('../../../examples/baseline.yaml')
    comp = CompilerMatlab(model)

    print comp.process_output()
