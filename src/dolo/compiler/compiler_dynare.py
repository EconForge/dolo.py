from dolo.compiler.compiler import *
from dolo.symbolic.derivatives import *

import math
import sympy

import time

class CustomPrinter(sympy.printing.StrPrinter):
    def _print_TSymbol(self, expr):
        return expr.__str__()

class DynareCompiler(Compiler):
    
    def compute_main_file(self,omit_nnz=True):
        model = self.model

        # should be computed somewhere else
        all_symbols = set([])
        for eq in model.equations:
            all_symbols.update( eq.atoms() )
        format_dict = dict()
        format_dict['fname'] = model.fname
        format_dict['maximum_lag'] = max([-v.lag for v in all_symbols if isinstance(v,TSymbol)])
        format_dict['maximum_lead'] = max([v.lag for v in all_symbols if isinstance(v,TSymbol)])
        format_dict['maximum_endo_lag'] = max([-v.lag for v in all_symbols if isinstance(v,Variable)])
        format_dict['maximum_endo_lead'] = max([v.lag for v in all_symbols if isinstance(v,Variable)])
        format_dict['maximum_exo_lag'] = max([-v.lag for v in all_symbols if isinstance(v,Shock)])
        format_dict['maximum_exo_lead'] = max([v.lag for v in all_symbols if isinstance(v,Shock)])
        format_dict['endo_nbr'] = len(model.variables)
        format_dict['exo_nbr'] = len(model.shocks)
        format_dict['param_nbr'] = len(model.parameters)



        output = """%
% Status : main Dynare file
%
% Warning : this file is generated automatically by Dolo for Dynare
%           from model file (.mod)

clear all
tic;
global M_ oo_ options_
global ys0_ ex0_ ct_
options_ = [];
M_.fname = '{fname}';
%
% Some global variables initialization
%
global_initialization;
diary off;
warning_old_state = warning;
warning off;
delete {fname}.log;
warning warning_old_state;
logname_ = '{fname}.log';
diary {fname}.log;
options_.model_mode = 0;
erase_compiled_function('{fname}_static');
erase_compiled_function('{fname}_dynamic');

M_.exo_det_nbr = 0;
M_.exo_nbr = {exo_nbr};
M_.endo_nbr = {endo_nbr};
M_.param_nbr = {param_nbr};
M_.Sigma_e = zeros({exo_nbr}, {exo_nbr});
M_.exo_names_orig_ord = [1:{exo_nbr}];
""".format( **format_dict )

        string_of_names = lambda l: str.join(' , ',["'%s'"%str(v) for v in l])
        string_of_tex_names  = lambda l: str.join(' , ',["'%s'"%v._latex_() for v in l])
        output += "M_.exo_names = strvcat( {0} );\n" .format(string_of_names(model.shocks))
        output += "M_.exo_names_tex = strvcat( {0} );\n".format(string_of_tex_names(model.shocks))
        output += "M_.endo_names = strvcat( {0} );\n".format(string_of_names(model.variables))
        output += "M_.endo_names_tex = strvcat( {0} );\n".format(string_of_tex_names(model.variables))
        output += "M_.param_names = strvcat( {0} );\n".format( string_of_names(model.parameters) )
        output += "M_.param_names_tex = strvcat( {0} );\n".format( string_of_tex_names(model.parameters) )

        from dolo.misc.matlab import value_to_mat
        lli = value_to_mat(self.lead_lag_incidence_matrix())
        output += "M_.lead_lag_incidence = {0}';\n".format(lli)

        output += """M_.maximum_lag = {maximum_lag};
M_.maximum_lead = {maximum_lead};
M_.maximum_endo_lag = {maximum_endo_lag};
M_.maximum_endo_lead = {maximum_endo_lead};
M_.maximum_exo_lag = {maximum_exo_lag};
M_.maximum_exo_lead = {maximum_exo_lead};
oo_.steady_state = zeros({endo_nbr}, 1);
oo_.exo_steady_state = zeros({exo_nbr}, 1);
M_.params = repmat(NaN,{param_nbr}, 1);
""".format( **format_dict )

        # Now we add tags for equations
        tags_array_string = []
        for eq in model.equations:
            for k in eq.tags.keys():
                tags_array_string.append( "{n}, '{key}', '{value}'".format(n=eq.n, key=k, value=eq.tags[k]) )
        output += "M_.equations_tags = {{\n{0}\n}};\n".format(";\n".join(tags_array_string))


        if not omit_nnz:
            # we don't set the number of non zero derivatives yet
            order = max([ndt.depth() for ndt in self.dynamic_derivatives])
            output += "M_.NNZDerivatives = zeros({0}, 1); % parrot mode\n".format(order)
            output += "M_.NNZDerivatives(1) = {0}; % parrot mode\n".format(self.NNZDerivatives(1))
            output += "M_.NNZDerivatives(2) = {0}; % parrot mode\n".format(self.NNZDerivatives(2))
            output += "M_.NNZDerivatives(3) = {0}; % parrot mode\n".format(self.NNZDerivatives(3))

        idp = DicPrinter(self.static_substitution_list(y='oo_.steady_state',params='M_.params'))

        # BUG: how do we treat parameters absent from the model, but present in parameters_values ?

        from dolo.misc.calculus import solve_triangular_system

        [junk,porder] = solve_triangular_system(model.parameters_values)
        porder = [p for p in porder if p in model.parameters]

        d = dict()
        d.update(model.parameters_values)
        d.update(model.init_values)

        [junk,vorder] = solve_triangular_system(model.init_values)
        vorder = [v for v in vorder if p in model.variables]

        for p in porder:
            i = model.parameters.index(p) + 1
            output += "M_.params({0}) = {1};\n".format(i,idp.doprint_matlab(model.parameters_values[p]))
            output += "{0} = M_.params({1});\n".format(p,i)

        output += '''%
% INITVAL instructions
%
'''
        #idp = DicPrinter(self.static_substitution_list(y='oo_.steady_state',params='M_.params'))
        output += "options_.initval_file = 0; % parrot mode\n"
        for v in vorder:
            if v in self.model.variables: # should be removed
                i = model.variables.index(v) + 1
                output += "oo_.steady_state({0}) = {1};\n".format(i,idp.doprint_matlab(model.init_values[v]))
        # we don't allow initialization of shocks to nonzero values
        output += "oo_.exo_steady_state = zeros({0},1);\n".format( len(model.shocks) )
        output += '''oo_.endo_simul=[oo_.steady_state*ones(1,M_.maximum_lag)];
if M_.exo_nbr > 0;
    oo_.exo_simul = [ones(M_.maximum_lag,1)*oo_.exo_steady_state'];
end;
if M_.exo_det_nbr > 0;
    oo_.exo_det_simul = [ones(M_.maximum_lag,1)*oo_.exo_det_steady_state'];
end;
'''
        #output += '''steady;
        output +="""
%
% SHOCKS instructions
%
make_ex_;
M_.exo_det_length = 0; % parrot
"""

        for i in range(model.covariances.shape[0]):
            for j in range(model.covariances.shape[1]):
                expr = model.covariances[i,j]
                if expr != 0:
                    v = str(expr).replace("**","^")
                    #if (v != 0) and not isinstance(v,sympy.core.numbers.Zero):
                    output += "M_.Sigma_e({0}, {1}) = {2};\n".format(i+1,j+1,v)

        # This results from the stoch_simul(
#        output += '''options_.drop = 200;
#options_.order = 3;
#var_list_=[];
#info = stoch_simul(var_list_);
#'''
#
#        output += '''save('example2_results.mat', 'oo_', 'M_', 'options_');
#diary off
#
#disp(['Total computing time : ' dynsec2hms(toc) ]);'''

        f = file(model.fname + '.m','w')
        f.write(output)
        f.close()

        return output

    def NNZDerivatives(self,order):
        n = 0
        for ndt in self.dynamic_derivatives:
            # @type ndt NonDecreasingTree
            l = ndt.list_nth_order_children(order)
            for c in l:
                # we must compute the number of permutations
                vars = c.vars
                s = factorial(len(vars))
                for v in set(vars):
                    s  = s / factorial(vars.count(v))
                n += s
        return n

    def NNZStaticDerivatives(self,order):
        n = 0
        for ndt in self.static_derivatives:
            # @type ndt NonDecreasingTree
            l = ndt.list_nth_order_children(order)
            for c in l:
                # we must compute the number of permutations
                vars = c.vars
                s = factorial(len(vars))
                for v in set(vars):
                    s  = s / factorial(vars.count(v))
                n += s
        return n



    def export_to_modfile(self,output_file=None,return_text=False,options={},append="",comments = "", solve_init_state=False):
        
        model = self.model
        #init_state = self.get_init_dict()

        default = {'steady':False,'check':False,'dest':'dynare','order':1,'use_dll':False}#default options
        for o in default:
            if not o in options:
                options[o] = default[o]

        init_block = ["// Generated by Dolo"]
        init_block.append( "// Model basename : " + self.model.fname)
        init_block.append(comments)
        init_block.append( "" )
        init_block.append( "var " + str.join(",", map(str,self.model.variables) ) + ";")
        init_block.append( "" )
        init_block.append( "varexo " + str.join(",", map(str,self.model.shocks) ) + ";" )
        init_block.append( "" )
        init_block.append( "parameters " + str.join(",",map(str,self.model.parameters)) + ";" )

        init_state = model.parameters_values.copy()
        init_state.update( model.init_values.copy() )

        from dolo.misc.calculus import solve_triangular_system
        if not solve_init_state:
            itd = model.init_values
            order = solve_triangular_system(init_state,return_order=True)
        else:
            itd,order = calculus.solve_triangular_system(init_state)



        for p in order:
            if p in model.parameters:
            #init_block.append(p.name + " = " + str(init_state[p]) + ";")
                if p in model.parameters_values:
                    init_block.append(p.name + " = " + str(model.parameters_values[p]).replace('**','^') + ";")

        printer = CustomPrinter()
        model_block = []
        if options['use_dll']:
            model_block.append( "model(use_dll);" )
        else:
            model_block.append( "model;" )
        for eq in self.model.equations:
            s = printer.doprint(eq)
            s = s.replace("==","=")
            s = s.replace("**","^")
#            s = s.replace("_b1","(-1)") # this should allow lags more than 1
#            s = s.replace("_f1","(+1)") # this should allow lags more than 1
            model_block += "\n"
            if 'name' in eq.tags:
                model_block.append("// eq %s : %s" %(eq.n,eq.tags.get('name')))
            else:
                model_block.append("// eq %s" %(eq.n))
            if len(eq.tags)>0:
                model_block.append("[ {0} ]".format(" , ".join(["{0} = '{1}'".format(k,eq.tags[k]) for k in eq.tags])))
            model_block.append(s + ";")
        model_block.append( "end;" )

        if options['dest'] == 'dynare':
            shocks_block = []
            if model.covariances != None:
                shocks_block.append("shocks;")
                cov_shape = model.covariances.shape
                for i in range(cov_shape[0]):
                    for j in range(i,cov_shape[1]):
                        cov = model.covariances[i,j]
                        if cov != 0:
                            tcov = str(cov).replace('**','^')
                            if i == j:
                                shocks_block.append("var " + str(self.model.shocks[i]) + " = " + tcov + " ;")
                            else:
                                shocks_block.append("var " + str(self.model.shocks[i]) + "," + str(self.model.shocks[j]) + " = " + tcov + " ;")
                shocks_block.append("end;")
        elif options['dest'] == 'dynare++':
            shocks_block = []
            shocks_block.append("vcov = [")
            cov_shape = self.covariances.shape
            shocks_matrix = []
            for i in range(cov_shape[0]):
                shocks_matrix.append(" ".join( map(str,self.covariances[i,:]) ))
            shocks_block.append( ";\n".join(shocks_matrix))
            shocks_block.append( "];" )


        initval_block = []
        if len(model.init_values)>0:
            initval_block.append("initval;")
            for v in order:
                if v in model.init_values:
                    initval_block.append(str(v) + " = " + str( itd[v] ).replace('**','^') + ";")
                elif options['dest'] == 'dynare++':
                    # dynare++ doesn't default to zero for non initialized variables
                    initval_block.append(str(v) + " = 0 ;")
            initval_block.append("end;")
        #else:
        #itd = self.get_init_dict()
        #initval_block = []
        #initval_block.append("initval;")
        #for v in self.model.variables:
        #if v in self.init_values:
        #initval_block.append(str(v) + " = " + str(itd[v]) +  ";")
        #initval_block.append("end;")
        model_text = ""
        model_text += "\n".join(init_block)
        model_text += "\n\n" + "\n".join(model_block)
        model_text += "\n\n" + "\n".join(shocks_block)
        model_text += "\n\n" + "\n".join(initval_block)

        if options['steady'] and options['dest']=='dynare':
            model_text += "\n\nsteady;\n"

        if options['check'] and options['dest']=='dynare':
            model_text += "\n\ncheck;\n"

        if options['dest'] == 'dynare++':
            model_text += "\n\norder = " + str(options['order']) + ";\n"

        model_text += "\n" + append

        if output_file == None:
            output_file = self.model.fname + '.mod'
        if output_file != False:
            f = file(output_file,'w' )
            f.write(model_text)
            f.close()

        if return_text:
            return(model_text)
        else:
            return None


    def compute_dynamic_mfile(self,max_order=2):

        print "Computing dynamic .m file at order {0}.".format(max_order)

        DerivativesTree.symbol_type = TSymbol

        model = self.model

        var_order = model.dyn_var_order + model.shocks

        # TODO create a log system
        t = []
        t.append(time.time())
        sols = []
        i = 0
        for eq in model.equations:
            i+=1
            ndt = DerivativesTree(eq.gap)
            ndt.compute_nth_order_children(max_order)
            sols.append(ndt)

        t.append(time.time())

        self.dynamic_derivatives = sols

        dyn_subs_dict = self.dynamic_substitution_list()
        dyn_printer = DicPrinter(dyn_subs_dict)

        txt = """function [residual, {gs}] = {fname}_dynamic(y, x, params, it_)
%
% Status : Computes dynamic model for Dynare
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

%
% Model equations
%

    residual = zeros({neq}, 1);
"""
        gs = str.join(', ',[('g'+str(i)) for i in range(1,(max_order+1))])
        txt = txt.format(gs=gs,fname=model.fname,neq=len(model.equations))

        t.append(time.time())
        for i in range(len(sols)):
            ndt = sols[i]
            eq = ndt.expr
            rhs = dyn_printer.doprint_matlab(eq)
            txt += '    residual({0}) = {1};\n'.format(i+1,rhs )
        t.append(time.time())
        
        for current_order in range(1,(max_order+1)):
            if current_order == 1:
                matrix_name = "Jacobian"
            elif current_order == 2:
                matrix_name = "Hessian"
            else:
                matrix_name = "{0}_th order".format(current_order)

            txt += """
    if nargout >= {orderr}
%
% {matrix_name} matrix
%

""".format(orderr=current_order+1,matrix_name=matrix_name)
            if current_order == 2:
                txt.format(matrix_name="Hessian")
            elif current_order == 1:
                txt.format(matrix_name="Jacobian")
            t.append(time.time())

            nnzd = self.NNZDerivatives(current_order)
            if True: # we write full matrices ...
                txt += "        v{order} = zeros({nnzd}, 3);\n".format(order=current_order, nnzd=nnzd)
                i = 0
                for n in range(len(sols)):
                    ndt = sols[n]
                    l = ndt.list_nth_order_children(current_order)
                    for nd in l:
                        i += 1
                         # here we compute indices where we write the derivatives
                        indices = nd.compute_index_set_matlab(var_order)
                        rhs = dyn_printer.doprint_matlab(nd.expr)
                        i0 = indices[0]
                        indices.remove(i0)
                        i_ref = i
                        txt += '        v{order}({i},:) = [{i_eq}, {i_col}, {value}] ;\n'.format(order=current_order,i=i,i_eq=n+1,i_col=i0,value=rhs)
                        for ind in indices:
                            i += 1
                            txt += '        v{order}({i},:) = [{i_eq}, {i_col}, v{order}({i_ref},3)];\n'.format(order=current_order,i=i,i_eq = n+1,i_col=ind,i_ref=i_ref)
                txt += '        g{order} = sparse(v{order}(:,1),v{order}(:,2),v{order}(:,3),{n_rows},{n_cols});\n'.format(order=current_order,n_rows=len(model.equations),n_cols=len(var_order)**current_order)
            else: # ... or sparse matrices
                print 'to be implemented'

            txt += """
    end
"""
        t.append(time.time())
        f = file(model.fname + '_dynamic.m','w')
        f.write(txt)
        f.close()
        return txt


    def compute_static_mfile(self,max_order=1):

        print "Computing static .m file at order {0}.".format(max_order)

        DerivativesTree.symbol_type = Variable

        model = self.model
        var_order = model.variables

        # TODO create a log system

        sols = []
        i = 0
        for eq in model.equations:
            i+=1
            l = [tv for tv in eq.atoms() if isinstance(tv,Variable)]
            expr = eq.gap
            for tv in l:
                if tv.lag != 0:
                    expr = expr.subs(tv,tv.P)
            ndt = DerivativesTree(expr)
            ndt.compute_nth_order_children(max_order)
            sols.append(ndt)
        self.static_derivatives = sols

        stat_subs_dict = self.static_substitution_list()
        
        stat_printer = DicPrinter(stat_subs_dict)

        txt = """function [residual, {gs}] = {fname}_static(y, x, params, it_)
%
% Status : Computes static model for Dynare
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

%
% Model equations
%

    residual = zeros({neq}, 1);
"""
        gs = str.join(', ',[('g'+str(i)) for i in range(1,(max_order+1))])
        txt = txt.format(gs=gs,fname=model.fname,neq=len(model.equations))

        for i in range(len(sols)):
            ndt = sols[i]
            eq = ndt.expr
            rhs = stat_printer.doprint_matlab(eq)
            txt += '    residual({0}) = {1};\n'.format(i+1,rhs )

        for current_order in range(1,(max_order+1)):
            if current_order == 1:
                matrix_name = "Jacobian"
            elif current_order == 2:
                matrix_name = "Hessian"
            else:
                matrix_name = "{0}_th order".format(current_order)

            txt += """
    if nargout >= {orderr}
        g{order} = zeros({n_rows}, {n_cols});

%
% {matrix_name} matrix
%
\n""".format(order=current_order,orderr=current_order+1,n_rows=len(model.equations),n_cols=len(var_order)**current_order,matrix_name=matrix_name)
            if current_order == 2:
                txt.format(matrix_name="Hessian")

            # What is the equivalent of NNZ for static files ?
            nnzd = self.NNZStaticDerivatives(current_order)

            if True:
                for n in range(len(sols)):
                    ndt = sols[n]
                    l = ndt.list_nth_order_children(current_order)
                    for nd in l:
                         # here we compute indices where we write the derivatives
                        indices = nd.compute_index_set_matlab(var_order)

                        rhs = stat_printer.doprint_matlab(nd.expr)

                        #rhs = comp.dyn_tabify_expression(nd.expr)
                        i0 = indices[0]
                        indices.remove(i0)
                        txt += '        g{order}({0},{1}) = {2};\n'.format(n+1,i0,rhs,order=current_order)
                        for ind in indices:
                            txt += '        g{order}({0},{1}) = g{order}({0},{2});\n'.format(n+1,ind,i0,order=current_order)
            else:
                print 'to be implemented'
            txt += """
    end
"""
        
        f = file(model.fname + '_static.m','w')
        f.write(txt)
        f.close()
        return txt

    def export_infos(self):
        from dolo.misc.matlab import struct_to_mat, value_to_mat
        model = self.model
        txt = 'global infos_;\n'
        txt += struct_to_mat(model.info,'infos_')
        # we compute the list of portfolio equations
        def select(x):
            if x.tags.get('portfolio','false') == 'true':
                return '1'
            else:
                return '0'
        txt += 'infos_.portfolio_equations = [%s];\n' % str.join(' ',[select(x) for x in model.equations])

        incidence_matrix = model.incidence_matrix_static()
        txt += 'infos_.incidence_matrix_static = %s;\n' % value_to_mat(incidence_matrix)
        f = file(model.fname + '_infos.m','w')
        f.write(txt)
        f.close()


def factorial(n):
    return math.factorial(n)