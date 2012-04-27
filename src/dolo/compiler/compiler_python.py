from __future__ import division

from dolo.compiler.compiler import *
from dolo.misc.caching import memoized
from dolo.symbolic.derivatives import *

import math
import sympy

class CustomPrinter(sympy.printing.StrPrinter):
    def _print_TSymbol(self, expr):
        return expr.__str__()

class PythonCompiler(Compiler):

    def compute_static_pfile(self,max_order):

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

        stat_subs_dict = self.static_substitution_list(ind_0=0,brackets = True)

        stat_printer = DicPrinter(stat_subs_dict)

        txt = """def static_gaps(y, x, params):
#
# Status : Computes static model for Python
#
# Warning : this file is generated automatically by Dynare
#           from model file (.mod)
#
#
# Model equations
#
    import numpy as np
    from numpy import exp,log, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh

    it_ = 1 # should remove this !

    g = []

    residual = np.zeros({neq})
"""
        gs = str.join(', ',[('g'+str(i)) for i in range(1,(max_order+1))])
        txt = txt.format(gs=gs,fname=model.fname,neq=len(model.equations))

        for i in range(len(sols)):
            ndt = sols[i]
            eq = ndt.expr
            rhs = stat_printer.doprint_matlab(eq)
            txt += '    residual[{0}] = {1}\n'.format(i,rhs )

        txt += '    g.append(residual)\n'


        for current_order in range(1,(max_order+1)):
            if current_order == 1:
                matrix_name = "Jacobian"
            elif current_order == 2:
                matrix_name = "Hessian"
            else:
                matrix_name = "{0}_th order".format(current_order)

            txt += """
    g{order} = np.zeros(({n_rows}, {n_cols}))
#    {matrix_name} matrix\n""".format(order=current_order,orderr=current_order+1,n_rows=len(model.equations),n_cols=len(var_order)**current_order,matrix_name=matrix_name)
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
                    txt += '    g{order}[{0},{1}] = {2}\n'.format(n,i0-1,rhs,order=current_order)
                    for ind in indices:
                        txt += '    g{order}[{0},{1}] = g{order}[{0},{2}]\n'.format(n,ind-1,i0-1,order=current_order)
            txt += '    g.append(g{order})\n'.format(order=current_order)

        txt += "    return g\n"
        txt = txt.replace('^','**')
        exec txt
        return static_gaps

    @memoized
    def compute_dynamic_pfile_cached(self,max_order,compact_order,with_parameters):
        return self.compute_dynamic_pfile(max_order=max_order,compact_order=compact_order,with_parameters=with_parameters)

    def compute_dynamic_pfile(self,max_order=1,compact_order=True,with_parameters=False):

        if with_parameters:
            DerivativesTree.symbol_type = sympy.Symbol
        else:
            DerivativesTree.symbol_type = TSymbol


        model = self.model

        if compact_order:
            var_order = model.dyn_var_order + model.shocks
        else:
            var_order = [v(1) for v in model.variables]
            var_order += model.variables
            var_order += [v(-1) for v in model.variables]
            var_order += model.shocks
        if with_parameters:
            var_order += model.parameters

        # TODO create a log system

        sols = []
        i = 0
        for eq in model.equations:
            i+=1
            ndt = DerivativesTree(eq.gap)
            ndt.compute_nth_order_children(max_order)
            sols.append(ndt)

        self.dynamic_derivatives = sols

        dyn_subs_dict = self.dynamic_substitution_list(brackets = True,compact=compact_order)
        dyn_printer = DicPrinter(dyn_subs_dict)

        txt = """def dynamic_gaps(y, x, params):
#
# Status : Computes dynamic model for Dynare
#
# Warning : this file is generated automatically by Dynare
#           from model file (.mod)

#
# Model equations
#
    it_ = 0

    import numpy as np
    from numpy import exp, log
    from numpy import arctan as atan

    g = []

    residual = np.zeros({neq});
"""
        gs = str.join(', ',[('g'+str(i)) for i in range(1,(max_order+1))])
        txt = txt.format(gs=gs,fname=model.fname,neq=len(model.equations))

        for i in range(len(sols)):
            ndt = sols[i]
            eq = ndt.expr
            rhs = dyn_printer.doprint_matlab(eq)
            txt += '    residual[{0}] = {1};\n'.format(i,rhs )

        txt += '    g.append(residual)\n'

        for current_order in range(1,(max_order+1)):
            if current_order == 1:
                matrix_name = "Jacobian"
            elif current_order == 2:
                matrix_name = "Hessian"
            else:
                matrix_name = "{0}_th order".format(current_order)

            txt += """
#
# {matrix_name} matrix
#

""".format(orderr=current_order+1,matrix_name=matrix_name)
            if current_order == 2:
                txt.format(matrix_name="Hessian")
            elif current_order == 1:
                txt.format(matrix_name="Jacobian")

            #nnzd = self.NNZDerivatives(current_order)

            n_cols = (len(var_order),)*current_order
            n_cols = ','.join( [str(s) for s in n_cols] )
            txt += "    g{order} = np.zeros( ({n_eq}, {n_cols}) );\n".format(order=current_order,n_eq=len(model.equations), n_cols=n_cols )
            for n in range(len(sols)):
                ndt = sols[n]
                l = ndt.list_nth_order_children(current_order)
                for nd in l:
                     # here we compute indices where we write the derivatives
                    indices = nd.compute_index_set(var_order)
                    rhs = dyn_printer.doprint_numpy(nd.expr)
                    i0 = indices[0]
                    i_col_s = ','.join([str(nn) for nn in i0])
                    indices.remove(i0)

                    i_col_s_ref = i_col_s
                    txt += '    g{order}[{i_eq},{i_col}] = {value}\n'.format(order=current_order,i_eq=n,i_col=i_col_s,value=rhs)
                    for ind in indices:
                        i += 1
                        i_col_s = ','.join([str(nn) for nn in ind])
                        txt += '    g{order}[{i_eq},{i_col}] = g{order}[{i_eq},{i_col_ref}] \n'.format(order=current_order,i_eq = n,i_col=i_col_s,i_col_ref = i_col_s_ref)

            txt += "    g.append(g{order})\n".format(order=current_order)
        txt += "    return g\n"
        txt = txt.replace('^','**')
        
        exec txt
        return dynamic_gaps


def factorial(n):
    math.factorial(n)
