"""

Symbolic expressions can be evaluated by substituting the values of each symbol. It is however an expensive operation
 which becomes very costly when the number of evaluations grows.

The functions in this module  take a list of symbolic expressions representing a function :math:`R^p \\rightarrow R^n`
and turn it into an efficient python function, which can be evaluated repeatedly at lower cost. They use one of the next
libraries for efficient vectorization: `numpy <http://numpy.scipy.org/>`_, `numexpr <http://code.google.com/p/numexpr/>`_ or `theano <http://deeplearning.net/software/theano/>`_:

"""


from __future__ import division

from dolo.symbolic.derivatives import DerivativesTree
from dolo.compiler.compiler import DicPrinter
from dolo.symbolic.symbolic import TSymbol, Shock, Variable

DerivativesTree.symbol_type = TSymbol

def compile_function(equations, args, parms, max_order, return_function=True):

    """
    :param equations:
    :param args:
    :param parms:
    :param max_order:
    :param return_function:
    :return:
    """

    var_order = args

    sols = []
    for eq in equations:
        ndt = DerivativesTree(eq, ref_var_list=var_order)
        ndt.compute_nth_order_children(max_order)
        sols.append(ndt)

    dyn_subs_dict = dict()
    for i,v in enumerate(args):
        dyn_subs_dict[v] = 'x_' + str(i)

    for i,p in enumerate(parms):
        dyn_subs_dict[p] = 'p_' + str(i)

    preamble_l = ['    x_{i} = x[{i}]   # {v}'.format(i=i,v=v) for i,v in enumerate(args)]
    preamble_l += ['    p_{i} = p[{i}]    # {p}'.format(i=i,p=p) for i,p in enumerate(parms)]
    preamble = str.join('\n',preamble_l)

    dyn_printer = DicPrinter(dyn_subs_dict)

    txt = """def dynamic_function(x, p):
#
#
#
    import numpy as np
    from numpy import exp, log
    from numpy import sin, cos, tan
    from numpy import arcsin as asin
    from numpy import arccos as acos
    from numpy import arctan as atan
    from numpy import sinh, cosh, tanh
    from numpy import pi

{preamble}

    f = []

    residual = np.zeros({neq},dtype=np.float64);
"""
    gs = str.join(', ',[('f'+str(i)) for i in range(1,(max_order+1))])
    txt = txt.format(gs=gs,fname='noname',neq=len(equations), preamble=preamble)

    for i in range(len(sols)):
        ndt = sols[i]
        eq = ndt.expr
        rhs = dyn_printer.doprint_numpy(eq)
        txt += '    residual[{0}] = {1}\n'.format(i,rhs )

    txt += '    f.append(residual)\n'

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
        txt += "    f{order} = np.zeros( ({n_eq}, {n_cols}), dtype=np.float64 )\n".format(order=current_order,n_eq=len(equations), n_cols=n_cols )
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
                txt += '    f{order}[{i_eq},{i_col}] = {value}\n'.format(order=current_order,i_eq=n,i_col=i_col_s,value=rhs)
                for ind in indices:
                    i += 1
                    i_col_s = ','.join([str(nn) for nn in ind])
                    txt += '    f{order}[{i_eq},{i_col}] = f{order}[{i_eq},{i_col_ref}] \n'.format(order=current_order,i_eq = n,i_col=i_col_s,i_col_ref = i_col_s_ref)

        txt += "    f.append(f{order})\n".format(order=current_order)
    txt += "    return f\n"
    txt = txt.replace('^','**')
    if return_function:
        exec txt
        dynamic_function.__source__ = txt
        return dynamic_function
    else:
        return txt


def compile_multiargument_function(equations, args_list, args_names, parms, fname='anonymous_function', diff=True, vectorize=True, return_text=False):
    """
    :param equations: list of sympy expressions
    :param args_list: list of lists of symbols (e.g. [[a_1,a_2], [b_1,b_2]])
    :param args_names: list of strings (['a','b']
    :param parms: list of symbols to be used as parameters
    :param fname: name of the python function to be generated
    :param diff: include symbolic derivates in generated function
    :param vectorize: arguments are vectorized (not parameters)
    :param return_function: a python function f(a,b,p) where p is a vector of parameters and a, b, arrays
    :return:
    """

    if vectorize:
        template = '{0}[{1},...]'
    else:
        template = '{0}[{1}]'

    sub_list = {}

    for i,args in enumerate(args_list):
        vec_name = args_names[i]
        for j,v in enumerate(args):
            sub_list[v] = template.format(vec_name,j)

    for i,p in enumerate(parms):
        sub_list[p] = '{0}[{1}]'.format('p',i)




    text = '''
def {fname}({args_names}, {param_names}, derivs=False):

    import numpy as np
    from numpy import exp, log
    from numpy import sin, cos, tan
    from numpy import arcsin as asin
    from numpy import arccos as acos
    from numpy import arctan as atan
    from numpy import sinh, cosh, tanh
    from numpy import pi

    n = {var}.shape[-1]

{content}

    return {return_names}
    '''

    from dolo.compiler.compiler import DicPrinter

    dp = DicPrinter(sub_list)

    def write_eqs(eq_l,outname='val'):
        eq_block = '    {0} = np.zeros( ({1},n) )\n'.format(outname, len(eq_l))
        for i,eq in enumerate(eq_l):
            eq_block += '    {0}[{1},:] = {2}\n'.format(outname, i,  dp.doprint_numpy(eq))
        return eq_block

    def write_der_eqs(eq_l,v_l,lhs):
        eq_block = '    {lhs} = np.zeros( ({0},{1},n) )\n'.format(len(eq_l),len(v_l),lhs=lhs)
        eq_l_d = eqdiff(eq_l,v_l)
        for i,eqq in enumerate(eq_l_d):
            for j,eq in enumerate(eqq):
                s = dp.doprint_numpy( eq )
                eq_block += '    {lhs}[{0},{1},:] = {2}\n'.format(i,j,s,lhs=lhs)
        return eq_block

    content = write_eqs(equations)
    content += '''
    if not derivs:
        return val
    '''


    if diff:
        for i,a_g in enumerate(args_list):
            lhs = 'val_' + args_names[i]
            content += "\n    # Derivatives w.r.t: {0}\n\n".format(args_names[i])
            content += write_der_eqs(equations, a_g, lhs)

    return_names = '[val, ' + str.join(', ', [ 'val_'+ str(a) for a in args_names] ) + ']' if diff else 'val'
    text = text.format(
            fname = fname,
            var = args_names[0],
            content = content,
            return_names = return_names,
            args_names = str.join(', ', args_names),
            param_names = 'p'
            )


    if return_text:
        return text

    import numpy as np
    inf = np.inf
    exec text in locals(), globals()
    l = globals()
    return l[fname]


def eqdiff(leq,lvars):
    resp = []
    for eq in leq:
        el = [ eq.diff(v) for v in lvars]
        resp += [el]
    return resp