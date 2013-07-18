"""

Symbolic expressions can be evaluated by substituting the values of each symbol. It is however an expensive operation
 which becomes very costly when the number of evaluations grows.

The functions in this module  take a list of symbolic expressions representing a function :math:`R^p \\rightarrow R^n`
and turn it into an efficient python function, which can be evaluated repeatedly at lower cost. They use one of the next
libraries for efficient vectorization: `numpy <http://numpy.scipy.org/>`_, `numexpr <http://code.google.com/p/numexpr/>`_ or `theano <http://deeplearning.net/software/theano/>`_:

"""


from __future__ import division

from dolo.symbolic.derivatives import DerivativesTree
from dolo.compiler.common import DicPrinter
from dolo.symbolic.symbolic import TSymbol

DerivativesTree.symbol_type = TSymbol

def compile_function(equations, args, parms, max_order, return_text=False):

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

    if return_text:
        return txt
    else:
        return code_to_function(txt,'dynamic_function')


from numpy import zeros
from numpy import exp, log
from numpy import sin, cos, tan
from numpy import arcsin as asin
from numpy import arccos as acos
from numpy import arctan as atan
from numpy import sinh, cosh, tanh
from numpy import pi
from numpy import inf

from numba import float64

def compile_multiargument_function(equations, args_list, args_names, parms, fname='anonymous_function', diff=True, return_text=False, use_numexpr=False, order='columns'):

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

    if order == 'rows':
        raise(Exception('not implemented'))

    template = '{0}[{1}]'

    declarations = ""
    sub_list = {}

    for i,args in enumerate(args_list):
        vec_name = args_names[i]
        for j,v in enumerate(args):
            sub_list[v] = template.format(vec_name,j)
            # declarations += "    {0}_{1} = {0}[{1}]\n".format(vec_name, j)

    for i,p in enumerate(parms):
        sub_list[p] = '{0}[{1}]'.format('p',i)
        # declarations += "    {0}_{1} = {0}[{1}]\n".format('p', i)

    import sympy
    # TODO: construct a common list of symbol that should be understood everywhere
    sub_list[sympy.Symbol('inf')] = 'inf'

    text = '''
from numpy import zeros
from numpy import exp, log
from numpy import sin, cos, tan
from numpy import arcsin as asin
from numpy import arccos as acos
from numpy import arctan as atan
from numpy import sinh, cosh, tanh
from numpy import pi
from numpy import inf

def {fname}({args_names}, {param_names}, val):

    val = zeros({n_equations})
{content}

    return val

    '''

    from dolo.compiler.common import DicPrinter

    dp = DicPrinter(sub_list)

    def write_eqs(eq_l,outname='val'):
        eq_block = ''
        for i,eq in enumerate(eq_l):
            eq_string = dp.doprint_numpy(eq)
            eq_block += '    val[{0}] = {1}\n'.format(i, eq_string)
        return eq_block

    content = write_eqs(equations)



    return_names = 'val'
    text = text.format(
            fname = fname,
            n_equations = len(equations),
            declarations = declarations,
            var = args_names[0],
            content = content,
            return_names = return_names,
            args_names = str.join(', ', args_names),
            param_names = 'p'
            )

    if return_text:
        return text

    args_size = [len(e) for e in args_list] + [len(parms)]
    return_size = len(equations)


    return code_to_function(text,fname,args_size,return_size)


def code_to_function(text, name, args_size, return_size):
    from numpy import zeros
    from numpy import exp, log
    from numpy import sin, cos, tan
    from numpy import arcsin as asin
    from numpy import arccos as acos
    from numpy import arctan as atan
    from numpy import sinh, cosh, tanh
    from numpy import pi
    from numpy import inf
    d = locals()
    e = {}
    print(text)
    exec(text, d, e)
    fun = e[name]
    from numba.vectorize import GUVectorize
    from numba import float64
    signature = str.join(',',['(n{})'.format(i) for i in range(len(args_size))])
    signature += '->(n)'.format(return_size)
    args_types = [float64[:]]*(len(args_size)+1)
    print(signature)
    print(args_types)
    gufunc = GUVectorize(fun, signature)
    gufunc.add(argtypes=args_types)
    fun = gufunc.build_ufunc()
    return fun


def eqdiff(leq,lvars):
    resp = []
    for eq in leq:
        el = [ eq.diff(v) for v in lvars]
        resp += [el]
    return resp



if __name__ == '__main__':

    from dolo import *
    import numpy

    #
    # gm = yaml_import('examples/global_models/rbc.yaml', compiler='numba')
    # gmp = yaml_import('examples/global_models/rbc.yaml', compiler='numpy')

    import yaml
    # with file('/home/pablo/Programmation/washington/code/recipes.yaml') as f:
    with file('../washington/code/recipes.yaml') as f:
        recipes = yaml.load(f)

    # fname = '/home/pablo/Programmation/washington/code/rbc_fg.yaml'
    fname = '../washington/code/rbc_fg.yaml'

    first = 'numexpr'
    second = 'numba'

    gm = yaml_import(fname, compiler=first, order='columns', recipes=recipes)
    gmp = yaml_import(fname, compiler=second, order='columns', recipes=recipes)


    ss = gmp.calibration['states']
    xx = gmp.calibration['controls']
    p = gmp.calibration['parameters']

    ee = numpy.array([0],dtype=numpy.double)

    N = 100000

    ss = numpy.ascontiguousarray( numpy.tile(numpy.atleast_2d(ss), (N,1) ) )
    xx = numpy.ascontiguousarray( numpy.tile(numpy.atleast_2d(xx), (N,1) ) )
    ee = numpy.ascontiguousarray( numpy.tile(numpy.atleast_2d(ee), (N,1) ) )



    g = gm.functions['transition']
    f = gm.functions['arbitrage']

    gp = gmp.functions['transition']
    fp = gmp.functions['arbitrage']

    import time

    print('numpy')
    tmp = gp(ss,xx,ee,p)
    t1 = time.time()
    for i in range(50):
        tmp = gp(ss,xx,ee,p)
    t2 = time.time()

    tmp = fp(ss,xx,ss,xx,p)
    t3 = time.time()
    for i in range(50):
        tmp = fp(ss,xx,ss,xx,p)
    t4 = time.time()

    print('first {}'.format(t2-t1))
    print('second {}'.format(t4-t3))

    print('numba')

    tmp2 = g(ss,xx,ee,p)
    t1 = time.time()
    for i in range(50):
        tmp2 = g(ss,xx,ee,p)
    t2 = time.time()

    tmp2 = f(ss,xx,ss,xx,p)
    t3 = time.time()
    for i in range(50):
        tmp2 = f(ss,xx,ss,xx,p)
    t4 = time.time()

    print('first {}'.format(t2-t1))
    print('second {}'.format(t4-t3))


    print("Error : {}".format( abs(tmp2 - tmp).max() ))
