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

from numbapro import float64
from numbapro.vectorize import guvectorize

def compile_multiargument_function(equations, args_list, args_names, parms, fname='anonymous_function', diff=True, return_text=False, order='columns'):

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

    args_size = [len(e) for e in args_list] + [len(parms)]
    return_size = len(equations)

    size_same_as_output = args_size.index(return_size)

    signature = str.join(',',['(n{})'.format(i) for i in range(len(args_size))])
    signature += '->(n{})'.format(return_size-1)
    argtypes = [float64[:]]*(len(args_size)+1)


    text = '''
#from numpy import zeros
#from numpy import exp, log
#from numpy import sin, cos, tan
#from numpy import arcsin as asin
#from numpy import arccos as acos
#from numpy import arctan as atan
#from numpy import sinh, cosh, tanh
#from numpy import pi
#from numpy import inf

from numbapro.vectorize import guvectorize
from numbapro import float64, void

#@guvectorize( [float64[:](*{argtypes})], "{signature}", target='cpu' )
@guvectorize( [void(*{argtypes})], "{signature}", target='gpu' )
def {fname}({args_names}, {param_names}, val):

{content}

#    return val



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
            param_names = 'p',
            argtypes = str(argtypes),
            signature = signature,
    )

    if return_text:
        return text
    print(args_list)

    return code_to_function(text,fname,args_size,return_size, size_same_as_output)


def code_to_function(text, name, args_size, return_size, size_same_as_output):
    from numpy import zeros
    from numpy import exp, log
    from numpy import sin, cos, tan
    from numpy import arcsin as asin
    from numpy import arccos as acos
    from numpy import arctan as atan
    from numpy import sinh, cosh, tanh
    from numpy import pi
    from numpy import inf
    print('8'*10)
    print('8'*10)
    print(text)
    print('8'*10)
    print('8'*10)
    d = locals()
    e = {}
    exec(text, d, e)
    fun = e[name]
    fun.max_blocksize = 64
#    from numbapro.vectorize import GUVectorize
#    from numbapro import float64, void
#    signature = str.join(',',['(n{})'.format(i) for i in range(len(args_size))])
#    signature += '->(n{})'.format(return_size-1)
#    args_types =[float64[:]]*(len(args_size)+1)
#    args_types = # void(*args_types)
#    gufunc = GUVectorize(fun, signature, target='gpu')
#    print(signature)
#    print(args_types)
#    gufunc.add(argtypes=args_types)
#    fun = gufunc.build_ufunc()
    print('Success !')
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


    import yaml
    with file('/home/pablo/Programmation/washington/code/recipe_fga.yaml') as f:
        recipes = yaml.load(f)

    fname = '/home/pablo/Programmation/washington/code/rbc_fg.yaml'

    first = 'numexpr'
    second = 'numba'
    gm = yaml_import(fname, compiler=first, order='columns', recipes=recipes)
    gmp = yaml_import(fname, compiler=second, recipes=recipes, order='columns')

    # gmp = yaml_import('examples/global_models/rbc.yaml', compiler='numexpr')

    # print(model.__class__)
    # gm = GModel(model, compiler='numexpr')
    # # gm = GModel(model, compiler='theano')
#    gm = GModel(model)

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

    print(first)

    tmp = g(ss,xx,ee,p)
    t1 = time.time()
    for i in range(50):
        tmp = g(ss,xx,ee,p)
    t2 = time.time()

    tmp = f(ss,xx,ss,xx,p)
    t3 = time.time()
    for i in range(50):
        tmp = f(ss,xx,ss,xx,p)
    t4 = time.time()

    print('first {}'.format(t2-t1))
    print('second {}'.format(t4-t3))

    from numbapro.cuda import to_device

    g_ss = to_device(ss)
    g_xx = to_device(xx)
    g_ee = to_device(ee)
    g_p = to_device(p)
    g_o = to_device(ss.copy()) 
    print(second)
    gp(g_ss,g_xx,g_ee,g_p)
#    exit()
    t1 = time.time()
    for i in range(50):
        g_o = gp(g_ss,g_xx,g_ee,g_p)
    t2 = time.time()
#    exit()
    tmp = fp(ss,xx,ss,xx,p)
    t3 = time.time()
    for i in range(50):
        tmp = fp(ss,xx,ss,xx,p)
    t4 = time.time()

    print('first {}'.format(t2-t1))
    print('second {}'.format(t4-t3))


