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

def compile_multiargument_function(equations, args_list, args_names, parms, fname='anonymous_function', diff=True,
                                   return_text=False, order='columns'):

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

#    val = zeros({n_equations})
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
   
    import random
    tfname = "__temp{}".format(random.random())
    tfname = tfname.replace('.','')
    with file(tfname+'.py','w') as f:
        f.write(text)
    e = __import__(tfname)
    fun = e.__dict__[name]
#    execfile('temp.py', d, e)
    
#    print(text)
#    exec(text, d, e)
#    fun = e[name]
    print(fun)
#    fun = transition
#    from numbapro.vectorizers import GUVectorize
#    from numbapro import float64
    from numba.vectorize import GUVectorize
    from numba import float64
    signature = str.join(',',['(n{})'.format(i) for i in range(len(args_size))])
    signature += '->(n)'.format(return_size)
    #args_types = [float64[:]]*(len(args_size)+1)
    args_types = [float64[:]]*(len(args_size)-1)
    args_types += [float64[:], float64[:]]
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

    gm = yaml_import(fname, compiler=second, order='columns', recipes=recipes)
    gmp = yaml_import(fname, compiler=first, order='columns', recipes=recipes)


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

    print(second)
    print(ss.shape)
    print(xx.shape)
    print(ee.shape)
    print(p.shape)

    tmp2 = g(ss,xx,ee,p)
    g(ss,xx,ee,p,tmp2)
    t1 = time.time()
    for i in range(50):
        #tmp2 = g(ss,xx,ee,p,tmp2)
        g(ss,xx,ee,p,tmp2)
    t2 = time.time()

    tmp2 = f(ss,xx,ss,xx,p)
    t3 = time.time()
    for i in range(50):
        f(ss,xx,ss,xx,p, tmp2)
    t4 = time.time()

    print('first {}'.format(t2-t1))
    print('second {}'.format(t4-t3))


    print("Error : {}".format( abs(tmp2 - tmp).max() ))
