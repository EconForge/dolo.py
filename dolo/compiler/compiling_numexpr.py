from __future__ import division

from dolo.symbolic.derivatives import DerivativesTree
from dolo.compiler.compiler import DicPrinter
from dolo.symbolic.symbolic import TSymbol, Shock, Variable

DerivativesTree.symbol_type = TSymbol

def compile_multiargument_function(equations, args_list, args_names, parms, fname='anonymous_function', return_text=False):
    """

    :param equations: list of sympy expressions
    :param args_list: list of lists of symbols (e.g. [[a_1,a_2], [b_1,b_2]])
    :param args_names: list of strings (['a','b']
    :param parms: list of symbols to be used as parameters
    :param fname: name of the python function to be generated
    :param diff: include symbolic derivates in generated function
    :param vectorize: arguments are vectorized (not parameters)
    :param return_function: if False, returns source code instead of a function
    :return: a python function f(a,b,p) where p is a vector of parameters and a, b are stacked vectors (explain better)
    """

    template = '{0}_{1}'


    declarations = ""

    sub_list = {}
    for i,args in enumerate(args_list):
        vec_name = args_names[i]
        for j,v in enumerate(args):
            sub_list[v] = template.format(vec_name,j)
            declarations += "    {0}_{1} = {0}[{1},...]\n".format(vec_name, j)

    for i,p in enumerate(parms):
        sub_list[p] = '{0}_{1}'.format('p',i)
        declarations += "    {0}_{1} = {0}[{1}]\n".format('p', i)


    text = '''
def {fname}({args_names}, {param_names}, derivs=True):

    import numexpr

    from numpy import exp, log

{declarations}

    n = {var}.shape[-1]

{content}

    return {return_names}
    '''

    from dolo.compiler.compiler import DicPrinter

    dp = DicPrinter(sub_list)

    def write_eqs(eq_l,outname='val'):
        eq_block = '    {0} = np.zeros( ({1},n) )\n'.format(outname, len(eq_l))
        for i,eq in enumerate(eq_l):
            #eq_block += "    {0}[{1},:] = numexpr.evaluate('{2}')\n".format(outname, i,  dp.doprint_numpy(str(eq)))
            tt = dp.doprint(eq)
            eq_block += "    {0}[{1},:] = numexpr.evaluate('{2}')\n".format(outname, i,  tt)
        return eq_block


    content = write_eqs(equations)

    return_names = 'val'
    text = text.format(
            fname = fname,
            declarations = declarations,
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




if __name__ == '__main__':

    import sympy
    from pprint import pprint

    [w,x,y,z,t] = vars = sympy.symbols('w, x, y, z, t')
    [a,b,c,d] = parms = sympy.symbols('a, b, c, d')

    [k_1,k_2] = s_sym = sympy.symbols('k_1, k_2')
    [x_1,x_2] = x_sym = sympy.symbols('x_1, x_2')

    args_list = [
        s_sym,
        x_sym
    ]

    from sympy import exp

    eqs = [
        x + y*k_2 + z*exp(x_1 + t),
        (y + z)**0.3,
        z,
        (k_1 + k_2)**0.3,
        k_2**x_1
    ]

    sdict = {s:eqs[i] for i,s in enumerate(vars) }

    from dolo.misc.triangular_solver import solve_triangular_system
    order = solve_triangular_system(sdict, return_order=True)

    ordered_vars  = [ v for v in order ]
    ordered_eqs = [ eqs[vars.index(v)] for v in order ]

    pprint(ordered_vars)
    pprint(ordered_eqs)


    import numpy

    floatX = numpy.float32
    s0 = numpy.array( [2,5], dtype=floatX)
    x0 = numpy.array( [2,2], dtype=floatX)
    p0 = numpy.array( [4,3], dtype=floatX)

    N = 2000
    s1 = numpy.column_stack( [s0]*N )
    x1 = numpy.column_stack( [x0]*N )
    p1 = numpy.array( [4,3, 6, 7], dtype=floatX )



    #    f = create_fun()
    #
    #    test = f(s1,x1,p0)
    #    print(test)
    args_names = ['s','x']
    #
#
    solution = solve_triangular_system(sdict)
    vals = [sympy.sympify(solution[v]) for v in ordered_vars]


    from dolo.compiler.compiling import compile_multiargument_function as numpy_compiler
    from dolo.compiler.compiling_theano import compile_multiargument_function as theano_compiler



    f_numexpr = compile_multiargument_function( vals, args_list, args_names, parms )
    f_numpy = numpy_compiler( vals, args_list, args_names, parms )
    f_theano = theano_compiler( vals, args_list, args_names, parms )



    n_exp = 1000
    import time


    r = time.time()
    for i in range(n_exp):
        res_numexpr = f_numexpr(s1,x1,p1)
    #        res = numpy.row_stack(res)
    s = time.time()

    print('Time (numexpr) : '+ str(s-r))





    r = time.time()
    for i in range(n_exp):
        res_theano = f_theano(s1,x1,p1)
        #        res = numpy.row_stack(res)
    s = time.time()

    print('Time (theano) : '+ str(s-r))


    r = time.time()
    for i in range(n_exp):
        res_numpy = f_numpy(s1,x1,p1)
        #        res = numpy.row_stack(res)
    s = time.time()

    print('Time (numpy) : '+ str(s-r))

    print( abs(res_numpy - res_theano).max() )
    print( abs(res_numexpr - res_numpy).max() )