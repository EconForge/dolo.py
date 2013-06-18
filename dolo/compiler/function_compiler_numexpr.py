from __future__ import division

from dolo.symbolic.derivatives import DerivativesTree
from dolo.symbolic.symbolic import TSymbol
from dolo.compiler.function_compiler import compile_multiargument_function as compile_multiargument_function_regular


DerivativesTree.symbol_type = TSymbol

def compile_multiargument_function(equations, args_list, args_names, parms, fname='anonymous_function', diff=True, return_text=False, order='rows'):
    return compile_multiargument_function_regular(equations, args_list, args_names, parms, fname=fname, diff=diff, return_text=return_text, use_numexpr=True, order=order)


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