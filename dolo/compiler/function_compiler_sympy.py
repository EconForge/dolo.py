from __future__ import division

import sympy
import numpy
import ast


def ast_to_sympy(expr):
    '''Converts an AST expression to a sympy expression (STUPID)'''

    from .codegen import to_source
    s = to_source(expr)
    return sympy.sympify(s)

def non_decreasing_series(n, size):
    '''Lists all combinations of 0,...,n-1 in increasing order'''

    if size == 1:
        return [[a] for a in range(n)]
    else:
        lc = non_decreasing_series(n, size-1)
        ll = []
        for l in lc:
            last = l[-1]
            for i in range(last, n):
                e = l + [i]
                ll.append(e)
        return ll

def higher_order_diff(eqs, syms, order=2):
    '''Takes higher order derivatives of a list of equations w.r.t a list of paramters'''

    import numpy

    eqs = list([sympy.sympify(eq) for eq in eqs])
    syms = list([sympy.sympify(s) for s in syms])

    neq = len(eqs)
    p = len(syms)

    D = [numpy.array(eqs)]
    orders = []

    for i in range(1,order+1):

        par = D[i-1]
        mat = numpy.empty([neq] + [p]*i, dtype=object)   #.append( numpy.zeros(orders))
        for ind in non_decreasing_series(p,i):

            ind_parent = ind[:-1]
            k = ind[-1]

            for line in range(neq):

                ii = [line] + ind
                iid = [line] + ind_parent
                eeq = par[ tuple(iid) ]
                mat[tuple(ii)] = eeq.diff(syms[k])

        D.append(mat)

    return D


def compile_higher_order_function(eqs, syms, params, order=2, funname='anonymous',
    return_code=False, compile=False):
    '''From a list of equations and variables, define a multivariate functions with higher order derivatives.'''

    from .function_compiler_ast import std_date_symbol, StandardizeDatesSimple

    all_vars = syms + [(p,0) for p in params]

    sds = StandardizeDatesSimple(all_vars)



    # TEMP: compatibility fix when eqs is an Odict:
    eqs = [eq for eq in eqs]

    if isinstance(eqs[0], str):
    # elif not isinstance(eqs[0], sympy.Basic):
    # assume we have ASTs
        eqs = list([ast.parse(eq).body[0] for eq in eqs])
        eqs_std = list( [sds.visit(eq) for eq in eqs] )
        eqs_sym = list( [ast_to_sympy(eq) for eq in eqs_std] )
    else:
        eqs_sym = eqs

    symsd = list( [std_date_symbol(a,b) for a,b in syms] )
    paramsd = list( [std_date_symbol(a,0) for a in params] )
    D = higher_order_diff(eqs_sym, symsd, order=order)

    txt = """def {funname}(x, p, order=1):

    import numpy
    from numpy import log, exp, tan, sqrt, pi
    from scipy.special import erfc

""".format(funname=funname)

    for i in range(len(syms)):
        txt += "    {} = x[{}]\n".format(symsd[i], i)

    txt += "\n"

    for i in range(len(params)):
        txt += "    {} = p[{}]\n".format(paramsd[i], i)

    txt += "\n    out = numpy.zeros({})".format(len(eqs))

    for i in range(len(eqs)):
        txt += "\n    out[{}] = {}".format(i, D[0][i])

    txt += """

    if order == 0:
        return out

"""
    if order >= 1:
        # Jacobian
        txt += "    out_1 = numpy.zeros(({},{}))\n".format(len(eqs), len(syms))

        for i in range(len(eqs)):
            for j in range(len(syms)):
                val = D[1][i,j]
                if val != 0:
                    txt += "    out_1[{},{}] = {}\n".format(i,j,D[1][i,j])

        txt += """

    if order == 1:
        return [out, out_1]

"""

    if order >= 2:
        # Hessian
        txt += "    out_2 = numpy.zeros(({},{},{}))\n".format(len(eqs), len(syms), len(syms))

        for n in range(len(eqs)):
            for i in range(len(syms)):
                for j in range(len(syms)):
                    val = D[2][n,i,j]
                    if val is not None:
                        if val != 0:
                            txt += "    out_2[{},{},{}] = {}\n".format(n,i,j,D[2][n,i,j])
                    else:
                        i1, j1 = sorted( (i,j) )
                        if D[2][n,i1,j1] != 0:
                            txt += "    out_2[{},{},{}] = out_2[{},{},{}]\n".format(n,i,j,n,i1,j1)

        txt += """

    if order == 2:
        return [out, out_1, out_2]

"""


    if order >= 3:
        # Hessian
        txt += "    out_3 = numpy.zeros(({},{},{},{}))\n".format(len(eqs), len(syms), len(syms), len(syms))

        for n in range(len(eqs)):
            for i in range(len(syms)):
                for j in range(len(syms)):
                    for k in range(len(syms)):
                        val = D[3][n,i,j,k]
                        if val is not None:
                            if val != 0:
                                txt += "    out_3[{},{},{},{}] = {}\n".format(n,i,j,k,D[3][n,i,j,k])
                        else:
                            i1, j1, k1 = sorted( (i,j,k) )
                            if D[3][n,i1,j1,k1] != 0:
                                txt += "    out_3[{},{},{},{}] = out_3[{},{},{},{}]\n".format(n,i,j,k,n,i1,j1,k1)

        txt += """

    if order == 3:
        return [out, out_1, out_2, out_3]
    """

    if return_code:
        return txt
    else:
        d = {}
        d['division'] = division

        exec(txt, d)
        fun = d[funname]

        if compile:
            raise Exception("Not implemented.")

        return fun


        # if compile:
        #     from numba import jit
        #     return jit(fun)
        # else:
        #     return fun


def test_deriv():

    # list of equations
    eqs = ['(a*x + 2*b)**2', 'y + exp(a + 2*b)*c(1)']

    # list of variables (time symbols)
    syms = [('a',0),('b',0),('c',1)]

    # list of parameters
    params = ['x','y']

    # compile a function with its derivatives
    fun = compile_higher_order_function(eqs, syms, params, order=3)


    # evaluate the function
    import numpy
    v = numpy.array([0.0, 0.0, 0.0])
    p = numpy.array([1.0, 0.5])


    f0,f1,f2,f3 = fun(v, p, order=3)

    assert( (f0[1]-0.5) == 0 )
    assert( (f1[1,2]==1))
    assert( (f2[0,1,1]==8))
    assert( (f2[0,1,0]==f2[0,1,0]))


if __name__ == '__main__':

    test_deriv()
