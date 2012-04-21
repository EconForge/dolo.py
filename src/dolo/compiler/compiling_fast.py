from __future__ import division

from dolo.symbolic.derivatives import DerivativesTree
from dolo.compiler.compiler import DicPrinter
from dolo.symbolic.symbolic import TSymbol, Shock, Variable

DerivativesTree.symbol_type = TSymbol

def compile_function_numexpr(equations, args_list, args_names, parms, fname='anonymous_function', diff=True, vectorize=True, return_function=True):
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
            eq_block += "    {0}[{1},:] = numexpr.evaluate('{2}')\n".format(outname, i,  dp.doprint_numpy(eq))
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

    if return_function:
        import numpy as np
        inf = np.inf
        exec text in locals(), globals()
        l = globals()
        return l[fname]
    else:
        return text