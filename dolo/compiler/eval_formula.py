from ast import *
from dolang.symbolic import stringify, stringify_symbol
from dolang.codegen import to_source
from dolo.compiler.misc import CalibrationDict

import xarray

def eval_formula(expr:str, dataframe=None, context=None):
    '''
    expr: string
        Symbolic expression to evaluate.
        Example: `k(1)-delta*k(0)-i`
    table: (optional) pandas dataframe
        Each column is a time series, which can be indexed with dolo notations.
    context: dict or CalibrationDict
    '''

    print("Evaluating: {}".format(expr))
    if context is None:
        dd = {}  # context dictionary
    elif isinstance(context, CalibrationDict):
        dd = context.flat.copy()
    else:
        dd = context.copy()

    from dolang.symbolic import list_variables

    from dolang.parser import parse_string
    # compat since normalize form for parameters doesn't match calib dict.
    for k in [*dd.keys()]:
        dd[stringify_symbol(k)] = dd[k]


    expr_ast = parse_string(expr).value
    variables = list_variables(expr_ast)
    nexpr = stringify(expr_ast)
    print(expr)
    print(variables)

    from numpy import log, exp
    dd['log'] = log
    dd['exp'] = exp

    if dataframe is not None:

        import pandas as pd
        for (k,t) in variables:
            dd[stringify_symbol((k, t))] = dataframe[k].shift(t)
        dd['t'] = pd.Series(dataframe.index, index=dataframe.index)

    expr = to_source(nexpr)
    print(expr)
    print(dd.keys())
    res = eval(expr, dd)

    return res
