# from ast import *
from dolang.symbolic import stringify, stringify_symbol, parse_string, list_variables
from dolang.codegen import to_source
from dolo.compiler.misc import CalibrationDict
from numpy import log, exp

import xarray


def eval_formula(expr: str, dataframe=None, context=None):
    '''
    expr: string
        Symbolic expression to evaluate.
        Example: `k(1)-delta*k(0)-i`
    table: (optional) pandas dataframe
        Each column is a time series, which can be indexed with dolo notations.
    context: dict or CalibrationDict
    '''

    if context is None:
        dd = {}  # context dictionary
    elif isinstance(context, CalibrationDict):
        dd = context.flat.copy()
    else:
        dd = context.copy()

    # compat since normalize form for parameters doesn't match calib dict.
    for k in [*dd.keys()]:
        dd[stringify_symbol(k)] = dd[k]

    expr_ast = parse_string(expr).value
    variables = list_variables(expr_ast)
    nexpr = stringify(expr_ast)


    dd['log'] = log
    dd['exp'] = exp

    if dataframe is not None:

        import pandas as pd
        for (k, t) in variables:
            dd[stringify_symbol((k, t))] = dataframe[k].shift(t)
        dd['t_'] = pd.Series(dataframe.index, index=dataframe.index)

    expr = to_source(nexpr)

    res = eval(expr, dd)

    return res
