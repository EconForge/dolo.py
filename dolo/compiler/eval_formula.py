def eval_formula(expr, dataframe=None, context=None):
    '''
    expr: string
        Symbolic expression to evaluate.
        Example: `k(1)-delta*k-i`
    table: (optional) pandas dataframe
        Each column is a time series, which can be indexed with dolo notations.
    context: dict or CalibrationDict
    '''
    from dolo.compiler.function_compiler_ast import std_date_symbol, StandardizeDatesSimple, to_source
    from dolo.compiler.misc import CalibrationDict


    if context is None:
        dd = {} # context dictionary
    elif isinstance(context,CalibrationDict):
        dd = context.full.copy()
    else:
        dd = context.copy()

    from numpy import log, exp
    dd['log'] = log
    dd['exp'] = exp

    if dataframe is not None:

        import pandas as pd
        tvariables = dataframe.columns
        for k in tvariables:
            if k in dd:
                dd[k+'_ss'] = dd[k] # steady-state value
            dd[k] = dataframe[k]
            for h in range(1,3): # maximum number of lags
                dd[std_date_symbol(k,-h)] = dataframe[k].shift( h)
                dd[std_date_symbol(k, h)] = dataframe[k].shift(-h)
        dd['t'] =  pd.Series(dataframe.index, index=dataframe.index)

        import ast
        expr_ast = ast.parse(expr).body[0].value
        nexpr = StandardizeDatesSimple(tvariables).visit(expr_ast)
        expr = to_source(nexpr)

    res = eval(expr, dd)

    return res
