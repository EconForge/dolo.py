def convert_struct_to_dict(s):
    # imperfect but enough for now
    if len(s.dtype) == 0:
        if s.shape == (1,):
            return str(s[0])
        elif s.shape == (0,0):
            return []
        elif s.shape == (1,1):
            return s[0,0]
        else:
            return s
    else:
        # we suppose that we found a structure
        d = dict()
        ss = s[0,0] # actual content of the structure
        for n in ss.dtype.names:
            d[n] = convert_struct_to_dict(ss[n])
        return d

def map_function_to_expression(f,expr):
    if expr.is_Atom:
        return( f(expr) )
    else:
        l = list( expr._args )
        args = []
        for a in l:
            args.append(map_function_to_expression(f,a))
        return( expr.__class__(* args) )

def timeshift(expr, tshift):
    from dolo.symbolic.symbolic import TSymbol
    def fun(e):
        if isinstance(e,TSymbol):
            return e(tshift)
        else:
            return e
    return map_function_to_expression(fun, expr)