def get_fg_functions(model):

    self = model

    ff = self.functions['arbitrage']
    gg = self.functions['transition']

    if model.model_type == 'fg':
        return [ff, gg]

    aa = self.functions['auxiliary']

    from dolo.numeric.serial_operations import serial_multiplication as serial_mult

    def f(s,x,E,S,X,p,diff=False):
        if diff:
            [y,y_s,y_x] = aa(s,x,p,diff=True)
            [Y,Y_S,Y_X] = aa(S,X,p,diff=True)
            [r,r_s,r_x,r_y,r_E,r_S,r_X,r_Y] = ff(s,x,y,E,S,X,Y,p,diff=True)
            r_s = r_s + serial_mult(r_y,y_s)
            r_x = r_x + serial_mult(r_y,y_x)
            r_S = r_S + serial_mult(r_Y,Y_S)
            r_X = r_X + serial_mult(r_Y,Y_X)
            return [r, r_s, r_x, r_E, r_S, r_X]
        y = aa(s,x,p)
        Y = aa(S,X,p)
        r = ff(s,x,y,E,S,X,Y,p)
        return r

    def g(s,x,e,p,diff=False):
        if diff:
            [y,y_s,y_x] = aa(s,x,p,diff=True)
            [S,S_s,S_x,S_y,S_e] = gg(s,x,y,e,p,diff=True)
            S_s = S_s + serial_mult(S_y,y_s)
            S_x = S_x + serial_mult(S_y,y_x)
            return [S,S_s,S_x,S_e]
        y = aa(s,x,p)
        S = gg(s,x,y,e,p)
        return S
    #
    # from dolo.compiler.function_compiler import standard_function
    # f = standard_function(f, len(model.symbols['controls']))
    # g = standard_function(g, len(model.symbols['states']))
    return [f,g]
