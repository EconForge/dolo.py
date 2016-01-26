def get_fg_functions(model):

    if isinstance(model, dict):
        functions = model
    else:
        functions = model.functions

    ff = functions['arbitrage']
    gg = functions['transition']

    return [ff, gg]
    #
    # if 'auxiliary' not in model.functions:
    #     return [ff, gg]
    #
    # aa = functions['auxiliary']
    #
    # from dolo.numeric.serial_operations import serial_multiplication as serial_mult
    #
    # def f(m,s,x,M,S,X,p,diff=False):
    #     if diff:
    #         [y,y_m,y_s,y_x] = aa(m,s,x,p,diff=True)
    #         [Y,Y_M,Y_S,Y_X] = aa(M,S,X,p,diff=True)
    #         [r,r_m,r_s,r_x,r_y,r_M,r_S,r_X,r_Y] = ff(m,s,x,y,M,S,X,Y,p,diff=True)
    #         r_s = r_s + serial_mult(r_y,y_s)
    #         r_x = r_x + serial_mult(r_y,y_x)
    #         r_S = r_S + serial_mult(r_Y,Y_S)
    #         r_X = r_X + serial_mult(r_Y,Y_X)
    #         return [r, r_m, r_s, r_x, r_M, r_S, r_X]
    #     y = aa(m,s,x,p)
    #     Y = aa(M,S,X,p)
    #     r = ff(m,s,x,y,M,S,X,Y,p)
    #     return r
    #
    # def g(m,s,x,M,p,diff=False):
    #     if diff:
    #         [y,y_m,y_s,y_x] = aa(m,s,x,p,diff=True)
    #         [S,S_m,S_s,S_x,S_y,S_M] = gg(m,s,x,y,M,p,diff=True)
    #         S_s = S_s + serial_mult(S_y,y_s)
    #         S_x = S_x + serial_mult(S_y,y_x)
    #         return [S,S_m,S_s,S_x,S_M]
    #     y = aa(m,s,x,p)
    #     S = gg(m,s,x,y,M,p)
    #     return S
    #
    # return [f,g]
