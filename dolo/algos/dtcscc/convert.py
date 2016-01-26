# TODO: As far as I can tell these functions aren't used anymore
from dolo.numeric.serial_operations import serial_multiplication as serial_mult


def get_f(model):

    if isinstance(model, dict):
        functions = model
    else:
        functions = model.functions

    ff = functions['arbitrage']
    # return ff
    # if 'auxiliary' not in functions:
    #     return ff
    #
    # aa = functions['auxiliary']
    #
    # def f(s, x, E, S, X, p, diff=False):
    #     return ff(s,x,E,S,X,p,diff=diff)
    #     # if diff:
            # [y, y_s, y_x] = aa(s, x, p, diff=True)
            # [Y, Y_S, Y_X] = aa(S, X, p, diff=True)
            # [r, r_s, r_x, r_y, r_E, r_S, r_X, r_Y] = ff(s, x, y, E, S, X, Y,
            #                                             p, diff=True)
            # r_s = r_s + serial_mult(r_y, y_s)
            # r_x = r_x + serial_mult(r_y, y_x)
            # r_S = r_S + serial_mult(r_Y, Y_S)
            # r_X = r_X + serial_mult(r_Y, Y_X)
            # return [r, r_s, r_x, r_E, r_S, r_X]


        # y = aa(s, x, p)
        # Y = aa (S, X, p)
        # r = ff(s, x, y, E, S, X, Y, p)
        # return r

    return ff


def get_g(model):

    if isinstance(model, dict):
        functions = model
    else:
        functions = model.functions

    gg = functions['transition']
    return gg
    #
    # if 'auxiliary' not in functions:
    #     return gg
    #
    # aa = functions['auxiliary']
    #
    # def g(s, x, e, p, diff=False):
    #     return gg(s,x,e,p,diff=diff)
    #     # if diff:
    #     #     [y, y_s, y_x] = aa(s, x, p, diff=True)
    #     #     [S, S_s, S_x, S_y, S_e] = gg(s, x, y, e, p, diff=True)
    #     #     S_s = S_s + serial_mult(S_y, y_s)
    #     #     S_x = S_x + serial_mult(S_y, y_x)
    #     #     return [S, S_s, S_x, S_e]
    #     # y = aa(s, x, p)
    #     # S = gg(s, x, y, e, p)
    #     # return S
    #
    # return g


def get_v(model):

    if isinstance(model, dict):
        functions = model
    else:
        functions = model.functions

    vv = functions['value']
    return vv
    # return vv
    # if 'auxiliary' not in functions:
    #     return gg
    #
    # aa = functions['auxiliary']
    #
    # def value(s, x, S, X, V, p, diff=False):
    #     if diff:
    #         [y, y_s, y_x] = aa(s, x, p, diff=True)
    #         [y, y_S, y_X] = aa(S, X, p, diff=True)
    #         [v, v_s, v_x, v_y, v_S, v_X, v_Y, v_V] = vv(s, x, y, S, X, Y, V,
    #                                                     p, diff=True)
    #         v_s = v_s + serial_mult(v_y, y_s)
    #         v_x = v_x + serial_mult(v_y, y_x)
    #         v_S = v_S + serial_mult(v_Y, Y_S)
    #         v_X = v_X + serial_mult(v_Y, Y_X)
    #         return [v, v_s, v_x, v_S, v_X, v_V]
    #     y = aa(s, x, p)
    #     Y = aa(S, X, p)
    #     v = vv(s, x, y, S, X, Y, V, p)
    #     return v
    #
    # return value


def get_h(model):

    if isinstance(model, dict):
        functions = model
    else:
        functions = model.functions

    hh = functions['expectation']
    return hh
    #
    # if 'auxiliary' not in functions:
    #     return hh
    #
    # aa = functions['auxiliary']
    #
    # def expectation(S, X, p, diff=False):
    #     if diff:
    #         [y, y_S, y_X] = aa(S, X, p, diff=True)
    #         [z, z_S, z_X, z_Y] = hh(S, X, Y, p, diff=True)
    #         z_S = z_S + serial_mult(z_Y, Y_S)
    #         z_X = z_X + serial_mult(z_Y, Y_X)
    #         return [z, z_S, z_X]
    #     Y = aa(S, X, p)
    #     z = hh(S, X, Y, p)
    #     return z
    #
    # return expectation


def convert_all(d):
    if 'auxiliary' not in d:
        return d
    new_d = dict()
    for k in d:
        fun = d[k]
        if k == 'arbitrage':
            new_fun = get_f(d)
        elif k == 'value':
            new_fun = get_v(d)
        elif k == 'transition':
            new_fun = get_g(d)
        elif k == 'expectation':
            new_fun = get_h(d)
        else:
            # TODO: ensure that we never reach that branch
            new_fun = fun
        new_d[k] = new_fun
    return new_d


def get_fg_functions(model):

    return [fun(model) for fun in [get_f, get_g]]
