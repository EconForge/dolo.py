from __future__ import division

#
#
#def compile_theano_old(vars, values):
#
#
#    """
#    :param vars: parameters
#    :param values: list of values (already triangular)
#
#
#    """
#
#    assert(len(vars)==len(values))
#
#
#    sub_dict = { v: '_'+str(v) for v in vars }
#    from dolo.compiler.compiler import DicPrinter
#
#    dp = DicPrinter(sub_dict)
#    strings = [ ]
#    for i,eq in enumerate( values ):
#        rhs = ( dp.doprint( eq) )
#        lhs = vars[i]
#        strings.append( '{} = OO + {}'.format(sub_dict[lhs],rhs))
#
#    source = """
#def test():
#
#    from theano import tensor as T
#    from theano import function
#
#    s = T.matrix('s')
#    x = T.matrix('x')
#    p = T.vector('p')
#
#    OO =  T.zeros((1,s.shape[1]))
#
#{declarations}
#
#{computations}
#
#    f = function( [{args}], [{vars}])
#
#    return f
#    """
#
#    source = source.format(
#        computations = str.join( '\n', [' '*4 +e for e in strings]),
#        declarations = '',
#        vars = str.join(', ', [sub_dict[v] for v in vars]),
#        args = 's,x,p'
#    )
#
#    return source
#


def compile_theano(vars, values, args_list, args_names, parms):


    """
    :param vars: parameters
    :param values: list of values (already triangular)


    """

    assert(len(vars)==len(values))


    sub_dict = { v: '_'+str(v) for v in vars }
    from dolo.compiler.compiler import DicPrinter

    dec = ''

    for s in args_names:
        dec += "    {} = T.matrix('{}')\n".format(s,s)

    dec += "    p = T.vector('p')\n"
    dec += '\n'


    for i,p in enumerate(parms):
        sn = '_'+str(p)
        sub_dict[p] = sn
        dec += ' '*4 + '{} = p[{}]\n'.format(sn,i)

    for i, l in enumerate( args_list):
        name = args_names[i]
        for j, e in enumerate(l):
            try:
                sn = e.safe_name()
            except Exception:
                sn = '_'+str(e)
            sub_dict[e] = sn
            dec += ' '*4 +  '{} = {}[{},:]\n'.format(sn,name,j)


#    print(dec)
#    print(sub_dict)


    dp = DicPrinter(sub_dict)
    strings = [ ]
    for i,eq in enumerate( values ):
        rhs = ( dp.doprint( eq) )
        lhs = vars[i]
        strings.append( '{} = {}'.format(sub_dict[lhs],rhs))

    source = """
def test():

    from theano import tensor as T
    from theano import function
    from theano.tensor import exp


{declarations}

#    OO =  T.zeros((1,s.shape[1]))

{computations}

    f = function( [{args}], [{vars}],mode='FAST_RUN',on_unused_input='ignore')

    return f
    """

    source = source.format(
        computations = str.join( '\n', [' '*4 +e for e in strings]),
        declarations = dec,
        vars = str.join(', ', [sub_dict[v] for v in vars]),
        args = str.join(', ', [str(v) for v in args_names] + ['p'] )    )
#    print(source)
    return source


def compile_theano_2(values, args_list, args_names, parms):


    """
    :param vars: parameters
    :param values: list of values (already triangular)


    """

    vars = ['_res_{}'.format(i) for i in range(len(values))]

    sub_dict = {}
    for e in vars:
        try:
            sn = e.safe_name()
        except Exception:
            sn = '_'+str(e)
        sub_dict[e] = sn

    from dolo.compiler.compiler import DicPrinter

    dec = ''

    for s in args_names:
        dec += "    {} = T.matrix('{}')\n".format(s,s)

    dec += "    p = T.vector('p')\n"

    for i,p in enumerate(parms):
        sn = '_'+str(p)
        sub_dict[p] = sn
        dec += ' '*4 + '{} = p[{}]\n'.format(sn,i)

    for i, l in enumerate( args_list):
        name = args_names[i]
        for j, e in enumerate(l):
            try:
                sn = e.safe_name()
            except Exception:
                sn = '_'+str(e)
            sub_dict[e] = sn
            dec += ' '*4 +  '{} = {}[{},:]\n'.format(sn,name,j)

    dp = DicPrinter(sub_dict)
    strings = [ ]
    for i,eq in enumerate( values ):
        rhs = ( dp.doprint( eq) )
        lhs = vars[i]
#        strings.append( '{} = OO + {}'.format(lhs,rhs))
        strings.append( '{} = {}'.format(lhs,rhs))

    source = """
def test():

    from theano import tensor as T
    from theano import function
    from theano.tensor import exp

{declarations}

#    OO =  T.zeros((1,s.shape[1]))

{computations}

    f = function([{args}], [{vars}],mode='FAST_RUN',on_unused_input='ignore')

    return f
    """

#    print(args_names)
    source = source.format(
        computations = str.join( '\n', [' '*4 +e for e in strings]),
        declarations = dec,
        vars = str.join(', ', [str(v) for v in vars]),
        args = str.join(', ', [str(v) for v in args_names] + ['p'] )
    )
    return source

def compile_function_2(values, args_list, args_names, parms, fname='anonymous_function', diff=True, vectorize=True, return_function=True):
    source = compile_theano_2(values, args_list, args_names, parms)
    exec(source)
    f = test()
    return f


def create_fun():
    import theano
    import theano.tensor as T

    s = T.matrix('s')
    x = T.matrix('x')
    p = T.vector('p')


    _z = s[0,:]*0 + 1 # exo
    _t = s[0,:]*0 + 1 # exo
    _y = _z
    _x = _y + _z
    _w = _x + _y + _z + _t

    eqs = [
        _z,_t,_y,_x,_w
    ]


    fun = theano.function([s,x,p],eqs)

    return fun



if __name__ == '__main__':

    import sympy
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
    s0 = numpy.array( [2,5])
    x0 = numpy.array( [2,2])
    p0 = numpy.array( [4,3] )

    s1 = numpy.column_stack( [s0]*20000 )
    x1 = numpy.column_stack( [x0]*20000 )
    p1 = numpy.array( [4,3] )



#    f = create_fun()
#
#    test = f(s1,x1,p0)
#    print(test)
    args_names = ['s','x']
#
    txt = compile_theano( ordered_vars, ordered_eqs, args_list, args_names, parms)
    exec(txt)
    f1 = test()

    solution = solve_triangular_system(sdict)

    vals = [sympy.sympify(solution[v]) for v in ordered_vars]

    from dolo.compiler.compiling import compile_function_2


    txt = compile_theano_2( vals, args_list, args_names, parms)
    exec(txt)
    f2 = test()

    g = compile_function_2( vals, args_list, args_names, parms)
    #f(s0,x0,p0)


    n_exp = 10000
    import time

    r = time.time()
    for i in range(n_exp):
        res = f1(s1,x1,p1)
#        res = numpy.row_stack(res)

    s = time.time()

    print('Time (theano) : '+ str(s-r))

    for i in range(n_exp):
        res = f2(s1,x1,p1)
#        res = numpy.row_stack(res)

    t = time.time()
    print('Time (theano2) : '+ str(t-s))


    for i in range(n_exp):
        res2 = g(s1,x1,p1,derivs=False)[0]
    #print(res)

    u = time.time()
    print('Time (numpy) : '+ str(u-t))

    err = abs(res - res2).max()

    print('Error : ' + str(err))
