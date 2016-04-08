import ast
from ast import BinOp, Compare, Sub

import sympy

from dolo.compiler.function_compiler_sympy import compile_higher_order_function
from dolo.compiler.function_compiler_sympy import ast_to_sympy
from dolo.numeric.decision_rules_states import CDR
from dolo.compiler.function_compiler_ast import (StandardizeDatesSimple,
                                                 std_date_symbol)
from dolo.compiler.function_compiler_ast import std_date_symbol

def timeshift(expr, variables, date):
    from sympy import Symbol
    from dolo.compiler.function_compiler_ast import std_date_symbol
    d = {Symbol(std_date_symbol(v, 0)): Symbol(std_date_symbol(v, date))
         for v in variables}
    return expr.subs(d)


def parse_equation(eq_string, vars, substract_lhs=True, to_sympy=False):

    sds = StandardizeDatesSimple(vars)

    eq = eq_string.split('|')[0]  # ignore complentarity constraints

    if '==' not in eq:
        eq = eq.replace('=', '==')

    expr = ast.parse(eq).body[0].value
    expr_std = sds.visit(expr)

    from dolo.compiler.codegen import to_source

    if isinstance(expr_std, Compare):
        lhs = expr.left
        rhs = expr.comparators[0]
        if substract_lhs:
            expr_std = BinOp(left=rhs, right=lhs, op=Sub())
        else:
            if to_sympy:
                return [ast_to_sympy(lhs), ast_to_sympy(rhs)]
            return [lhs, rhs]

    if to_sympy:
        return ast_to_sympy(expr_std)
    else:
        return expr_std


def model_to_fg(model, order=2):

    if hasattr(model, '__higher_order_functions__') and model.__highest_order__  >= order:
        f = model.__higher_order_functions__['f']
        g = model.__higher_order_functions__['g']
        return [f, g]

    all_variables = sum([model.symbols[e] for e in model.symbols if e != 'parameters'], [])
    all_dvariables = ([(d, 0) for d in all_variables] +
                      [(d, 1) for d in all_variables] +
                      [(d, -1) for d in all_variables])
    psyms = [(e,0) for e in model.symbols['parameters']]

    if hasattr(model.symbolic, 'definitions'):
        definitions = model.symbolic.definitions
    else:
        definitions = {}

    ddef = dict()
    for k in definitions:
        v = parse_equation(definitions[k], all_dvariables + psyms, to_sympy=True)
        ddef[sympy.Symbol(k)] = v

    # all_sym_variables = [std_date_symbol(s,0) for s in all_variables]

    params = model.symbols['parameters']

    f_eqs = model.symbolic.equations['arbitrage']
    f_eqs = [parse_equation(eq, all_dvariables + psyms, to_sympy=True) for eq in f_eqs]
    f_eqs = [eq.subs(ddef) for eq in f_eqs]  # TODO : replace it everywhere else

    y_eqs = model.symbolic.equations['auxiliary']
    syms = [(e, 0) for e in model.symbols['states']] + \
           [(e, 0) for e in model.symbols['controls']] + \
           [(e, 0) for e in model.symbols['auxiliaries']]

    # Solve recursively
    y_eqs = [parse_equation(eq, all_dvariables+psyms, to_sympy=True, substract_lhs=False) for eq in y_eqs]
    d = {}
    for eq in y_eqs:
        d[eq[0]] = eq[1].subs(d)

    # Construct dictionary
    for k in list(d.keys()):
        d[timeshift(k,all_variables,1)] = timeshift(d[k],all_variables,1)
        d[timeshift(k,all_variables,-1)] = timeshift(d[k],all_variables,-1)
    f_eqs = [eq.subs(d) for eq in f_eqs]

    g_eqs = model.symbolic.equations['transition']
    g_eqs = [parse_equation(eq, all_dvariables + psyms, to_sympy=True, substract_lhs=False) for eq in g_eqs]

    #solve_recursively
    from collections import OrderedDict
    dd = OrderedDict()
    for eq in g_eqs:
        dd[eq[0]] = eq[1].subs(dd).subs(d)
    g_eqs = dd.values()


    syms = [(e,-1) for e in model.symbols['states']] + \
           [(e,-1) for e in model.symbols['controls']] + \
           [(e,-1) for e in model.symbols['auxiliaries']] + \
           [(e,0) for e in model.symbols['shocks']] + \
           [(e,0) for e in model.symbols['states']]


    f_syms = [(e,0) for e in model.symbols['states']] + \
                [(e,0) for e in model.symbols['controls']] + \
                [(e,1) for e in model.symbols['states']] + \
                [(e,1) for e in model.symbols['controls']]

    g_syms = [(e,-1) for e in model.symbols['states']] + \
                [(e,-1) for e in model.symbols['controls']] + \
                [(e,0) for e in model.symbols['shocks']]

    f = compile_higher_order_function(f_eqs, f_syms, params, order=order,
        funname='f', return_code=False, compile=False)
    g = compile_higher_order_function(g_eqs, g_syms, params, order=order,
        funname='g', return_code=False, compile=False)

    # cache result
    model.__higher_order_functions__ = dict(f=f, g=g)
    model.__highest_order__ = order

    return [f, g]




def approximate_controls(model, order=1, lambda_name=None, return_dr=True, steady_state=None, verbose=True, eigmax=1.0+1e-6):

    assert(model.model_type=='dtcscc')

    [f_fun, g_fun] = model_to_fg(model, order=order)


    import numpy

    states = model.symbols['states']
    controls = model.symbols['controls']

    parms = model.calibration['parameters']
    sigma = model.covariances

    if steady_state is None:
        calibration = model.calibration
    else:
        calibration = steady_state

    states_ss = calibration['states']
    controls_ss = calibration['controls']
    shocks_ss = calibration['shocks']

    f_args_ss = numpy.concatenate( [states_ss, controls_ss, states_ss, controls_ss] )
    g_args_ss = numpy.concatenate( [states_ss, controls_ss, shocks_ss] )


    f = f_fun( f_args_ss, parms, order=order)
    g = g_fun( g_args_ss, parms, order=order)


    if lambda_name:
        epsilon = 0.001
        sigma_index = [p.name for p in model.parameters].index(lambda_name)
        pert_parms = parms.copy()
        pert_parms[sigma_index] += epsilon

        g_pert = g_fun(g_args_ss, pert_parms)
        sig2 = (g_pert[0] - g[0])/epsilon*2
        sig2_s = (g_pert[1] - g[1])/epsilon*2
        pert_sol = state_perturb(f, g, sigma, sigma2_correction = [sig2, sig2_s], eigmax=eigmax, verbose=verbose)

    else:
        g = g_fun( g_args_ss, parms, order=order)
        pert_sol = state_perturb(f, g, sigma, eigmax=eigmax, verbose=verbose )


    n_s = len(states_ss)
    n_c = len(controls_ss)

    if order == 1:
        if return_dr:
            S_bar = numpy.array( states_ss )
            X_bar = numpy.array( controls_ss )
            # add transitions of states to the d.r.



            X_s = pert_sol[0]
            A = g[1][:,:n_s] + numpy.dot( g[1][:,n_s:n_s+n_c], X_s )
            B = g[1][:,n_s+n_c:]
            dr = CDR([S_bar, X_bar, X_s])
            dr.A = A
            dr.B = B
            dr.sigma = sigma
            return dr

        return [controls_ss] + pert_sol

    if order == 2:
        [[X_s,X_ss],[X_tt]] = pert_sol
        X_bar = controls_ss + X_tt/2
        if return_dr:
            S_bar = states_ss
            S_bar = numpy.array(S_bar)
            X_bar = numpy.array(X_bar)
            dr = CDR([S_bar, X_bar, X_s, X_ss])
            A = g[1][:,:n_s] + numpy.dot( g[1][:,n_s:n_s+n_c], X_s )
            B = g[1][:,n_s+n_c:]
            dr.sigma = sigma
            dr.A = A
            dr.B = B
            return dr
        return [X_bar, X_s, X_ss]


    if order == 3:
        [[X_s,X_ss,X_sss],[X_tt, X_stt]] = pert_sol
        X_bar = controls_ss + X_tt/2
        X_s = X_s + X_stt/2
        if return_dr:
            S_bar = states_ss
            dr = CDR([S_bar, X_bar, X_s, X_ss, X_sss])
            dr.sigma = sigma
            return dr
        return [X_bar, X_s, X_ss, X_sss]

def state_perturb(f_fun, g_fun, sigma, sigma2_correction=None, verbose=True, eigmax=1.0+1e-6):
    """Computes a Taylor approximation of decision rules, given the supplied derivatives.

    The original system is assumed to be in the the form:

    .. math::

        E_t f(s_t,x_t,s_{t+1},x_{t+1})

        s_t = g(s_{t-1},x_{t-1}, \\lambda \\epsilon_t)

    where :math:`\\lambda` is a scalar scaling down the risk.  the solution is a function :math:`\\varphi` such that:

    .. math::

        x_t = \\varphi ( s_t, \\sigma )

    The user supplies, a list of derivatives of f and g.

    :param f_fun: list of derivatives of f [order0, order1, order2, ...]
    :param g_fun: list of derivatives of g [order0, order1, order2, ...]
    :param sigma: covariance matrix of :math:`\\epsilon_t`
    :param sigma2_correction: (optional) first and second derivatives of g w.r.t. sigma if :math:`g` explicitely depends
        :math:`sigma`


    Assuming :math:`s_t` ,  :math:`x_t` and :math:`\\epsilon_t` are vectors of size
    :math:`n_s`, :math:`n_x`  and :math:`n_x`  respectively.
    In general the derivative of order :math:`i` of :math:`f`  is a multimensional array of size :math:`n_x \\times (N, ..., N)`
    with :math:`N=2(n_s+n_x)` repeated :math:`i` times (possibly 0).
    Similarly the derivative of order :math:`i` of :math:`g`  is a multidimensional array of size :math:`n_s \\times (M, ..., M)`
    with :math:`M=n_s+n_x+n_2` repeated :math:`i` times (possibly 0).



    """

    import numpy as np
    from numpy.linalg import solve

    approx_order = len(f_fun) - 1 # order of approximation

    [f0,f1] = f_fun[:2]

    [g0,g1] = g_fun[:2]
    n_x = f1.shape[0]           # number of controls
    n_s = f1.shape[1]//2 - n_x   # number of states
    n_e = g1.shape[1] - n_x - n_s
    n_v = n_s + n_x

    f_s = f1[:,:n_s]
    f_x = f1[:,n_s:n_s+n_x]
    f_snext = f1[:,n_v:n_v+n_s]
    f_xnext = f1[:,n_v+n_s:]

    g_s = g1[:,:n_s]
    g_x = g1[:,n_s:n_s+n_x]
    g_e = g1[:,n_v:]

    A = np.row_stack([
        np.column_stack( [ np.eye(n_s), np.zeros((n_s,n_x)) ] ),
        np.column_stack( [ -f_snext    , -f_xnext             ] )
    ])
    B = np.row_stack([
        np.column_stack( [ g_s, g_x ] ),
        np.column_stack( [ f_s, f_x ] )
    ])



    from dolo.numeric.extern.qz import qzordered
    [S,T,Q,Z,eigval] = qzordered(A,B,eigmax)

    # Check Blanchard=Kahn conditions
    n_big_one = sum(eigval>eigmax)
    n_expected = n_x
    if verbose:
        print( "There are {} eigenvalues greater than 1. Expected: {}.".format( n_big_one, n_x ) )

    if n_big_one != n_expected:
        raise Exception("There should be exactly {} eigenvalues greater than one. Not {}.".format(n_x, n_big_one))

    Q = Q.real # is it really necessary ?
    Z = Z.real

    Z11 = Z[:n_s,:n_s]
    Z12 = Z[:n_s,n_s:]
    Z21 = Z[n_s:,:n_s]
    Z22 = Z[n_s:,n_s:]
    S11 = S[:n_s,:n_s]
    T11 = T[:n_s,:n_s]

    # first order solution
    C = solve(Z11.T, Z21.T).T
    P = np.dot(solve(S11.T, Z11.T).T , solve(Z11.T, T11.T).T )
    Q = g_e

    if False:
        from numpy import dot
        test = f_s + dot(f_x,C) + dot( f_snext, g_s + dot(g_x,C) ) + dot(f_xnext, dot( C, g_s + dot(g_x,C) ) )
        print('Error: ' + str(abs(test).max()))

    if approx_order == 1:
        return [C]

    # second order solution
    from dolo.numeric.tensor import sdot, mdot
    from numpy import dot
    from dolo.numeric.matrix_equations import solve_sylvester

    f2 = f_fun[2]
    g2 = g_fun[2]
    g_ss = g2[:,:n_s,:n_s]
    g_sx = g2[:,:n_s,n_s:n_v]
    g_xx = g2[:,n_s:n_v,n_s:n_v]

    X_s = C



    V1_3 = g_s + dot(g_x,X_s)
    V1 = np.row_stack([
        np.eye(n_s),
        X_s,
        V1_3,
        dot( X_s, V1_3 )
    ])

    K2 = g_ss + 2 * sdot(g_sx,X_s) + mdot(g_xx,[X_s,X_s])
    #L2 =
    A = f_x + dot( f_snext + dot(f_xnext,X_s), g_x)
    B = f_xnext
    C = V1_3
    D = mdot(f2,[V1,V1]) + sdot(f_snext + dot(f_xnext,X_s),K2)

    X_ss = solve_sylvester(A,B,C,D)

#    test = sdot( A, X_ss ) + sdot( B,  mdot(X_ss,[V1_3,V1_3]) ) + D


    # if sigma is not None:
    if True:
        g_ee = g2[:,n_v:,n_v:]

        v = np.row_stack([
            g_e,
            dot(X_s,g_e)
        ])

        K_tt = mdot( f2[:,n_v:,n_v:], [v,v] )
        K_tt += sdot( f_snext + dot(f_xnext,X_s) , g_ee )
        K_tt += mdot( sdot( f_xnext, X_ss), [g_e, g_e] )
        K_tt = np.tensordot( K_tt, sigma, axes=((1,2),(0,1)))

        if sigma2_correction is not None:
            K_tt += sdot( f_snext + dot(f_xnext,X_s) , sigma2_correction[0] )

        L_tt = f_x  + dot(f_snext, g_x) + dot(f_xnext, dot(X_s, g_x) + np.eye(n_x) )
        X_tt = solve( L_tt, - K_tt)

    if approx_order == 2:
        return [[X_s,X_ss],[X_tt]]

    # third order solution

    f3 = f_fun[3]
    g3 = g_fun[3]
    g_sss = g3[:,:n_s,:n_s,:n_s]
    g_ssx = g3[:,:n_s,:n_s,n_s:n_v]
    g_sxx = g3[:,:n_s,n_s:n_v,n_s:n_v]
    g_xxx = g3[:,n_s:n_v,n_s:n_v,n_s:n_v]

    V2_3 = K2 + sdot(g_x,X_ss)
    V2 = np.row_stack([
        np.zeros( (n_s,n_s,n_s) ),
        X_ss,
        V2_3,
        dot( X_s, V2_3 ) + mdot(X_ss,[V1_3,V1_3])
    ])

    K3 = g_sss + 3*sdot(g_ssx,X_s) + 3*mdot(g_sxx,[X_s,X_s]) + 2*sdot(g_sx,X_ss)
    K3 += 3*mdot( g_xx,[X_ss,X_s] ) + mdot(g_xxx,[X_s,X_s,X_s])
    L3 = 3*mdot(X_ss,[V1_3,V2_3])

    # A = f_x + dot( f_snext + dot(f_xnext,X_s), g_x) # same as before
    # B = f_xnext # same
    # C = V1_3 # same
    D = mdot(f3,[V1,V1,V1]) + 3*mdot(f2,[ V2,V1 ]) + sdot(f_snext + dot(f_xnext,X_s),K3)
    D += sdot( f_xnext, L3 )

    X_sss = solve_sylvester(A,B,C,D)

    # now doing sigma correction with sigma replaced by l in the subscripts

    # if not sigma is None:
    if True:
        g_se= g2[:,:n_s,n_v:]
        g_xe= g2[:,n_s:n_v,n_v:]

        g_see= g3[:,:n_s,n_v:,n_v:]
        g_xee= g3[:,n_s:n_v,n_v:,n_v:]


        W_l = np.row_stack([
            g_e,
            dot(X_s,g_e)
        ])

        I_e = np.eye(n_e)

        V_sl = g_se + mdot( g_xe, [X_s, np.eye(n_e)])

        W_sl = np.row_stack([
            V_sl,
            mdot( X_ss, [ V1_3, g_e ] ) + sdot( X_s, V_sl)
        ])

        K_ee = mdot(f3[:,:,n_v:,n_v:], [V1, W_l, W_l ])
        K_ee += 2 * mdot( f2[:,n_v:,n_v:], [W_sl, W_l])

        # stochastic part of W_ll

        SW_ll = np.row_stack([
            g_ee,
            mdot(X_ss, [g_e, g_e]) + sdot(X_s, g_ee)
        ])

        DW_ll = np.concatenate([
            X_tt,
            dot(g_x, X_tt),
            dot(X_s, sdot(g_x,X_tt )) + X_tt
        ])

        K_ee += mdot( f2[:,:,n_v:], [V1, SW_ll])

        K_ = np.tensordot(K_ee, sigma, axes=((2,3),(0,1)))

        K_ += mdot(f2[:,:,n_s:], [V1, DW_ll])

        def E(vec):
            n = len(vec.shape)
            return np.tensordot(vec,sigma,axes=((n-2,n-1),(0,1)))

        L = sdot(g_sx,X_tt) + mdot(g_xx,[X_s,X_tt])

        L += E(g_see + mdot(g_xee,[X_s,I_e,I_e]) )

        M = E( mdot(X_sss,[V1_3, g_e, g_e]) + 2*mdot(X_ss, [V_sl,g_e]) )
        M += mdot( X_ss, [V1_3, E( g_ee ) + sdot(g_x, X_tt)] )


        A = f_x + dot( f_snext + dot(f_xnext,X_s), g_x) # same as before
        B = f_xnext # same
        C = V1_3 # same
        D = K_ + dot( f_snext + dot(f_xnext,X_s), L) + dot( f_xnext, M )

        if sigma2_correction is not None:
            g_sl = sigma2_correction[1][:,:n_s]
            g_xl = sigma2_correction[1][:,n_s:(n_s+n_x)]
            D += dot( f_snext + dot(f_xnext,X_s), g_sl + dot(g_xl,X_s) )   # constant

        X_stt = solve_sylvester(A,B,C,D)

    if approx_order == 3:
        # if sigma is None:
        #     return [X_s,X_ss,X_sss]
        # else:
        #     return [[X_s,X_ss,X_sss],[X_tt, X_stt]]
        return [[X_s,X_ss,X_sss],[X_tt, X_stt]]


if __name__ == '__main__':
    from dolo import yaml_import
    model = yaml_import('examples/models/rbc.yaml')
    # model = yaml_import('/home/pablo/Programming/papers/finint/models/integration_B_pert.yaml')

    import time
    t1 = time.time()
    dr = approximate_controls(model, order=2, eigmax=1.000001)
    print(dr.X_s)
    # print(dr.X_ss)
    # print(dr.X_sss)
    t2 = time.time()
    print("Elapsed {}".format(t2-t1))

    t1 = time.time()
    dr = approximate_controls(model, order=2, eigmax=1.000001)
    print(dr.X_s)
    # print(dr.X_ss)
    # print(dr.X_sss)
    t2 = time.time()
    print("Elapsed {}".format(t2-t1))
