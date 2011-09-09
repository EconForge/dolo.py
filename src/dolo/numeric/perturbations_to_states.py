from dolo.symbolic.model import Model

from dolo.symbolic.symbolic import Equation,Variable,Shock,Parameter

def simple_global_representation(self):
    resp = {}
    eq_g = self['equations_groups']
    v_g = self['variables_groups']
    resp['f_eqs'] = [ eq.gap for eq in  eq_g['arbitrage'] + eq_g['expectation']] # TODO: complementarity conditions
    resp['g_eqs'] = [eq.rhs for eq in  eq_g['transition'] ]
    resp['states'] = v_g['states']
    resp['controls'] = v_g['controls'] + v_g['expectations']
    resp['shocks'] = self.shocks #['shocks_ordering'] # TODO: bad
    resp['parameters'] = self.parameters #['parameters_ordering']
    return resp

def state_perturb(f_fun, g_fun):
    """
    Compute the perturbation of a system in the form:
    $E_t f(s_t,x_t,s_{t+1},x_{t+1})$
    $s_t = g(s_{t-1},x_{t-1},\\epsilon_t$
    
    :param f_fun: list of derivatives of f [order0, order1, order2, ...]
    :param g_fun: list of derivatives of g [order0, order1, order2, ...]
    """
    import numpy as np
    from dolo.numeric.extern.qz import qzordered
    from numpy.linalg import solve

    approx_order = len(f_fun) - 1 # order of approximation
    
    [f0,f1] = f_fun[:2]
    [g0,g1] = g_fun[:2]
    n_x = f1.shape[0]           # number of controls
    n_s = f1.shape[1]/2 - n_x   # number of states
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

    [S,T,Q,Z,eigval] = qzordered(A,B,n_s)
    
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


    if approx_order == 1:
        return [C,P,Q]

    # second order solution
    from dolo.numeric.tensor import sdot, mdot
    from numpy import dot
    from dolo.numeric.matrix_equations import solve_sylvester

    f2 = f_fun[2]
    g2 = g_fun[2]
    g_ss = g2[:,n_s,:n_s]
    g_sx = g2[:,n_s,n_s:n_v]
    g_xx = g2[:,n_s:n_v,n_s:n_v]

    #print g_ss.shape

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

    if approx_order == 2:
        return [X_s,X_ss]  # here, we don't approximate the law of motion of the states

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

    if approx_order == 3:
        return [X_s,X_ss,X_sss]




if __name__ == '__main__':

    #dolo.config.engine.dynare_config()
    #dolo.config.use_engine['sylvester'] = True

    from dolo.misc.yamlfile import yaml_import
    model = yaml_import('../../../examples/global_models/optimal_growth.yaml')
    
    gm = simple_global_representation(model)

    g_eqs = gm['g_eqs']
    g_args = [s(-1) for s in gm['states']] + [x(-1) for x in gm['controls']] + gm['shocks']
    f_eqs = gm['f_eqs']
    f_args = gm['states'] + gm['controls'] + [s(1) for s in gm['states']] + [x(1) for x in gm['controls']]
    p_args = gm['parameters']

    from dolo.compiler.compiling import compile_function

    g_fun = compile_function(g_eqs, g_args, p_args, 3)
    f_fun = compile_function(f_eqs, f_args, p_args, 3)

    # get steady_state
    [y,x,parms] = model.read_calibration()
    states_ss = [y[model.variables.index(v)] for v in gm['states']]
    controls_ss = [y[model.variables.index(v)] for v in gm['controls']]
    shocks_ss = x

    f = f_fun( states_ss + controls_ss + states_ss + controls_ss, parms)
    g = g_fun( states_ss + controls_ss + shocks_ss, parms)


    import time
    nit = 100
    t = time.time()
    for i in range(nit):
        state_perturb(f,g)
    s = time.time()
    print('Mean iteration :' + str((s-t)/nit) )
