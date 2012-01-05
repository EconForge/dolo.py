def full_functions(model):
    from dolo.misc.misc import timeshift

    eq_g = model['equations_groups']
    v_g = model['variables_groups']

    # auxiliaries_2 are simply replaced in all other types of equations
    a2_dict = {}
    a2_dict_g = {}

    if 'auxiliary_2' in eq_g:
        for eq in eq_g['auxiliary_2']:
            a2_dict[eq.lhs] = eq.rhs
            a2_dict[eq.lhs(1)] = timeshift( eq.rhs, 1 )
        for eq in eq_g['auxiliary_2']:
            a2_dict_g[eq.lhs(-1)] = timeshift( eq.rhs, -1 )

    controls = v_g['controls']
    auxiliaries = v_g['auxiliary']
    states = v_g['states']

    parameters = model.parameters
    shocks = model.shocks

    f_eqs =  [eq.gap.subs(a2_dict) for eq in eq_g['arbitrage']]

    g_eqs =  [eq for eq in eq_g['transition']]

    dd = {eq.lhs: eq.rhs for eq in g_eqs}
    from dolo.misc.calculus import simple_triangular_solve
    ds = simple_triangular_solve(dd)
    g_eqs = [ds[eq.lhs] for eq in g_eqs]

    dd = {eq.lhs: eq.rhs for eq in eq_g['auxiliary']}
    from dolo.misc.calculus import simple_triangular_solve
    ds = simple_triangular_solve(dd)
    a_eqs = [ds[eq.lhs] for eq in eq_g['auxiliary']]


    f_eqs = [eq.subs(a2_dict) for eq in f_eqs]
    g_eqs = [eq.subs(a2_dict_g) for eq in g_eqs]
    a_eqs = [eq.subs(a2_dict) for eq in a_eqs]


    auxiliaries_f = [c(1) for c in auxiliaries]
    auxiliaries_p = [c(-1) for c in auxiliaries]

    controls_f = [c(1) for c in controls]
    states_f = [c(1) for c in states]
    controls_p = [c(-1) for c in controls]
    states_p = [c(-1) for c in states]
    shocks_f = [c(1) for c in shocks]

    args_g =  [states_p, controls_p, auxiliaries_p, shocks]
    args_f =  [states, controls, states_f, controls_f, auxiliaries, auxiliaries_f]
    args_a =  [states, controls]

    from dolo.compiler.compiling import compile_function_2

    g = compile_function_2(g_eqs, args_g, ['s','x','y','e'], parameters, 'g' )
    f = compile_function_2(f_eqs, args_f, ['s','x','snext','xnext','y','ynext'], parameters, 'f' )
    a = compile_function_2(a_eqs, args_a, ['s','x'], parameters, 'a' )

    return [f,a,g]


def full_functions_2(model):
    resp = {}

    eq_g = model['equations_groups']
    v_g = model['variables_groups']

    controls = v_g['controls']
    auxiliaries = v_g['auxiliary']
    states = v_g['states']

    parameters = model.parameters
    shocks = model.shocks

    f_eqs =  [eq.gap for eq in eq_g['arbitrage']]
    g_eqs =  [eq.rhs for eq in eq_g['transition']]

    a_eqs = [eq.rhs for eq in eq_g['auxiliary']]

    auxiliaries_f = [c(1) for c in auxiliaries]
    controls_f = [c(1) for c in controls]
    states_f = [c(1) for c in states]
    controls_p = [c(-1) for c in controls]
    states_p = [c(-1) for c in states]
    shocks_f = [c(1) for c in shocks]

    args_g =  [states_p, controls_p, shocks]
    args_f =  [states, controls, states_f, controls_f, auxiliaries, auxiliaries_f, shocks_f]
    args_a =  [states, controls]

    g = compile_function_2(g_eqs, args_g, ['s','x','e'], parameters, 'g' )
    f = compile_function_2(f_eqs, args_f, ['s','x','snext','xnext','y','ynext','e'], parameters, 'f' )
    a = compile_function_2(a_eqs, args_a, ['s','x'], parameters, 'a' )

    return [f,a,g]
