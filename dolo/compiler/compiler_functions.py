
# TODO : this file deserves severe simplification !

def simple_global_representation(self, allow_future_shocks=True, solve_systems=False):


    resp = {}
    eq_g = self['equations_groups']
    v_g = self['variables_groups']
    if 'expectation' in eq_g:
        resp['f_eqs'] = [ eq.gap for eq in  eq_g['arbitrage'] + eq_g['expectation']] # TODO: complementarity conditions
        resp['controls'] = v_g['controls'] + v_g['expectations']
    else:
        resp['f_eqs'] = [ eq.gap for eq in  eq_g['arbitrage']] # TODO: complementarity conditions
        resp['controls'] = list( v_g['controls'] )

    resp['g_eqs'] = [eq.rhs for eq in  eq_g['transition'] ]


    if 'auxiliary' in eq_g:

        sdict = {}
        from dolo.symbolic.symbolic import timeshift
        auxies = list( eq_g['auxiliary'] )
        if 'auxiliary_2' in eq_g:
            auxies += list( eq_g['auxiliary_2'] )
        for eq in  auxies:
            sdict[eq.lhs] = eq.rhs
            sdict[eq.lhs(1)] = timeshift( eq.rhs, 1)
            sdict[eq.lhs(-1)] = timeshift( eq.rhs, -1)
        from dolo.misc.triangular_solver import solve_triangular_system
        sdict = solve_triangular_system(sdict)
        resp['a_eqs'] = [sdict[v] for v in v_g['auxiliary']]
        resp['auxiliaries'] = [v for v in v_g['auxiliary']]
        resp['f_eqs'] = [eq.subs(sdict) for eq in resp['f_eqs']]
        resp['g_eqs'] = [eq.subs(sdict) for eq in resp['g_eqs']]

    elif 'auxiliary_2' in eq_g:
        sdict = {}
        from dolo.misc.misc import timeshift
        auxies = eq_g['auxiliary_2']
        for eq in  auxies:
            sdict[eq.lhs] = eq.rhs
            sdict[eq.lhs(1)] = timeshift( eq.rhs, 1)
            sdict[eq.lhs(-1)] = timeshift( eq.rhs, -1)
        from dolo.misc.calculus import simple_triangular_solve
        sdict = simple_triangular_solve(sdict)
        resp['f_eqs'] = [eq.subs(sdict) for eq in resp['f_eqs']]
        resp['g_eqs'] = [eq.subs(sdict) for eq in resp['g_eqs']]


    if not allow_future_shocks:
        # future shocks are replaced by 0
        zero_shocks = {s(1):0 for s in self.shocks}
        resp['f_eqs'] = [ eq.subs(zero_shocks) for eq in resp['f_eqs'] ]

    if solve_systems:
        from dolo.misc.triangular_solver import solve_triangular_system
        system = {s: resp['g_eqs'][i] for i,s in enumerate(v_g['states'])}
        new_geqs = solve_triangular_system(system)
        resp['g_eqs'] = [new_geqs[s] for s in v_g['states']]

    resp['states'] = v_g['states']
    resp['shocks'] = self.shocks #['shocks_ordering'] # TODO: bad
    resp['parameters'] = self.parameters #['parameters_ordering']

    return resp


def model_to_fg(model,substitute_auxiliary=False, solve_systems=False, compiler='numpy'):

    sgm = simple_global_representation(model, solve_systems=solve_systems)

    controls = sgm['controls']
    states = sgm['states']
    parameters = sgm['parameters']
    shocks = sgm['shocks']



    f_eqs =  sgm['f_eqs']
    g_eqs =  sgm['g_eqs']

    controls_f = [c(1) for c in controls]
    states_f = [c(1) for c in states]
    controls_p = [c(-1) for c in controls]
    states_p = [c(-1) for c in states]
    shocks_f = [c(1) for c in shocks]


    args_g =  [states_p, controls_p, shocks]
    args_f =  [states, controls, states_f, controls_f, shocks_f]

    keep_auxiliary = 'a_eqs' in sgm

    if keep_auxiliary:
#        auxiliaries = sgm['auxiliaries']
        a_eqs = sgm['a_eqs']
        args_a = [states, controls]


    if compiler=='numpy':
        from dolo.compiler.compiling import compile_multiargument_function
        compile_multiargument_function
    elif compiler == 'theano':
        from dolo.compiler.compiling_theano import compile_multiargument_function
    elif compiler == 'numexpr':
        from dolo.compiler.compiling_numexpr import compile_multiargument_function
    else:
        raise Exception('Unknown compiler type : {}'.format(compiler))

    g = compile_multiargument_function(g_eqs, args_g, ['s','x','e'], parameters, 'g' )
    f = compile_multiargument_function(f_eqs, args_f, ['s','x','snext','xnext','e'], parameters, 'f' )

    if keep_auxiliary:
        a = compile_multiargument_function(a_eqs, args_a, ['s','x'], parameters, 'a' )
        return [f,g,a]
    else:
        return [f,g]


def model_to_fga(model, compiler='numpy'):

    from dolo.misc.triangular_solver import solve_triangular_system

    from dolo.misc.misc import timeshift

    eq_g = model['equations_groups']
    v_g = model['variables_groups']
    
    f_eqs =  [eq.gap for eq in eq_g['arbitrage']]
    g_eqs =  [eq for eq in eq_g['transition']]
    a_eqs =  [eq for eq in eq_g['auxiliary']]

    # auxiliaries_2 are simply replaced in all other types of equations
    a2_dict = {}
    a2_dict_g = {}

    if 'auxiliary_2' in eq_g:
        aux2_eqs = eq_g['auxiliary_2']
        dd = {eq.lhs: eq.rhs for eq in aux2_eqs}
        dd.update( { eq.lhs(1): timeshift(eq.rhs,1) for eq in aux2_eqs } )
        dd.update( { eq.lhs(-1): timeshift(eq.rhs,-1) for eq in aux2_eqs } )
        ds = solve_triangular_system(dd)
        
        f_eqs =  [eq.subs(ds) for eq in f_eqs]
        a_eqs =  [eq.subs(ds) for eq in a_eqs]
        g_eqs =  [eq.subs(ds) for eq in g_eqs]
    
    controls = v_g['controls']
    auxiliaries = v_g['auxiliary']
    states = v_g['states']

    parameters = model.parameters
    shocks = model.shocks

    dd = {eq.lhs: eq.rhs for eq in g_eqs}
    ds = solve_triangular_system(dd)
    g_eqs = [ds[eq.lhs] for eq in g_eqs]

    dd = {eq.lhs: eq.rhs for eq in a_eqs}
    ds = solve_triangular_system(dd)
    a_eqs = [ds[eq.lhs] for eq in a_eqs]


    auxiliaries_f = [c(1) for c in auxiliaries]
    auxiliaries_p = [c(-1) for c in auxiliaries]

    controls_f = [c(1) for c in controls]
    states_f = [c(1) for c in states]
    controls_p = [c(-1) for c in controls]
    states_p = [c(-1) for c in states]
    shocks_f = [c(1) for c in shocks]

    args_g =  [states_p, controls_p, auxiliaries_p, shocks]
    args_f =  [states, controls, states_f, controls_f, auxiliaries, auxiliaries_f, shocks_f]
    args_a =  [states, controls]


    if compiler=='numpy':
        from dolo.compiler.compiling import compile_multiargument_function
        compile_multiargument_function
    elif compiler == 'theano':
        from dolo.compiler.compiling_theano import compile_multiargument_function
    elif compiler == 'numexpr':
        from dolo.compiler.compiling_numexpr import compile_multiargument_function

    g = compile_multiargument_function(g_eqs, args_g, ['s','x','y','e'], parameters, 'g' )
    f = compile_multiargument_function(f_eqs, args_f, ['s','x','snext','xnext','y','ynext','e'], parameters, 'f' )
    a = compile_multiargument_function(a_eqs, args_a, ['s','x'], parameters, 'a' )

    return [f,a,g]