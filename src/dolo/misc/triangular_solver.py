import copy

def triangular_solver( incidence ):
    n = len(incidence)

    current = copy.deepcopy( incidence )

    solved = []
    max_steps = len(incidence)
    steps = 0

    while (steps < max_steps) and len(solved)<n :

        possibilities = [i for i in range(n) if (i not in solved) and (len(current[i])==0)]
        for i in possibilities:
            for e in current:
                if i in e:
                    e.remove(i)
            solved.append(i)
        steps += 1

    if len(solved) < n:
        raise Exception('System is not triangular')
    return solved

def solve_triangular_system( sdict, return_order=False ):
    var_order = sdict.keys()
    var_set = set(var_order)
    expressions = [sdict[k] for k in var_order]
    incidence = []
    for eq in expressions:
        try:
            atoms = eq.atoms()
        except:
            atoms = []
        vars = var_set.intersection( atoms )
        inds = [var_order.index(v) for v in vars]
        incidence.append(inds)
    ll = [list(l) for l in incidence]
    sol_order = triangular_solver(ll)

    if return_order:
        return [ var_order[i] for i in  sol_order]

    d = copy.copy(sdict)
    for i in sol_order:
        v = var_order[i]
        try:
            d[v] = d[v].subs(d)
        except: # in case d[v] is an int
            pass
    return d