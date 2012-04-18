def portfolios_to_deterministic(model,pf_names):

    #######
    #######

    import re
    regex = re.compile('.*<=(.*)<=.*')
    for i,eq in enumerate(model['equations']):
        from dolo.symbolic.symbolic import Variable, Equation
        if 'complementarity' in eq.tags:
            m = regex.match(eq.tags['complementarity'])
            vs = m.group(1).strip()
            if vs in pf_names:
                v = Variable(vs,0)
                neq = Equation(v,0)
                neq.tag(**eq.tags)
                model['equations'][i] = neq

    print('Warning : initial model changed')
    model.check()

    return model

def solve_portfolio_model(model, pf_names, order=1):

    pf_model = model

    from dolo import Variable, Parameter, Equation
    import re

    n_states = len(pf_model['variables_groups']['states'])
    states = pf_model['variables_groups']['states']
    steady_states = [Parameter(v.name+'_bar') for v in pf_model['variables_groups']['states']]
    n_pfs = len(pf_names)

    pf_vars = [Variable(v,0) for v in pf_names]
    res_vars = [Variable('res_'+str(i),0) for i in range(n_pfs)]


    pf_parms = [Parameter('K_'+str(i)) for i in range(n_pfs)]
    pf_dparms = [[Parameter('K_'+str(i)+'_'+str(j)) for j in range(n_states)] for i in range(n_pfs)]

    from sympy import Matrix

    # creation of the new model

    import copy
    print('Warning: initial model has been changed.')
    new_model = copy.copy(pf_model)
    new_model['variables_groups']['controls']+=res_vars
    new_model.check()

    for p in pf_parms + Matrix(pf_dparms)[:]:
        new_model['parameters_ordering'].append(p)
        new_model.parameters_values[p] = 0

    compregex = re.compile('(.*)<=(.*)<=(.*)')
    to_be_added = []

    expressions = Matrix(pf_parms) + Matrix(pf_dparms)*( Matrix(states) - Matrix(steady_states))

    for eq in new_model['equations_groups']['arbitrage']:
        if 'complementarity' in eq.tags:
            tg = eq.tags['complementarity']
            [lhs,mhs,rhs] = compregex.match(tg).groups()
            mhs = new_model.eval_string(mhs)
        else:
            mhs = None
        if mhs in pf_vars:

            i = pf_vars.index(mhs)
            eq_n = eq.tags['eq_number']
            neq = Equation(mhs, expressions[i])
            neq.tag(**eq.tags)
            new_model['equations'][eq_n] = neq
            eq_res = Equation(eq.gap, res_vars[i])
            eq_res.tag(eq_type='arbitrage')
            to_be_added.append(eq_res)

    new_model['equations'].extend(to_be_added)
    new_model.check()
    new_model.check_consistency()

    # now, we need to solve for the optimal portfolio coefficients
    from dolo.numeric.perturbations_to_states import approximate_controls


    import numpy

    n_controls = len(model['variables_groups']['controls'])

    def constant_residuals(x):
        for i in range(n_pfs):
            p = pf_parms[i]
            v = pf_vars[i]
            model.parameters_values[p] = x[i]
            model.init_values[v] = x[i]
        [X_bar, X_s, X_ss] = approximate_controls(new_model, order=2, return_dr=False)
        return X_bar[n_controls-n_pfs:n_controls]

    x0 = numpy.zeros(n_pfs)

    from dolo.numeric.solver import solver
    portfolios_0 = solver(constant_residuals, x0)

    print('Zero order portfolios : ')
    print(portfolios_0)

    print('Zero order: Final error:')
    print(constant_residuals(portfolios_0))

    def dynamic_residuals(X, return_dr=False):
        x = X[:,0]
        dx = X[:,1:]
        for i in range(n_pfs):
            p = pf_parms[i]
            v = pf_vars[i]
            model.parameters_values[p] = x[i]
            model.init_values[v] = x[i]
            for j in range(n_states):
                model.parameters_values[pf_dparms[i][j]] = dx[i,j]
        if return_dr:
            dr = approximate_controls(new_model, order=2, return_dr=True)
            return dr
        else:
            [X_bar, X_s, X_ss, X_sss] = approximate_controls(new_model, order=3, return_dr=False)
            crit = numpy.column_stack([
                X_bar[n_controls-n_pfs:n_controls],
                X_s[n_controls-n_pfs:n_controls,:],
            ])
            return crit



    y0 = numpy.column_stack([x0, numpy.zeros((n_pfs, n_states))])
    print('Initial error:')
    print(dynamic_residuals(y0))
    portfolios_1 = solver(dynamic_residuals, y0)

    print('First order portfolios : ')
    print(portfolios_1)

    print('Final error:')
    print(dynamic_residuals(portfolios_1))

    dr = dynamic_residuals(portfolios_1, return_dr=True)

    return dr



if __name__ == '__main__':
    from dolo import *
    model = yaml_import('/home/pablo/Documents/Research/Thesis/chapter_4/code/models/open_economy_with_pf_pert.yaml')
    sol = solve_portfolio_model(model,['x_1','x_2'])
    print(sol.X_s)