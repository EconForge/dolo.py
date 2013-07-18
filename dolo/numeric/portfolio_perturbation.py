def portfolios_to_deterministic(model,pf_names):

    #######
    #######

    import re
    regex = re.compile('.*<=(.*)<=.*')
    for i,eq in enumerate(model.equations_groups['arbitrage']):
        from dolo.symbolic.symbolic import Variable, Equation
        m = regex.match(eq.tags['complementarity'])
        vs = m.group(1).strip()
        if vs in pf_names:
            v = Variable(vs)
            neq = Equation(v,0)
            neq.tag(**eq.tags)
            model.equations_groups['arbitrage'][i] = neq
            print(neq)

    print('Warning : initial model changed')
    model.update()

    return model

def solve_portfolio_model(model, pf_names, order=1):

    pf_model = model

    from dolo import Variable, Parameter, Equation
    import re

    n_states = len(pf_model.symbols_s['states'])
    states = pf_model.symbols_s['states']
    steady_states = [Parameter(v.name+'_bar') for v in pf_model.symbols_s['states']]
    n_pfs = len(pf_names)

    pf_vars = [Variable(v) for v in pf_names]
    res_vars = [Variable('res_'+str(i)) for i in range(n_pfs)]


    pf_parms = [Parameter('K_'+str(i)) for i in range(n_pfs)]
    pf_dparms = [[Parameter('K_'+str(i)+'_'+str(j)) for j in range(n_states)] for i in range(n_pfs)]

    from sympy import Matrix

    # creation of the new model

    import copy

    new_model = copy.copy(pf_model)

    new_model.symbols_s['controls'] += res_vars
    for v in res_vars + pf_vars:
        new_model.calibration_s[v] = 0


    new_model.symbols_s['parameters'].extend(steady_states)
    for p in pf_parms + Matrix(pf_dparms)[:]:
        new_model.symbols_s['parameters'].append(p)
        new_model.calibration_s[p] = 0

    compregex = re.compile('(.*)<=(.*)<=(.*)')

    to_be_added_1 = []
    to_be_added_2 = []

    expressions = Matrix(pf_parms) + Matrix(pf_dparms)*( Matrix(states) - Matrix(steady_states))

    for n,eq in enumerate(new_model.equations_groups['arbitrage']):
        if 'complementarity' in eq.tags:
            tg = eq.tags['complementarity']
            [lhs,mhs,rhs] = compregex.match(tg).groups()
            mhs = new_model.eval_string(mhs)
        else:
            mhs = None
        if mhs in pf_vars:
            i = pf_vars.index(mhs)
            neq = Equation(mhs, expressions[i])
            neq.tag(**eq.tags)
            eq_res = Equation(eq.gap, res_vars[i])
            eq_res.tag(eq_type='arbitrage')
            to_be_added_2.append(eq_res)
            new_model.equations_groups['arbitrage'][n] = neq
            to_be_added_1.append(neq)

    # new_model.equations_groups['arbitrage'].extend(to_be_added_1)
    new_model.equations_groups['arbitrage'].extend(to_be_added_2)
    new_model.update()

    for eq in new_model.equations_groups['arbitrage']:
        print(eq)

    print(new_model.parameters)

    print("number of equations {}".format(len(new_model.equations)))
    print("number of arbitrage equations {}".format( len(new_model.equations_groups['arbitrage'])) )

    print('parameters_ordering')
    print("number of parameters {}".format(new_model.symbols['parameters']))
    print("number of parameters {}".format(new_model.parameters))


    # now, we need to solve for the optimal portfolio coefficients
    from dolo.numeric.perturbations_to_states import approximate_controls

    dr = approximate_controls(new_model)
    print('ok')

    import numpy

    n_controls = len(model.symbols_s['controls'])

    def constant_residuals(x):
        d = {}
        for i in range(n_pfs):
            p = pf_parms[i]
            v = pf_vars[i]
            d[p] = x[i]
            d[v] = x[i]
        new_model.set_calibration(d)
            # new_model.parameters_values[p] = x[i]
            # new_model.init_values[v] = x[i]
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
        d = {}
        for i in range(n_pfs):
            p = pf_parms[i]
            v = pf_vars[i]
            d[p] = x[i]
            d[v] = x[i]
            for j in range(n_states):
                d[pf_dparms[i][j]] = dx[i,j]
        new_model.set_calibration(d)
        if return_dr:
            dr = approximate_controls(new_model, order=2)
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
    # model = yaml_import('examples/global_models/open_economy_with_pf_pert.yaml', compiler=None)
    model = yaml_import('/home/pablo/Documents/Research/Thesis/chapter_4/code/models/portfolios_capital.yaml', compiler=None)
    print(model.calibration)
    print(model)


    sol = solve_portfolio_model(model,['x_1','x_2'])
    print(sol.X_s)
