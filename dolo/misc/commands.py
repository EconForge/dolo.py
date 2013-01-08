


def dynare(filename, order=1, plot=True):

    from dolo.misc.modfile import dynare_import
    from dolo.numeric.perturbations import solve_decision_rule
    from dolo.numeric.decision_rules import stoch_simul, impulse_response_function

    model = dynare_import(filename)

    dr = solve_decision_rule

    simul = stoch_simul(dr, plot=plot)

    return dict(
        model = model,
        dr = dr,
        simul = simul
    )



