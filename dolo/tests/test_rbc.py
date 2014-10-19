from __future__ import print_function

from dolo import *

def test_rbc_model():


    model = yaml_import('examples/models/rbc.yaml')

    print(model)
    print(model.options)


    dr = approximate_controls(model)

    drg = global_solve(model)

    sim = plot_decision_rule(model,dr,'k')

    from dolo.algos.fg.vfi import evaluate_policy

    pol = evaluate_policy(model, dr, verbose=True)
    polg = evaluate_policy(model, drg, verbose=True)



if __name__ == '__main__':

    test_rbc_model()
