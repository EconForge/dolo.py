from __future__ import print_function

from dolo import *
import time

def test_rbc_model():


    model = yaml_import('examples/models/rbc.yaml')

    dr = approximate_controls(model)
    drg = time_iteration(model)
    t1 = time.time()
    drg = time_iteration(model)
    t2 = time.time()

    sim_dr = plot_decision_rule(model,dr,'k')

    from dolo.algos.dtcscc.vfi import evaluate_policy

    pol = evaluate_policy(model, dr, verbose=True)
    polg = evaluate_policy(model, drg, verbose=True)

    sim = simulate(model, dr, n_exp=0) # irf
    sim = simulate(model, dr, n_exp=2) # stochastic simulations (2 draws)
    # extract first simulation
    assert(len(sim[0]['k'])==40)



if __name__ == '__main__':

    test_rbc_model()
