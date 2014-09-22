from __future__ import print_function

from dolo import *
from pylab import *

def test_rbc_model():



    model = yaml_import('examples/models/rbc.yaml')

    print(model)

    print(model.options)


    dr = approximate_controls(model)

    drg = global_solve(model)

    from dolo.algos.fg.simulations import plot_decision_rule
    sim = plot_decision_rule(model,dr,'k')
    print(sim)

    eri = array(sim[['z','k']])

    from dolo.algos.fg.vfi import evaluate_policy

    pol = evaluate_policy(model, dr, verbose=True)
    polg = evaluate_policy(model, drg, verbose=True)


    plot(sim['k'], polg(eri).ravel() - pol(eri).ravel())


    # plot(sim['k'], pol(eri).ravel())
    # plot(sim['k'], polg(eri).ravel())
    show()



if __name__ == '__main__':

    test_rbc_model()
