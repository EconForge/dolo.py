from dolo import *

from dolo.algos.dynare.perturbations import *
from dolo.algos.dynare.simulations import *

def test_dynare_model():

    model = yaml_import("examples/models/rbc_dynare.yaml")
    dr = solve_decision_rule(model, order=2)
    print(dr['ys'])

    n_a = len(model.variables) + len(model.symbols['shocks'])

    irf = simulate(dr, n_exp=0, start='risky')
    print(irf)
    assert(irf.shape == (41,n_a))
    assert(irf.std().max()<0.00001)

    irf = simulate(dr, n_exp=0, start='deterministic')
    assert(irf.shape == (41,n_a))
    assert(irf.std().max()>0.00001)

    sim = simulate(dr, n_exp=1)
    assert(sim.shape == (41,n_a))
    assert(sim.std().max()>0.00001)

    sim2 = simulate(dr, n_exp=10)
    assert(sim2.shape == (10,41,n_a))

    sim_shock = simulate(dr, n_exp=0, shock=[0.01])



if __name__ == '__main__':

    test_dynare_model()
