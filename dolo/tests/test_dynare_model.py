from dolo import *

from dolo.algos.dynare.perturbations import *
from dolo.algos.dynare.simulations import *

def test_dynare_model():

    model = yaml_import("examples/models/rbc_dynare.yaml")
    dr = solve_decision_rule(model)
    sim = stoch_simul(dr, 'e_z', plot=False)

if __name__ == '__main__':

    test_dynare_model()
