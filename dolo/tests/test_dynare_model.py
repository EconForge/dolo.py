from dolo import *

model = yaml_import("examples/models/rbc_dynare.yaml")
from dolo.algos.dynare.perturbations import *

from dolo.algos.dynare.simulations import *

dr = solve_decision_rule(model)


# sim = impulse_response_function(dr, 'e_z')
sim = stoch_simul(dr, 'e_z', plot=False)

print(sim)
print(sim.shape)

# print(sim)
# print(dr)
