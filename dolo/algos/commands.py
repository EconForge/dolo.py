# this is for compatibility purposes only

from dolo.algos.time_iteration import time_iteration
from dolo.algos.simulations import simulate, response, tabulate, plot_decision_rule
from dolo.algos.value_iteration import evaluate_policy, solve_policy
from dolo.algos.steady_state import residuals
from dolo.algos.perturbation import perturbate

approximate_controls = perturbate
