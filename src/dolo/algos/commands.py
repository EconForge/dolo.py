# this is for compatibility purposes only

from dolo.algos.time_iteration import time_iteration
from dolo.algos.perfect_foresight import deterministic_solve
from dolo.algos.simulations import simulate, response, tabulate, plot_decision_rule
from dolo.algos.value_iteration import evaluate_policy, value_iteration
from dolo.algos.improved_time_iteration import improved_time_iteration
from dolo.algos.newton import newton
from dolo.algos.steady_state import residuals
from dolo.algos.perturbation import perturb
from dolo.algos.ergodic import ergodic_distribution
from dolo.numeric.residuals import F, dF_1, F_and_dF_1

approximate_controls = perturb

