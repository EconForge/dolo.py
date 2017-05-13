# this is for compatibility purposes only

from dolo.algos.time_iteration import time_iteration
from dolo.algos.simulations import simulate, tabulate, plot_decision_rule
from dolo.algos.value_iteration import evaluate_policy, solve_policy
from dolo.algos.steady_state import residuals

#
# def approximate_controls(model, *args, **kwargs):
#
#     if model.is_dtcscc():
#         order = kwargs.get('order')
#         if order is None or order==1:
#             from dolo.algos.dtcscc.perturbations import approximate_controls
#         else:
#             from dolo.algos.dtcscc.perturbations_higher_order import approximate_controls
#     else:
#         raise Exception("Model type {} not supported.".format(model.model_type))
#
#     return approximate_controls(model, *args, **kwargs)
#
#
#
# def plot_decision_rule(model, *args, **kwargs):
#
#     if model.is_dtcscc():
#         from dolo.algos.dtcscc.simulations import plot_decision_rule
#     elif model.is_dtmscc():
#         from dolo.algos.dtmscc.simulations import plot_decision_rule
#     else:
#         raise Exception("Model type {} not supported.".format(model.model_type))
#
#     return plot_decision_rule(model, *args, **kwargs)
# #
#
# def perfect_foresight(model, *args, **kwargs):
#
#     if model.is_dtcscc():
#         from dolo.algos.dtcscc.perfect_foresight import deterministic_solve
#
#     else:
#         raise Exception("Model type {} not supported.".format(model.model_type))
#
#     return deterministic_solve(model, *args, **kwargs)
