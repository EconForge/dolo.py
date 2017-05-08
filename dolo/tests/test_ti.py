# from dolo.algos.time_iteration import time_iteration
# from dolo import yaml_import
#
# model_ar1 = yaml_import('examples/models/rbc_dtcc_ar1.yaml')
# dr_ar1 = time_iteration(model_ar1)
#
# model_mc = yaml_import('examples/models/rbc_dtcc_mc.yaml')
# dr_mc = time_iteration(model_mc)
#
# model_iid = yaml_import('examples/models/rbc_dtcc_iid.yaml')
# dr_iid = time_iteration(model_iid)
#


from dolo.algos.value_iteration import solve_policy as value_iteration, evaluate_policy
from dolo import yaml_import
from dolo.algos.time_iteration import time_iteration


# model_mc = yaml_import('examples/models/rbc_dtcc_mc.yaml')
# dr_mc = value_iteration(model_mc, verbose=True)

# model_ar1 = yaml_import('examples/models/rbc_dtcc_ar1.yaml')
# dr_ar1 = value_iteration(model_ar1, verbose=True)
#
model_iid = yaml_import('examples/models/rbc_dtcc_mc.yaml')

endo_grid = model_iid.get_grid()
exo_grid = model_iid.exogenous.discretize()

mdr = time_iteration(model_iid)

drv = evaluate_policy(model_iid, mdr)

dr_iid = value_iteration(model_iid, verbose=True)
