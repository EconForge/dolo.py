from dolo.algos.time_iteration import time_iteration
from dolo import yaml_import

model_ar1 = yaml_import('examples/models/rbc_dtcc_ar1.yaml')
dr_ar1 = time_iteration(model_ar1)

model_mc = yaml_import('examples/models/rbc_dtcc_mc.yaml')
dr_mc = time_iteration(model_mc)

model_iid = yaml_import('examples/models/rbc_dtcc_iid.yaml')
dr_iid = time_iteration(model_iid)
