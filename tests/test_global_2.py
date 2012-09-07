from dolo import *

from dolo.numeric.global_solve import global_solve

filename = '../examples/global_models/rbc.yaml'

model = yaml_import(filename)

t1 = time.time()

dr = global_solve(model, pert_order=1, maxit=5, smolyak_order=5, memory_hungry=True, verbose=True)

t2 = time.time()

dr = global_solve(model, pert_order=1, maxit=5, interp_type='mlinear', memory_hungry=True, verbose=True, polish=False)

t3 = time.time()


#dr = global_solve(model, pert_order=1, smolyak_order=5, memory_hungry=True, integration='optimal_quantization')