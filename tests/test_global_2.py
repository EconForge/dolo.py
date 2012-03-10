from dolo import *

from dolo.numeric.global_solve import global_solve

filename = '../examples/global_models/rbc.yaml'

model = yaml_import(filename)

dr = global_solve(model, pert_order=1, smolyak_order=5, memory_hungry=True)