from dolo import *

from dolo.algos.value_iteration import constant_policy


filenames = [
    "examples/models/rbc_dtcc_iid.yaml",
    "examples/models/rbc_dtcc_mc.yaml",
    # "examples/models/rbc_dtcc_ar1.yaml"
]
for fname in filenames:

    model = yaml_import(fname, check=False)
    print(model)
    # print("Exogenous shocks:")
    # print(model.exogenous)
    # print("Discretized shock:")
    # print(model.exogenous.discretize())
    # try:
    #     print("Distribution;")
    #     print(model.get_distribution())
    # except:
    #     pass
    #
    # dprocess = model.exogenous.discretize()
    #
    # # print( dprocess.n_nodes() )
    # # print( dprocess.n_inodes(0) )
    # # print( dprocess.inode(0,0) )
    # # print( dprocess.node(0) )
    #
    # # from dolo.algos.time_iteration import time_iteration
    # # sol = time_iteration(model)
    #
    # from dolo.algos.value_iteration import solve_policy, evaluate_policy
    # from dolo.algos.time_iteration import time_iteration
    # from dolo.algos.simulations import simulate
    #
    # dri = constant_policy(model)
    # # val = evaluate_policy(model, dri)
    #
    # # dr_valit = solve_policy(model, verbose=True, maxit=2000)
    #
    # dr = time_iteration(model)
    #
    # sim = simulate(model, dr)
    #
    # print(sim)
    # exit(0)
    # ev = evaluate_policy(model, dr, verbose=False)
