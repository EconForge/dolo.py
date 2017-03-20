from dolo import *

filenames = [
    # "examples/models/rbc_dtcc_iid.yaml",
    "examples/models/rbc_dtcc_mc.yaml",
    # "examples/models/rbc_dtcc_ar1.yaml"
]
for fname in filenames:

    model = yaml_import(fname, check=False)
    print("Exogenous shocks:")
    print(model.exogenous)
    print("Discretized shock:")
    print(model.exogenous.discretize())
    try:
        print("Distribution;")
        print(model.get_distribution())
    except:
        pass

    dprocess = model.exogenous.discretize()

    print( dprocess.n_nodes() )
    print( dprocess.n_inodes(0) )
    print( dprocess.inode(0,0) )
    print( dprocess.node(0) )

    # from dolo.algos.time_iteration import time_iteration
    # sol = time_iteration(model)

    from dolo.algos.value_iteration import solve_policy
    sol = solve_policy(model, verbose=True)
