def test_vfi():

    from dolo.algos.value_iteration import solve_policy as value_iteration, evaluate_policy
    from dolo import yaml_import
    from dolo.algos.time_iteration import time_iteration

    model_iid = yaml_import('examples/models/rbc_dtcc_iid_ar1.yaml')

    endo_grid = model_iid.get_grid()
    exo_grid = model_iid.exogenous.discretize()

    mdr = time_iteration(model_iid)

    drv = evaluate_policy(model_iid, mdr)

    dr_iid = value_iteration(model_iid, verbose=True, maxit=5)
