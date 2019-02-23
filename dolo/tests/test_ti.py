def test_vfi():

    from dolo.algos.value_iteration import value_iteration as value_iteration, evaluate_policy
    from dolo import yaml_import
    from dolo.algos.time_iteration import time_iteration

    model_iid = yaml_import('examples/models/rbc_iid.yaml')

    mdr = time_iteration(model_iid, with_complementarities=False)
    drv = evaluate_policy(model_iid, mdr)

    dr_iid = value_iteration(model_iid, verbose=True, maxit=5)
