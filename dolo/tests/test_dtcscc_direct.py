def test_direct():

    from dolo import yaml_import
    from dolo.algos.dtcscc.perturbations import approximate_controls
    from dolo.algos.dtcscc.time_iteration import time_iteration_direct, time_iteration

    model = yaml_import("examples/models/rbc_full.yaml")

    dr = time_iteration_direct(model)
    ddr = time_iteration(model)

    x0 = dr(dr.grid)
    x1 = ddr(dr.grid)

    print(abs(x1 - x0).max()<1e-5)


if __name__ =='__main__':

    test_pea()
