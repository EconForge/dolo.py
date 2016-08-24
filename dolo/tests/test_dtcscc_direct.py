def test_direct():

    from dolo import yaml_import
    from dolo.algos.dtcscc.perturbations import approximate_controls
    from dolo.algos.dtcscc.time_iteration import time_iteration_direct, time_iteration

    model = yaml_import("examples/models/rbc_full.yaml")

    # Check without complementarity conditions
    dr = time_iteration_direct(model, with_complementarities=False)
    ddr = time_iteration(model, with_complementarities=False)

    x0 = dr(dr.grid)
    x1 = ddr(dr.grid)

    print(abs(x1 - x0).max()<1e-5)


    # Check with complementarity conditions
    dr = time_iteration_direct(model, with_complementarities=True)
    ddr = time_iteration(model, with_complementarities=True)

    x0 = dr(dr.grid)
    x1 = ddr(dr.grid)

    print(abs(x1 - x0).max()<1e-5)


if __name__ =='__main__':

    test_direct()
