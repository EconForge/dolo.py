def test_parameterized_expectations_direct():

    from dolo import yaml_import
    from dolo.algos.dtcscc.parameterized_expectations import parameterized_expectations
    from dolo.algos.dtcscc.time_iteration import time_iteration_direct

    model = yaml_import("examples/models/rbc_full.yaml")

    dr_ti = time_iteration_direct(model)
    dr_pea = parameterized_expectations(model, direct=True)

    x_ti = dr_ti(dr_ti.grid)
    x_pea = dr_pea(dr_ti.grid)

    print(abs(x_ti - x_pea).max() < 1e-5)

if __name__ == '__main__':
    test_parameterized_expectations_direct()


def test_parameterized_expectations():
    '''
    Test PEA against time iteration using a range of models:
    - rbc_full
    - rbc_taxes
    '''

    from dolo import yaml_import
    from dolo.algos.dtcscc.parameterized_expectations import parameterized_expectations
    from dolo.algos.dtcscc.time_iteration import time_iteration

    # Test rbc_full model
    model = yaml_import("examples/models/rbc_full.yaml")

    dr_time = time_iteration(model)
    dr_pea = parameterized_expectations(model, direct=False)

    x_time = dr_time(dr_time.grid)
    x_pea = dr_pea(dr_time.grid)
    assert(abs(x_time - x_pea).max() < 1e-5), "WARNING: rbc_full: Max deviation between PEA and time iteration solutions is 1e-5"

    # NEED TO FIX THE RBC_TAXES FILE, CURRENTLY DOESN'T WORK.
    # Test rbc_taxes model
    # model = yaml_import("examples/models/rbc_taxes.yaml")
    #
    # dr_time = time_iteration(model)
    # dr_pea = parameterized_expectations(model, direct=False)
    #
    # x_time = dr_time(dr_time.grid)
    # x_pea = dr_pea(dr_time.grid)
    # assert(abs(x_time - x_pea).max() < 1e-5), "rbc_taxes: Max deviation between PEA and time iteration solutions is 1e- 5"


if __name__ == '__main__':
    test_parameterized_expectations()
