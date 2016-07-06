def test_time_iteration():
    '''
    Test That time iteration works for a range of models
    - rbc_full
    - rbc_taxes
    - NK_dsge ...
    '''

    from dolo import yaml_import
    from dolo.algos.dtcscc.time_iteration import time_iteration

    # Test rbc_full model
    model = yaml_import("examples/models/rbc_full.yaml")

    dr_time = time_iteration(model)
    assert(dr_time is not None)

    # Test NK_dsge model
    model = yaml_import("examples/models/NK_dsge.yaml")

    dr_time = time_iteration(model)
    assert(dr_time is not None)

if __name__ == '__main__':
    test_time_iteration()
