def test_fga_higher_order_perturbations():

    from dolo import yaml_import
    from dolo.algos.dtcscc.perturbations_higher_order import approximate_controls

    model = yaml_import('examples/models/rbc.yaml')
    # for i in [1,2,3]:
    dr1 = approximate_controls(model, order=1)
    dr2 = approximate_controls(model, order=2)
    dr3 = approximate_controls(model, order=3)

    assert(dr1.order==1)
    assert(dr1.X_s.ndim==2)
    assert(dr3.X_ss.ndim==3)
