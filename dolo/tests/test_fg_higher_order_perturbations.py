def test_fga_higher_order_perturbations():

    from dolo import yaml_import
    from dolo.algos.fg.perturbations_higher_order import approximate_controls

    model = yaml_import('examples/models/rbc.yaml')
    for i in [1,2,3]:
        dr = approximate_controls(model, order=2)
