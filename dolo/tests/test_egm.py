# def test_egm_mc():
#
#    from dolo import yaml_import
#    from dolo.algos.egm import egm
#    import numpy as np
#
#    model = yaml_import("examples/models/consumption_savings.yaml")
#    sol = egm(model, a_grid=np.linspace(0.1, 10, 10) ** 2, verbose=True)


def test_egm_iid():

    from dolo import yaml_import
    from dolo.algos.egm import egm
    import numpy as np

    model = yaml_import("examples/models/consumption_savings_iid.yaml")
    sol = egm(model, a_grid=np.linspace(0.1, 10, 10) ** 2, verbose=True)
