def test_irf():

    from dolo import yaml_import, perturb, response
    import numpy as np

    model = yaml_import("examples/models/rbc.yaml")
    dr = perturb(model).dr

    irf = response(model, dr, "e_z")
    print(irf)


test_irf()
