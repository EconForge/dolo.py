def test_custom_dr():

    from dolo import yaml_import, simulate, time_iteration
    import numpy as np
    from dolo.numeric.decision_rule import CustomDR

    model = yaml_import("examples/models/rbc.yaml")

    values = {"n": "0.33 + z*0.01", "i": "delta*k-0.07*(k-9.35497829)"}

    edr = CustomDR(values, model)

    m0, s0 = model.calibration["exogenous", "controls"]

    edr(m0, s0)

    sim = simulate(model, edr, s0=np.array([0.0, 8.0]))

    time_iteration(model, dr0=edr)
