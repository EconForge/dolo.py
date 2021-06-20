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


def finite_time_iteration():
    from dolo import yaml_import, time_iteration
    from dolo.numeric.decision_rule import CustomDR
    import matplotlib.pyplot as plt

    model = yaml_import("examples/models/consumption_savings_iid.yaml")

    # in the near future this will become:
    # """c[t] = w[t]"""
    values = {"c": "w"}

    edr = CustomDR(values, model)

    sol = time_iteration(model, dr0=edr, maxit=5, trace=True)

    # # exemple of use
    # wmin, wmax = model.domain['w']
    # import numpy as np
    # wvec = np.linspace(wmin, wmax, 1000)[:,None]
    # plt.plot()
    # for k in range(len(sol.trace)):
    #     # sol.trace[k] is a dictionary with some recorded values
    #     dr = sol.trace[k]['dr']
    #     cvec = dr(wvec)[:,0]
    #     plt.plot(wvec, cvec, label=f"T-{k}")
    # plt.xlabel("w")
    # plt.legend(loc='upper left')
    # plt.ylabel("c(w)")
    # plt.show()


finite_time_iteration()
