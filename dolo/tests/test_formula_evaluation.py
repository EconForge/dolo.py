def test_eval_formula():

    from dolo.compiler.eval_formula import eval_formula
    from dolo import yaml_import, time_iteration, simulate
    import dolo.config

    from dolo import perturb

    model = yaml_import("examples/models/rbc.yaml")

    dr = perturb(model)
    sim = simulate(model, dr)
    sim = sim.sel(N=0)
    sim = sim.to_pandas()

    print(sim.columns)
    rr = eval_formula("delta*k(0)-i(0)", sim, context=model.calibration)
    rr = eval_formula("y(1) - y(0)", sim, context=model.calibration)

    sim["diff"] = model.eval_formula("delta*k(0)-i(0)", sim)
    model.eval_formula("y(1) - y(0)", sim)
    sim["ddiff"] = model.eval_formula("diff(1)-diff(-1)", sim)
