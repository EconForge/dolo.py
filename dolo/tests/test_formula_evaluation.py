def test_eval_formula():

    from dolo.compiler.eval_formula import eval_formula
    from dolo import yaml_import, approximate_controls, simulate

    model = yaml_import('examples/models/rbc.yaml')
    dr = approximate_controls(model)
    sim = simulate(model, dr)

    rr = eval_formula("delta*k-i", sim, context=model.calibration)
    rr = eval_formula("y(1) - y", sim, context=model.calibration)

    sim['diff'] = model.eval_formula("delta*k-i", sim)
    model.eval_formula("y(1) - y", sim)
    sim['ddiff'] = model.eval_formula("diff(1)-diff(-1)", sim)
