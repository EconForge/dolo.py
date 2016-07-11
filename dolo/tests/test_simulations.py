from dolo import *

def test_simulations_dtcscc():
    model = yaml_import("examples/models/rbc.yaml")
    dr = approximate_controls(model)
    sim = plot_decision_rule(model,dr,'k')

def test_simulations_dtcscc_getcorrectbounds():
    model = yaml_import("examples/models/rbc.yaml")
    kss = model.get_calibration('k')

    dr = approximate_controls(model)
    sim = plot_decision_rule(model,dr,'k',bounds=[1.0,10.0])
    assert(sim['k'].iloc[0] == 1.0)
    assert(sim['k'].iloc[-1] == 10.0)

def test_simulations_dtcscc_getcorrectbounds2():
    model = yaml_import("examples/models/rbc.yaml")
    dr = approximate_controls(model)
    sim = plot_decision_rule(model,dr,'k')
    assert(not(sim['k'].iloc[0] == 1.0))
    assert(not(sim['k'].iloc[-1] == 10.0))

def test_simulations_dtmscc():
    model = yaml_import("examples/models/rbc_dtmscc.yaml")
    dr = time_iteration(model)
    sim = plot_decision_rule(model,dr,'k')

def test_simulations_dtmscc_getcorrectbounds():
    model = yaml_import("examples/models/rbc_dtmscc.yaml")
    dr = time_iteration(model)
    sim = plot_decision_rule(model,dr,'k',bounds=[1.0,10.0])
    assert(sim['k'].iloc[0] == 1.0)
    assert(sim['k'].iloc[-1] == 10.0)
