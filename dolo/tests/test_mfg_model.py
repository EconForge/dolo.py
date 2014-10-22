def test_mfg_model():

    from dolo import yaml_import

    # from dolo.algos.commands import time_iteration, simulate, evaluate_policy
    from dolo.algos.mfg.time_iteration import time_iteration
    from dolo.algos.mfg.value_iteration import evaluate_policy
    from dolo.algos.mfg.simulations import simulate


    model = yaml_import("examples/models/sudden_stop.yaml")

    mdr = time_iteration(model)

    #
    sim = simulate(model, mdr, 0, horizon=50, n_exp=0) # irf
    assert(sim.shape==(50,6))
    sim = simulate(model, mdr, 0, horizon=50, n_exp=1) # one stochastic simulation
    assert(sim.shape==(50,6))
    sim = simulate(model, mdr, 0, horizon=50, n_exp=10) # many stochastic simulations
    assert(sim.shape==(10,50,6))


    mv = evaluate_policy(model, mdr, verbose=True, maxit=500)

    mv = evaluate_policy(model, mdr, verbose=True, maxit=500, initial_guess=mv)

    sim_v = simulate(model, mdr, 0, drv=mv, horizon=50, n_exp=10)




if __name__ == '__main__':

    test_mfg_model()
