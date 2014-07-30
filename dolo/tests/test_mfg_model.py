def test_mfg_model():

    from dolo import yaml_import
    from dolo.algos.mfg.time_iteration import solve_mfg_model
    from dolo.algos.mfg.simulations import simulate
    from dolo.algos.mfg.value_iteration import evaluate_policy


    model = yaml_import("examples/models/sudden_stop.yaml")

    mdr = solve_mfg_model(model)

    sim = simulate(model, mdr, 0)

    mv = evaluate_policy(model, mdr, verbose=True, maxit=500)

if __name__ == '__main__':

    test_mfg_model()
