def test_ergo_iid():

    from dolo import yaml_import, improved_time_iteration, ergodic_distribution

    model = yaml_import("examples/models/rbc_iid.yaml")
    sol = improved_time_iteration(model)
    Π, μ = ergodic_distribution(model, sol.dr)
    assert μ.ndim == 2


def test_ergo_mc():

    from dolo import yaml_import, improved_time_iteration, ergodic_distribution

    model = yaml_import("examples/models/rbc_mc.yaml")
    sol = improved_time_iteration(model)
    Π, μ = ergodic_distribution(model, sol.dr)
    assert μ.ndim == 2
