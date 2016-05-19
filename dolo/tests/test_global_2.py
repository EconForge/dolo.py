def test_time_iteration_smolyak():
    from dolo import yaml_import, time_iteration


    filename = 'examples/models/rbc.yaml'

    model = yaml_import(filename)

    import time

    dr = time_iteration(model, pert_order=1, maxit=500, verbose=True)
    # dr = time_iteration(model, pert_order=1, maxit=5, smolyak_order=5, verbose=True)


def test_time_iteration_spline():

    import time
    from dolo import yaml_import, time_iteration


    filename = 'examples/models/rbc.yaml'

    model = yaml_import(filename)
    print(model.__class__)


    dr = time_iteration(model, pert_order=1, maxit=5, verbose=True)
    # dr = time_iteration(model, pert_order=1, maxit=5, verbose=True)


if __name__ == '__main__':
    test_time_iteration_spline()
    test_time_iteration_smolyak()
