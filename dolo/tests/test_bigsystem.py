def test_big_system():
    from dolo import yaml_import
    from dolo.algos.dtcscc.nonlinearsystem import nonlinear_system

    from dolo.algos.dtcscc.time_iteration import time_iteration

    model = yaml_import('examples/models/rbc.yaml')

    dr = time_iteration(model, grid={'type': 'smolyak', 'mu': 3}, verbose=True)
    sol = nonlinear_system(model, grid={'type': 'smolyak', 'mu': 3})

    diff = (sol.__values__) - dr.__values__
    assert(abs(diff).max() < 1e-6)

    sol_high_precision = nonlinear_system(model, grid={'type': 'smolyak', 'mu': 5}, initial_dr=sol)

    assert(sol_high_precision.grid.shape == (145, 2))

if __name__ == '__main__':
    test_big_system()
