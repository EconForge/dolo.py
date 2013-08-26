import unittest


class TestGlobal(unittest.TestCase):

    def test_global_solution(self):

        from dolo import yaml_import, global_solve


        filename = 'examples/global_models/rbc.yaml'

        model = yaml_import(filename)

        import time

        t1 = time.time()


        print('ok')
        dr = global_solve(model, pert_order=1, maxit=5, interp_type='spline', verbose=True, method='newton')

