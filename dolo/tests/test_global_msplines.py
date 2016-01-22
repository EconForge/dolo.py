import unittest


class TestGlobal(unittest.TestCase):

    def test_global_solution(self):

        from dolo import yaml_import, time_iteration


        filename = 'examples/models/rbc.yaml'

        model = yaml_import(filename)

        import time

        t1 = time.time()

        dr = time_iteration(model, pert_order=1, maxit=5, interp_type='spline', verbose=True)
