import unittest


class TestGlobal(unittest.TestCase):

    def test_global_solution(self):
        from dolo import yaml_import, global_solve


        filename = 'examples/models/rbc.yaml'

        model = yaml_import(filename)

        import time

        dr = global_solve(model, pert_order=1, maxit=500, smolyak_order=3, verbose=True, polish=False, method='newton')

        t1 = time.time()


        dr = global_solve(model, pert_order=1, maxit=5, smolyak_order=5, verbose=True, polish=False, method='newton')

        t2 = time.time()

        dr = global_solve(model, pert_order=1, maxit=5, interp_type='multilinear', verbose=True, polish=False, method='newton')
        t3 = time.time()

        print(t2-t1)
        print(t3-t2)

    def test_global_solution(self):

        import time
        from dolo import yaml_import, global_solve


        filename = 'examples/models/rbc.yaml'

        model = yaml_import(filename)
        print(model.__class__)

        t3 = time.time()

        dr = global_solve(model, pert_order=1, maxit=5, interp_type='spline', verbose=True, interp_orders=[100,100])

        t4 = time.time()


        print(t4-t3)

if __name__ == '__main__':
    unittest.main()