import unittest

class TestGlobal(unittest.TestCase):
 
    def test_global_solution(self):

        from dolo import yaml_import, global_solve 
        
        
        filename = 'examples/global_models/rbc.yaml'
        
        model = yaml_import(filename)
        
        import time
        
        t1 = time.time()
        
        dr = global_solve(model, pert_order=2, maxit=5, interp_type='mspline', verbose=True)


unittest.main()