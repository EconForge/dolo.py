import unittest

class TestGlobal(unittest.TestCase):
 
    def test_global_solution(self):

        from dolo import *
        
        from dolo.numeric.global_solve import global_solve
        
        filename = 'examples/global_models/rbc.yaml'
        
        model = yaml_import(filename)
        
        import time
        
        t1 = time.time()
        
        dr = global_solve(model, pert_order=1, maxit=5, smolyak_order=5, memory_hungry=True, verbose=True)
        
        t2 = time.time()
        
        dr = global_solve(model, pert_order=1, maxit=5, interp_type='mlinear', memory_hungry=True, verbose=True, polish=False)
            
        t3 = time.time()

