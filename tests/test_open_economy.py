from __future__ import division

import unittest

from dolo import *

class  TestInterpolation(unittest.TestCase):

    def test_smolyak(self):

        model = yaml_import('../examples/global_models/open_economy.yaml')
        print(model)
        dr  = approximate_controls(model,substitute_auxiliary=True,solve_systems=True)
        print dr
        
if __name__ == '__main__':
    unittest.main()
    tt = TestInterpolation()
    tt.test_2d_interpolation()

