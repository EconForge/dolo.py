from __future__ import division

import unittest
from numpy.testing import assert_allclose


class  TestInterpolation(unittest.TestCase):

    # def test_chebychev(self):
    #
    #     import numpy as np
    #     from dolo.numeric.interpolation.smolyak import chebychev, chebychev2
    #
    #     points = np.linspace(-1,1,100)
    #
    #     cheb = chebychev(points,5)
    #     cheb2 = chebychev2(points,5)
    #
    #     def T4(x):
    #         return ( 8*np.power(x,4) - 8*np.power(x,2) + 1 )
    #     def U4(x):
    #         return 4*( 16*np.power(x,4) - 12*np.power(x,2) + 1 )
    #
    #     true_values_T = np.array([T4(i) for i in points])
    #     true_values_U = np.array([U4(i) for i in points])
    #
    #     assert_allclose(true_values_T, cheb[4,:])
    #     assert_allclose(true_values_U, cheb2[4,:]*4)

    def test_smolyak(self):

        import numpy

        f = lambda x: numpy.column_stack([
            x[:,0] * x[:,1]**0.5,
            x[:,1] * x[:,1] - x[:,0] * x[:,0]
        ])


        a = [0.5,0.1]
        b = [2,3]
        bounds = numpy.row_stack([a,b])

        from dolo.numeric.interpolation.smolyak import SmolyakGrid

        sg = SmolyakGrid(a,b,3)

        values = f(sg.grid)
        sg.set_values(values)

        assert( abs( sg(sg.grid) - values ).max()<1e-8 )
    #
    # def test_smolyak_plot_2d(selfs):
    #
    #     import numpy
    #     from dolo.numeric.interpolation.smolyak import SmolyakGrid
    #
    #     bounds = numpy.column_stack([[-1,1]]*2)
    #     sg = SmolyakGrid(bounds[0,:],bounds[1,:],3)
    #     sg.plot_grid()
    #
    # def test_smolyak_plot_3d(selfs):
    #
    #     import numpy
    #     from dolo.numeric.interpolation.smolyak import SmolyakGrid
    #
    #     bounds = numpy.column_stack([[-1,1]]*3)
    #     sg = SmolyakGrid(bounds[0,:],bounds[1,:],3)
    #     sg.plot_grid()


    def test_smolyak_2(self):

        import numpy
        from dolo.numeric.interpolation.smolyak import SmolyakGrid
        d = 5
        l = 4

        bounds = numpy.row_stack([[-0.5]*d, [0.7]*d])
        sg = SmolyakGrid(bounds[0,:],bounds[1,:],l)
        f = lambda x: numpy.row_stack([
                    x[:,0] * x[:,1],
                    x[:,1] * x[:,1] - x[:,0] * x[:,0]
                ])
        values = f(sg.grid)

        import time
        t = time.time()
        for i in range(5):
            sg.set_values(sg.grid)

            val = sg(sg.grid)
        s = time.time()
        print(s-t)


if __name__ == '__main__':

    unittest.main()
