import unittest

from numpy.testing import assert_almost_equal


class  MultilinearInterpolationTestCase(unittest.TestCase):

    def test_interpolation(self):

        for d in range(1,5):
            self.interpolation(d)

    def interpolation(self, d):

        from itertools import product
        import numpy
        from numpy import array, column_stack
        from dolo.numeric.interpolation.multilinear import multilinear_interpolation


        smin = array([0.0]*d)
        smax = array([1.0]*d)
        orders = numpy.array( [4,3,5,6,3], dtype=numpy.int )
        orders = orders[:d]


        grid = column_stack( [e for e in product(*[numpy.linspace(smin[i],smax[i],orders[i]) for i in range(d)])])

        finer_grid = column_stack( [e for e in product(*[numpy.linspace(0,1,10)]*d) ] )

        if d == 1:
            f = lambda g: numpy.row_stack([
                2*g[0,:]
            ])
        elif d == 2:
            f = lambda g: numpy.row_stack([
                g[0,:] * g[1,:],
            ])
        elif d == 3:
            f = lambda g: numpy.row_stack([
                (g[0,:] - g[1,:]) * g[2,:],
            ])
        elif d== 4:
            f = lambda g: numpy.row_stack([
                (g[3,:] - g[1,:]) * (g[2,:] - g[0,:])
            ])
#

        values = f( grid )

        finer_grid = numpy.ascontiguousarray(finer_grid)

        interpolated_values = multilinear_interpolation(smin, smax, orders, values, finer_grid)


        from dolo.numeric.interpolation.smolyak import SmolyakGrid

        sg = SmolyakGrid( smin, smax ,3)
        sg.set_values( f(sg.grid) )

        smol_values = sg(finer_grid)

        true_values = f(finer_grid)

        err_0 = abs(true_values - smol_values).max()
        err_1 = abs(true_values - interpolated_values).max()

        # both errors should be 0, because interpolated function is a 2d order polynomial
        assert_almost_equal(err_1,0)

        from dolo.numeric.interpolation.multilinear import MultilinearInterpolator
        mul_interp = MultilinearInterpolator(smin,smax,orders)
        mul_interp.set_values( f( mul_interp.grid) )
        interpolated_values_2 = mul_interp( finer_grid )

        err_3 = (abs(interpolated_values- interpolated_values_2))
        assert_almost_equal(err_3,0)



if __name__ == '__main__':
    unittest.main()
