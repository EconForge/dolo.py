import unittest

from numpy.testing import assert_almost_equal


class  MultilinearInterpolationTestCase(unittest.TestCase):

    def test_interpolation(self):

        from itertools import product
        import numpy
        from numpy import array, column_stack, row_stack
        from dolo.numeric.multilinear import multilinear_interpolation

        d = 4

        smin = array([0]*d)
        smax = array([1]*d)
        orders = [4,3,5,6,3]
        orders = orders[:d]


        grid = column_stack( [e for e in product(*[numpy.linspace(smin[i],smax[i],orders[i]) for i in range(d)])])

        finer_grid = column_stack( [e for e in product(*[numpy.linspace(0,1,10)]*d) ] )

        f = lambda g: numpy.row_stack([
            g[0,:] * g[1,:],
            (g[3,:] - g[1,:]) * g[2,:],
            ])

        values = f( grid )

        interpolated_values = multilinear_interpolation(smin, smax, orders, values, finer_grid)


        from dolo.numeric.smolyak import SmolyakGrid

        sg = SmolyakGrid( smin, smax ,3)
        sg.set_values( f(sg.grid) )

        smol_values = sg(finer_grid)

        true_values = f(finer_grid)

        err_0 = abs(true_values - smol_values).max()
        err_1 = abs(true_values - interpolated_values).max()

        # both errors should be 0, because interpolated function is a 2d order polynomial
        assert_almost_equal(err_1,0)

        from dolo.numeric.multilinear import MultilinearInterpolator
        mul_interp = MultilinearInterpolator(smin,smax,orders)
        mul_interp.set_values( f( mul_interp.grid) )
        interpolated_values_2 = mul_interp( finer_grid )

        err_3 = (abs(interpolated_values- interpolated_values_2))
        assert_almost_equal(err_3,0)



if __name__ == '__main__':
    unittest.main()
