from __future__ import division

import unittest


class  TestInterpolation(unittest.TestCase):

#    def test_2d_interpolation(self):
#        import numpy
#        from dolo.numeric.interpolation import RectangularDomain, SplineInterpolation, LinearTriangulation, TriangulatedDomain
#        from dolo.numeric.smolyak import SmolyakGrid
#        smin = [-2,-2]
#        smax = [2,2]
#        orders = [5,5]
#        orders_ref = [100,100]
#        l= 6
#        bounds = numpy.row_stack([smin,smax])
#        recdomain = RectangularDomain( smin, smax, orders )
#        tridomain = TriangulatedDomain( recdomain.grid )
#        recdomain_ref = RectangularDomain( smin, smax, orders_ref )
#        smolyakgrid = SmolyakGrid( smin, smax, l)
#
#
#        fun = lambda x: 1.0/numpy.sqrt(2*numpy.pi)*numpy.exp( -( numpy.square(x[0:1,:]) + numpy.square(x[1:2,:]) ) / 2.0 )
#
#        values_rec = fun(recdomain.grid)
#        values_smol = fun(smolyakgrid.grid)
#
#        interp_rec = SplineInterpolation(smin,smax,orders)
#        interp_smol = smolyakgrid
#        interp_simplex = LinearTriangulation(tridomain)
#
#        interp_rec.set_values(values_rec)
#        interp_smol.set_values(values_smol)
#        interp_simplex.set_values(values_rec)
#
#        true_values = fun(recdomain_ref.grid).reshape(orders_ref)
#        interpolated_values_spline = interp_rec(recdomain_ref.grid).reshape(orders_ref)
#        interpolated_values_simplex = interp_simplex(recdomain_ref.grid).reshape(orders_ref)
#        interpolated_values_smolyak = interp_smol(recdomain_ref.grid).reshape(orders_ref)
#
##
##        from mayavi import mlab
##        mlab.figure(bgcolor=(1.0,1.0,1.0))
##        #mlab.surf(abs(interpolated_values_smolyak-true_values),warp_scale="auto")
###        mlab.surf(abs(interpolated_values_spline-true_values),warp_scale="auto")
##        mlab.surf(abs(interpolated_values_simplex-true_values),warp_scale="auto")
##        mlab.colorbar()

    def test_interpolation_time(self):
        d = 3
        l = 5
        n_x = 1
        N = 1000
        from numpy import column_stack, minimum, maximum

        smin = [-1]*d
        smax = [1]*d
        from dolo.numeric.interpolation.smolyak import SmolyakGrid
        sg = SmolyakGrid(smin,smax,l)
        print(sg.grid.shape)
        from numpy import exp
        values = column_stack( [sg.grid[:,0] + exp(sg.grid[:,0])]*n_x )

        print(values.shape)
        sg.set_values(values)

        from numpy import random
        points = random.rand( d*N )
        points = minimum(points,1)
        points = maximum(points,-1)
        points = points.reshape( (N,d) )

        # I need to add the corners of the grid !

        import time
        t = time.time()
        for i in range(10):
            test1 = sg(points)
        s = time.time()
        print('Smolyak : {}'.format(s-t))


        #
        # from dolo.numeric.interpolation.interpolation import SparseLinear
        # sp = SparseLinear(smin,smax,l)
        #
        # xvalues = sg(sp.grid)
        #
        # sp.set_values(xvalues)
        # t = time.time()
        # for i in range(10):
        #     test2 = sp(points)
        # s = time.time()
        # print('Sparse linear : {}'.format(s-t))
        #
        # import numpy
        # if False in numpy.isfinite(test2):
        #     print('Problem')

#
#        a = time.time()
#        from scipy.interpolate import LinearNDInterpolator
#        ggrid = array(sg.grid.T)
#        gvalues = array(values.T)
#        gpoints = array(points.T)
#        print ggrid.shape
#        print(gvalues.shape)
#        lininterp = LinearNDInterpolator(ggrid, gvalues)
#        b = time.time()
#        print('Triangulation : {}'.format(b-a))
#        t = time.time()
#        test2 = lininterp(gpoints)
#        s = time.time()
#        print('Linear : {}'.format(s-t))





if __name__ == '__main__':
    unittest.main()
    tt = TestInterpolation()
    #tt.test_2d_interpolation()
    tt.test_interpolation_time()

