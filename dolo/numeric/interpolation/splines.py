try:
    pass


except:
    raise Exception('Impossible to import spline library. You need to compile it with cython first')

from splines_cython import USpline, eval_UBspline

import numpy
import numpy as np

def mlinspace(smin,smax,orders):
    if len(orders) == 1:
        res = np.atleast_2d( np.linspace(np.array(smin),np.array(smax),np.array(orders)) )
        return res.copy() ## workaround for strange bug
    else:
        meshes = np.meshgrid( *[numpy.linspace(smin[i],smax[i],orders[i]) for i in range(len(orders))], indexing='ij' )
        table = np.row_stack( [l.flatten() for l in meshes])
        return np.ascontiguousarray(table)
#class MultivariateSplines(MultivariateSplinesCython):
#
#    def __init__(self, smin,smax,orders):

#        MultivariateSplinesCython.__init__(self,smin,smax,orders)

#        from dolo.numeric.misc import cartesian

#        grid = cartesian( [numpy.linspace(self.smin[i], self.smax[i], self.orders[i]) for i in range(self.d)] ).T
#        self.grid = numpy.ascontiguousarray(grid)
#        print(grid.shape)

 
class MultivariateSplines:

    __grid__ = None

    def __init__(self, smin, smax, orders):

        self.d = len(smin)
        assert(len(smax) == self.d)
        assert(len(orders) == self.d)
        self.smin = smin
        self.smax = smax
        self.orders = orders
        self.__splines__ = None

    def set_values(self, values):

        if not np.all( np.isfinite(values)):
            raise Exception('Trying to interpolate non-finite values')

        values = np.ascontiguousarray(values) # we don't need that since USpline already checks for contiguity

        n_v = values.shape[0]
        self.__splines__ = []
        for i in range(n_v):
            self.__splines__.append(  USpline(self.smin,self.smax,self.orders,values[i,:].reshape(self.orders)   ) )


    def interpolate(self, points, with_derivatives=False):

        import time

        points = np.ascontiguousarray(points)

        if not np.all( np.isfinite(points)):
            raise Exception('Spline interpolator evaluated at non-finite points.')

        n_v = len(self.__splines__)
        N = points.shape[1]
        n_s = points.shape[0]
        if not with_derivatives:
            values = np.empty((n_v,N))
            for i in range(n_v):
                sp =  self.__splines__[i]
                values[i,:] = eval_UBspline(self.smin, self.smax, self.orders, sp.coefs, points, diff=False )
            return values
        else:
            values = np.empty((n_v,N))
            dvalues = np.empty((n_v,n_s,N))
            for i in range(n_v):
                sp =  self.__splines__[i]
                [value, d_value] =  eval_UBspline(self.smin, self.smax, self.orders, sp.coefs, points, diff=True )
                values[i,:] = value
                dvalues[i,:,:] = d_value

            return [values,dvalues]

    @property
    def grid(self):
        if self.__grid__ is None:
            self.__grid__ = mlinspace(self.smin, self.smax, self.orders)
        return self.__grid__

    def __call__(self, s):
        return self.interpolate(s)

       
