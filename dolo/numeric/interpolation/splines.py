try:
    from dolo.numeric.interpolation.splines_cython import MultivariateSplines as MultivariateSplinesCython

except:
    raise Exception('Impossible to import spline library. You need to compile it with cython first')

import numpy

class MultivariateSplines(MultivariateSplinesCython):

    def __init__(self, smin,smax,orders):

        MultivariateSplinesCython.__init__(self,smin,smax,orders)

        from dolo.numeric.misc import cartesian

        grid = cartesian( [numpy.linspace(self.smin[i], self.smax[i], self.orders[i]) for i in range(self.d)] ).T
        self.grid = numpy.ascontiguousarray(grid)
        print(grid.shape)
