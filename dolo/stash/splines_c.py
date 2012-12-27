import numpy as np
import ctypes
global f

lib_spline  = ctypes.cdll.LoadLibrary('/home/pablo/Programmation/bigeco/dolo/dolo/numeric/interpolation/spline.so')


class MultivariateSpline:

    def __init__(self, smin, smax, orders):

        self.smin = smin
        self.smax = smax
        self.orders = orders
        self.d = len(self.orders)
        from itertools import product
        self.grid = np.ascontiguousarray(  np.row_stack( product( *[np.linspace(smin[i],smax[i], orders[i]) for i in range(self.d)] )).T )

    def set_values(self, values):
        if False in np.isfinite(values.flatten()):
            raise(Exception("Trying to set non finite values"))
        self.values = values

    def interpolate(self, points, with_derivatives=False):
        [value, d_value] = spline_c(self.smin, self.smax, self.orders, self.values, points, derivatives=True)
#        outside = (points < self.smin[:,None]) + (points > self.smax[:,None])
#        outside = outside.sum(axis=0) > 0
        projection = np.minimum( points, self.smax[:,None]-0.0000000001)
        projection = np.maximum( projection, self.smin[:,None]+0.0000000001)
        delta = (points - projection)

        # TODO : correct only outside observations
        for i in range(value.shape[0]):
            value[i,:] += (delta*d_value[i,:,:]).sum(axis=0)

        if with_derivatives:
            return [value,d_value]
        else:
            return value


    def __call__(self, points):
        return self.interpolate(points)


def spline_c(smin, smax, orders, values, points, derivatives=False):

    points = np.minimum( points, smax[:,None]-0.0000000001)
    points = np.maximum( points, smin[:,None]+0.0000000001)

    if False in np.isfinite(points):
        print(values)
        raise(Exception("non finite points"))

    if False in np.isfinite(values):
        print(values)
        raise(Exception("non finite value"))

    smin = np.ascontiguousarray(smin,dtype=np.double)
    smax = np.ascontiguousarray(smax,dtype=np.double)
    orders = np.ascontiguousarray(orders, dtype=np.int32)
    values = np.ascontiguousarray(values, dtype=np.double)
    points = np.ascontiguousarray(points, dtype=np.double)

    resp = np.zeros( (values.shape[0], points.shape[1]), dtype=np.double)

    d = len(smin)

    n = points.shape[1]


    smin_ptr = smin.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    smax_ptr = smax.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    orders_ptr = orders.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    if d == 1:
        points_x_ptr = points[0,:].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    if d == 2:
        points_x_ptr = points[0,:].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        points_y_ptr = points[1,:].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    if d == 3:
        points_x_ptr = points[0,:].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        points_y_ptr = points[1,:].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        points_z_ptr = points[2,:].ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    if derivatives:
        fun = lib_spline['evaluate_spline_{}d_g'.format(d)]
        resp_d = np.zeros( (values.shape[0], points.shape[1], d) )
    else:
        fun = lib_spline['evaluate_spline_{}d'.format(d)]

    for i in range(values.shape[0]):

        values_ptr = values[i,:].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        output_ptr = resp[i,:].ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        if d == 1:
            args = [smin_ptr, smax_ptr, orders_ptr, values_ptr, n, points_x_ptr, output_ptr]
        elif d == 2:
            args = [smin_ptr, smax_ptr, orders_ptr, values_ptr, n, points_x_ptr, points_y_ptr, output_ptr]
        elif d == 3:
            args = [smin_ptr, smax_ptr, orders_ptr, values_ptr, n, points_x_ptr, points_y_ptr, points_z_ptr, output_ptr]

        if derivatives:
            grad_ptr = resp_d[i,:,:].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            args.append(grad_ptr)

        fun(*args)

    if derivatives:
        resp_d = np.ascontiguousarray(resp_d.swapaxes(1,2))
        return [resp, resp_d]
    else:
        return resp


if __name__ == '__main__':

    N_grid = 50
    N_fine_grid = 100
    d = 3

    smin = np.array( [0.0]*d )
    smax = np.array( [1.0]*d )
    orders = [N_grid]*d
    orders = orders[:d]

    print('Grid dimensions : {}'.format(orders))

    from itertools import product

    grid = np.row_stack( product( *[np.linspace(smin[i],smax[i], orders[i]) for i in range(d)] )).T

    print('We try to interpolate on a grid with some points outside of [0,1]')
    points = np.row_stack( product( *[np.linspace(smin[i]-0.1,smax[i]+0.1, N_fine_grid) for i in range(d)] )).T

    print('Number of points to interpolate : {}'.format(points.shape[1]))

    if d == 1:
        f = lambda x: np.row_stack( [np.cos(x)] )
        #        f = lambda x: np.row_stack( [np.power(0.1+x,1.0/4.0)] )
        #f = lambda x: np.row_stack( [x*x] )
        v = f(grid[0,:])
    elif d == 2:
#        f = lambda x,y: np.row_stack( [np.sin(x)*y] )
        f = lambda x,y: np.row_stack( [np.sin(x)*y, np.sin(y)*x] )
        v = f(grid[0,:], grid[1,:])
    elif d == 3:
        f = lambda x,y,z: np.row_stack( [np.sin(x)*y/np.sqrt(z)] )
        f = lambda x,y,z: np.row_stack( [2*x + 3*y + 4*z] )
        v = f(grid[0,:], grid[1,:], grid[2,:])


    n_v = v.shape[0]

    import numpy.random

    #points = np.row_stack( product( *[np.linspace(smin[i],smax[i], N_fine_grid) for i in range(d)] )).T


    true_vals = f( *[p for i,p in enumerate(points) if i <= d] )


    mvs = MultivariateSpline(smin, smax, orders)
    mvs.set_values( f( *[s for i,s in enumerate(mvs.grid) if i <= d] ) )

    print('')

    import time
    t = time.time()
    for i in range(10):
        out = spline_c(smin,smax,orders,mvs.values, points)
    s = time.time()
    print('Splines (direct call): {} s per call'.format((s-t)/10))
    print( 'Max error : {}'.format(abs(out-true_vals).max()) )
    print( 'Mean error : {}'.format(abs(out-true_vals).mean()) )


    print('')
    import time
    t = time.time()
    for i in range(10):
        [out] = mvs(points)
    s = time.time()
    print('Splines (with linear extrapolation) : {} s per call'.format((s-t)/10/n_v))
    print( 'Max error : {}'.format(abs(out-true_vals).max()) )
    print( 'Mean error : {}'.format(abs(out-true_vals).mean()) )




    if d == 2:

        print('')

        from scipy.interpolate import SmoothBivariateSpline
        grid_x = grid[0,:]
        grid_y = grid[1,:]

        values = mvs.values[0,:]

        bs = SmoothBivariateSpline(grid_x, grid_y, values)

        t = time.time()
        for i in range(10):
            out = bs.ev(points[0,:], points[1,:])
        s = time.time()
        print('Splines (smooth splines from scipy) : {} s per call'.format((s-t)/10))
        print( 'Max error : {}'.format(abs(out-true_vals).max()) )
        print( 'Mean error : {}'.format(abs(out-true_vals).mean()) )




