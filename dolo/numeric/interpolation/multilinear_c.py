import numpy as np
import ctypes
f  = ctypes.cdll.LoadLibrary('/home/pablo/Programmation/bigeco//dolo/dolo/numeric/interpolation/multilinear_lib.so')

fun_1d = f['multilinear_interpolation_1d']
fun_2d = f['multilinear_interpolation_2d']
fun_3d = f['multilinear_interpolation_3d']
fun_4d = f['multilinear_interpolation_4d']

def multilinear_c(smin, smax, orders, values, y):

    smin = np.ascontiguousarray(smin,dtype=np.double)
    smax = np.ascontiguousarray(smax,dtype=np.double)
    orders = np.ascontiguousarray(orders, dtype=np.int32)

    resp = np.zeros( (values.shape[0], y.shape[1]),dtype=np.double)

    values = np.ascontiguousarray(values, dtype=np.double)
    y = np.ascontiguousarray(y,dtype=np.double)

    d = len(smin)
    n = y.shape[1]

    smin_ptr = smin.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    smax_ptr = smax.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    orders_ptr = orders.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


    if d == 1:
        fun = fun_1d
    elif d == 2:
        fun = fun_2d
    elif d == 3:
        fun = fun_3d
    elif d == 4:
        fun = fun_4d
    else:
        raise Exception( 'Linear interpolation not implemented for order {}'.format(d))

    for i in range(values.shape[0]):

        x = values[i,:]
        output = resp[i,:]

        x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        fun(d, smin_ptr, smax_ptr, orders_ptr, x_ptr, n, y_ptr, output_ptr)

    return resp


if __name__ == '__main__':

    N_grid = 50
    N_fine_grid = 10000
    d = 2

    smin = [0.5]*d
    smax = [1.8]*d
    orders = [2,4]
    orders = orders[:d]
    from itertools import product
    grid = np.row_stack( product( *[np.linspace(smin[i],smax[i], orders[i]) for i in range(d)] )).T


    if d == 1:
        f = lambda x: np.row_stack( [np.cos(x), np.sin(x)] )
        v = f(grid[0,:])
    elif d == 2:
        f = lambda x,y: np.row_stack( [np.sin(x)*y, np.sin(y)*x] )
        v = f(grid[0,:], grid[1,:])
    elif d == 3:
        f = lambda x,y,z: np.row_stack( [np.sin(x)*y/np.sqrt(z), np.sin(y)*x] )
        v = f(grid[0,:], grid[1,:], grid[2,:])

    elif d == 4:
        f = lambda x,y,z,t: np.row_stack( [np.sin(x)*y/np.sqrt(z)*t, np.sin(y)*x] )
        v = f(grid[0,:], grid[1,:], grid[2,:], grid[3,:])

    #points = np.row_stack( product( *[np.linspace(smin[i],smax[i], N_fine_grid) for i in range(d)] )).T
    import numpy.random

    v = numpy.random.random((2,grid.shape[1]))
    points = numpy.random.random((d,N_fine_grid)) + np.array([1]*d)[:,None]

#    print(points)
    import time
    t = time.time()
    for i in range(10):
        out = multilinear_c(smin,smax,orders,v, points)
    s = time.time()

    print('New {}'.format(s-t))
    print(out.shape)

    import time
    from dolo.numeric.interpolation.multilinear import multilinear_interpolation

    t = time.time()
    for i in range(10):
        out_old = multilinear_interpolation( smin, smax, orders, v, points)
    print(out_old.shape)
    s = time.time()

    print('Old {}'.format(s-t))






    print(abs(out - out_old).max())