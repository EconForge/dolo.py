# cython: profile=True

cimport splines_cython

import numpy as np
cimport numpy as np
import cython
cimport cython


cdef Ugrid mygrid


from cython.parallel import prange, parallel

from operator import mul

from itertools import product

cdef class Splines1d:

    cdef UBspline_1d_d* __spline__
    cdef Ugrid __grid__
    cdef BCtype_d __boundaries__

    def __init__(self, smin, smax, orders, boundaries=4):
        smin = np.atleast_1d(smin)
        smax = np.atleast_1d(smax)
        orders = np.atleast_1d(orders)
        self.__grid__.start = smin[0]
        self.__grid__.end = smax[0]
        self.__grid__.num = orders[0]

        self.__boundaries__.lCode = 4
        self.__boundaries__.rCode = 4


    def set_spline(self, np.ndarray[np.double_t] table):

        cdef double* data = <double *> table.data

        self.__spline__ = create_UBspline_1d_d(self.__grid__, self.__boundaries__, data)

    @cython.boundscheck(False)
    def eval_spline(self, np.ndarray[np.double_t, ndim=1] table):
#
        cdef int n = len(table)
        cdef int i

        cdef UBspline_1d_d* spline = self.__spline__
        cdef np.ndarray[np.double_t, ndim=1] output = np.empty(n)

        cdef double* data_in = <double*> table.data
        cdef double* data = <double*> output.data

        with nogil, parallel():
            for i in prange(n):
                eval_UBspline_1d_d(spline, data_in[i], &data[i])

        return output


cdef class Splines2d:

    cdef UBspline_2d_d* __spline__
    cdef Ugrid __grid_x__, __grid_y__
    cdef BCtype_d __boundaries_x__, __boundaries_y__

    def __init__(self, smin, smax, orders, boundaries=4):

        smin = np.atleast_1d(smin)
        smax = np.atleast_1d(smax)
        orders = np.atleast_1d(orders)

        self.__grid_x__.start = smin[0]
        self.__grid_x__.end = smax[0]
        self.__grid_x__.num = orders[0]

        self.__boundaries_x__.lCode = 4
        self.__boundaries_x__.rCode = 4

        self.__grid_y__.start = smin[1]
        self.__grid_y__.end = smax[1]
        self.__grid_y__.num = orders[1]

        self.__boundaries_y__.lCode = 4
        self.__boundaries_y__.rCode = 4


    def set_spline(self, np.ndarray[np.double_t] values):

        cdef double* data = <double *> values.data

        self.__spline__ = create_UBspline_2d_d(self.__grid_x__, self.__grid_y__, self.__boundaries_x__, self.__boundaries_y__, data)

    @cython.boundscheck(False)
    def eval_spline(self, np.ndarray[np.double_t, ndim=2] points):
    #
        cdef int n = points.shape[1]
        cdef int i

        cdef UBspline_2d_d* spline = self.__spline__
        cdef np.ndarray[np.double_t, ndim=1] output = np.empty(n)

        cdef np.ndarray[np.double_t, ndim=1] points_x = points[0,:]
        cdef np.ndarray[np.double_t, ndim=1] points_y = points[1,:]

        cdef double* x_in = <double*> points_x.data
        cdef double* y_in = <double*> points_y.data

        cdef double* data = <double*> output.data

        with nogil, parallel():
            for i in prange(n):
                eval_UBspline_2d_d(spline, x_in[i], y_in[i], &data[i])

        return output


cdef class Splines3d:

    cdef UBspline_3d_d* __spline__
    cdef Ugrid __grid_x__, __grid_y__, __grid_z__
    cdef BCtype_d __boundaries_x__, __boundaries_y__, __boundaries_z__

    def __init__(self, smin, smax, orders, boundaries=4):

        smin = np.atleast_1d(smin)
        smax = np.atleast_1d(smax)
        orders = np.atleast_1d(orders)

        self.__grid_x__.start = smin[0]
        self.__grid_x__.end = smax[0]
        self.__grid_x__.num = orders[0]

        self.__boundaries_x__.lCode = 4
        self.__boundaries_x__.rCode = 4

        self.__grid_y__.start = smin[1]
        self.__grid_y__.end = smax[1]
        self.__grid_y__.num = orders[1]

        self.__boundaries_y__.lCode = 4
        self.__boundaries_y__.rCode = 4


        self.__grid_z__.start = smin[2]
        self.__grid_z__.end = smax[2]
        self.__grid_z__.num = orders[2]

        self.__boundaries_z__.lCode = 4
        self.__boundaries_z__.rCode = 4


    def set_values(self, np.ndarray[np.double_t] values):

        cdef double* data = <double *> values.data

        self.__spline__ = create_UBspline_3d_d(self.__grid_x__, self.__grid_y__, self.__grid_z__,
                    self.__boundaries_x__, self.__boundaries_y__, self.__boundaries_z__, data)

    @cython.boundscheck(False)
    def interpolate(self, np.ndarray[np.double_t, ndim=2] points):
    #
        cdef int n = points.shape[1]
        cdef int i

        cdef UBspline_3d_d* spline = self.__spline__
        cdef np.ndarray[np.double_t, ndim=1] output = np.empty(n)

        cdef np.ndarray[np.double_t, ndim=1] points_x = points[0,:]
        cdef np.ndarray[np.double_t, ndim=1] points_y = points[1,:]
        cdef np.ndarray[np.double_t, ndim=1] points_z = points[2,:]

        cdef double* x_in = <double*> points_x.data
        cdef double* y_in = <double*> points_y.data
        cdef double* z_in = <double*> points_z.data

        cdef double* data = <double*> output.data

        with nogil, parallel():
            for i in prange(n):
                eval_UBspline_3d_d(spline, x_in[i], y_in[i], z_in[i], &data[i])

        return output

#######################################
# Splines with multiple return values #
#######################################

cdef class MSplines1d:

    cdef multi_UBspline_1d_d* __spline__
    cdef Ugrid __grid_x__
    cdef BCtype_d __boundaries_x__
    cdef int __n_splines__
    cdef np.ndarray values

#    cpdef np.ndarray grid
#    cpdef int d

    def __init__(self, smin, smax, orders, boundaries=4, int n_splines=1):

        smin = np.atleast_1d(smin)
        smax = np.atleast_1d(smax)
        orders = np.atleast_1d(orders)

        self.__grid_x__.start = smin[0]
        self.__grid_x__.end = smax[0]
        self.__grid_x__.num = orders[0]

        self.__boundaries_x__.lCode = 4
        self.__boundaries_x__.rCode = 4

        self.__n_splines__ = n_splines

        self.__spline__ = create_multi_UBspline_1d_d(self.__grid_x__, self.__boundaries_x__, n_splines)

    def set_values(self, np.ndarray[np.double_t, ndim=2] values):

        cdef double* data
        cdef int n_splines = self.__n_splines__
        cdef np.ndarray[np.double_t,ndim=1] vals
        cdef multi_UBspline_1d_d*  spline = self.__spline__


        for i in range(n_splines):
            vals = values[i,:]
            data = <double *> vals.data

            set_multi_UBspline_1d_d(spline, i, data)

        self.values = values

    @cython.boundscheck(False)
    def interpolate(self, np.ndarray[np.double_t, ndim=2] points):
    #
        cdef int n = points.shape[1]
        cdef int i
        cdef int n_v = self.values.shape[0] # number of splines

        cdef multi_UBspline_1d_d* spline = self.__spline__

        cdef np.ndarray[np.double_t, ndim=1] points_x = points[0,:]

        cdef double* x_in = <double*> points_x.data

        cdef np.ndarray[np.double_t, ndim=2] output = np.empty( (n, n_v), dtype=np.double )
        cdef np.ndarray[np.double_t, ndim=3] doutput = np.empty( (n, n_v, 1), dtype=np.double )

        cdef double* data = <double*> output.data
        cdef double* d_data = <double*> doutput.data

        with nogil, parallel():
            for i in prange(n):
                eval_multi_UBspline_1d_d_vg(spline, x_in[i], &data[i*n_v], &d_data[i*n_v])

        return [output, doutput]

cdef class MSplines2d:

    cdef multi_UBspline_2d_d* __spline__
    cdef Ugrid __grid_x__, __grid_y__
    cdef BCtype_d __boundaries_x__, __boundaries_y__
    cdef int __n_splines__
    cdef np.ndarray values
    cdef np.ndarray grid
    cdef int d

    def __init__(self, smin, smax, orders, boundaries=4, int n_splines=1):

        smin = np.atleast_1d(smin)
        smax = np.atleast_1d(smax)
        orders = np.atleast_1d(orders)

        self.__grid_x__.start = smin[0]
        self.__grid_x__.end = smax[0]
        self.__grid_x__.num = orders[0]

        self.__boundaries_x__.lCode = 4
        self.__boundaries_x__.rCode = 4

        self.__grid_y__.start = smin[1]
        self.__grid_y__.end = smax[1]
        self.__grid_y__.num = orders[1]

        self.__boundaries_y__.lCode = 4
        self.__boundaries_y__.rCode = 4

        self.__n_splines__ = n_splines

        self.__spline__ = create_multi_UBspline_2d_d(self.__grid_x__, self.__grid_y__, self.__boundaries_x__, self.__boundaries_y__, n_splines)

    def set_values(self, np.ndarray[np.double_t, ndim=2] values):

        cdef double* data
        cdef int n_splines = self.__n_splines__
        cdef np.ndarray[np.double_t,ndim=1] vals
        cdef multi_UBspline_2d_d*  spline = self.__spline__


        for i in range(n_splines):
            vals = values[i,:]
            data = <double *> vals.data

            set_multi_UBspline_2d_d(spline, i, data)

        self.values = values

    @cython.boundscheck(False)
    def interpolate(self, np.ndarray[np.double_t, ndim=2] points):
    #
        cdef int n = points.shape[1]
        cdef int i
        cdef int n_v = self.values.shape[0] # number of splines

        cdef multi_UBspline_2d_d* spline = self.__spline__

        cdef np.ndarray[np.double_t, ndim=1] points_x = points[0,:]
        cdef np.ndarray[np.double_t, ndim=1] points_y = points[1,:]

        cdef double* x_in = <double*> points_x.data
        cdef double* y_in = <double*> points_y.data

        cdef np.ndarray[np.double_t, ndim=2] output = np.empty( (n, n_v), dtype=np.double )
        cdef np.ndarray[np.double_t, ndim=3] doutput = np.empty( (n, n_v, 2), dtype=np.double )

        cdef double* data = <double*> output.data
        cdef double* d_data = <double*> doutput.data


        with nogil, parallel():
            for i in prange(n):
                eval_multi_UBspline_2d_d_vg(spline, x_in[i], y_in[i], &data[i*n_v], &d_data[2*i*n_v])

        return [output, doutput]

cdef class MSplines3d:

    cdef multi_UBspline_3d_d* __spline__
    cdef Ugrid __grid_x__, __grid_y__, __grid_z__
    cdef BCtype_d __boundaries_x__, __boundaries_y__, __boundaries_z__
    cdef int __n_splines__
    cdef np.ndarray values
    cdef np.ndarray grid
    cdef int d

    def __init__(self, smin, smax, orders, boundaries=4, int n_splines=1):

        smin = np.atleast_1d(smin)
        smax = np.atleast_1d(smax)
        orders = np.atleast_1d(orders)

        self.__grid_x__.start = smin[0]
        self.__grid_x__.end = smax[0]
        self.__grid_x__.num = orders[0]

        self.__boundaries_x__.lCode = 4
        self.__boundaries_x__.rCode = 4

        self.__grid_y__.start = smin[1]
        self.__grid_y__.end = smax[1]
        self.__grid_y__.num = orders[1]

        self.__boundaries_y__.lCode = 4
        self.__boundaries_y__.rCode = 4


        self.__grid_z__.start = smin[2]
        self.__grid_z__.end = smax[2]
        self.__grid_z__.num = orders[2]

        self.__boundaries_z__.lCode = 4
        self.__boundaries_z__.rCode = 4

        self.__n_splines__ = n_splines

        self.__spline__ = create_multi_UBspline_3d_d(self.__grid_x__, self.__grid_y__, self.__grid_z__,
            self.__boundaries_x__, self.__boundaries_y__, self.__boundaries_z__, n_splines)


    def set_values(self, np.ndarray[np.double_t, ndim=2] values):

        cdef double* data
        cdef int n_splines = self.__n_splines__
        cdef np.ndarray[np.double_t,ndim=1] vals
        cdef multi_UBspline_3d_d*  spline = self.__spline__


        for i in range(n_splines):
            vals = values[i,:]
            data = <double *> vals.data

            set_multi_UBspline_3d_d(spline, i, data)

        self.values = values

    @cython.boundscheck(False)
    def interpolate(self, np.ndarray[np.double_t, ndim=2] points):
    #
        cdef int n = points.shape[1]
        cdef int i
        cdef int n_v = self.values.shape[0] # number of splines

        cdef multi_UBspline_3d_d* spline = self.__spline__

        cdef np.ndarray[np.double_t, ndim=1] points_x = points[0,:]
        cdef np.ndarray[np.double_t, ndim=1] points_y = points[1,:]
        cdef np.ndarray[np.double_t, ndim=1] points_z = points[2,:]

        cdef double* x_in = <double*> points_x.data
        cdef double* y_in = <double*> points_y.data
        cdef double* z_in = <double*> points_z.data

        cdef np.ndarray[np.double_t, ndim=2] output = np.empty( (n, n_v), dtype=np.double )
        cdef np.ndarray[np.double_t, ndim=3] doutput = np.empty( (n, n_v, 3), dtype=np.double )

        cdef double* data = <double*> output.data
        cdef double* d_data = <double*> doutput.data


        with nogil, parallel():
            for i in prange(n):
                eval_multi_UBspline_3d_d_vg(spline, x_in[i], y_in[i], z_in[i], &data[i*n_v], &d_data[3*i*n_v])

        return [output, doutput]


class MultivariateSplines:

    def __init__(self, smin, smax, orders):

        self.d = len(smin)
        assert(len(smax) == self.d)
        assert(len(orders) == self.d)
        self.grid = np.ascontiguousarray( np.row_stack( product(*[np.linspace(smin[i], smax[i], orders[i]) for i in range(self.d)]) ).T )
        self.smin = smin
        self.smax = smax
        self.orders = orders
        self.__splines__ = None

    def set_values(self, values):
        n_v = values.shape[0]
        if self.__splines__ is None:
            self.n_v = n_v
            if self.d == 1:
                self.__splines__ = MSplines1d(self.smin, self.smax, self.orders, n_splines=n_v)
            elif self.d == 2:
                self.__splines__ = MSplines2d(self.smin, self.smax, self.orders, n_splines=n_v)
            elif self.d == 3:
                self.__splines__ = MSplines3d(self.smin, self.smax, self.orders, n_splines=n_v)
        else:
            if n_v != self.n_v:
                raise Exception('Trying to set {} values for the interpolant. Expected : {}'.format(n_v, self.n_v))
        self.__splines__.set_values(values)

    def interpolate(self, points, with_derivatives=False):

        projection = np.minimum( points, self.smax[:,None]-0.0000000001)
        projection = np.maximum( projection, self.smin[:,None]+0.0000000001)

        [value, d_value] = self.__splines__.interpolate(projection)

        value = np.ascontiguousarray( value.T )
        d_value = np.ascontiguousarray( np.rollaxis(d_value, 0, 3 ) )

        delta = (points - projection)

        # TODO : correct only outside observations
        for i in range(value.shape[0]):
            value[i,:] += (delta*d_value[i,:,:]).sum(axis=0)

        if with_derivatives:
            return [value,d_value]
        else:
            return value


    def __call__(self, s):
        return self.interpolate(s)


