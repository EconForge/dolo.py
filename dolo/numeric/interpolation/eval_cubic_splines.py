from __future__ import print_function


from numba import jit, njit

import numpy

## the functions in this file work for any dimension (d<=3)
## they can optionnally allocate memory for the result

def eval_cubic_spline(a, b, orders, coefs, point, value):

    pass


def eval_cubic_spline_d(a, b, orders, coefs, point, value, dvalue):

    pass


def eval_cubic_multi_spline(a, b, orders, mcoefs, point, values):

    pass


def eval_cubic_multi_spline_d(a, b, orders, mcoefs, point, values, dvalues):

    pass


from eval_cubic_splines_numba import vec_eval_cubic_multi_spline_1, vec_eval_cubic_multi_spline_2

# from eval_cubic_splines_numba import eval_cubic_multi_spline_3
from eval_cubic_splines_numba import Ad, dAd

def vec_eval_cubic_multi_spline(a, b, orders, mcoefs, points, values=None):



    d = a.shape[0]

    if values is None:

        N = points.shape[0]
        n_sp = mcoefs.shape[0]
        values = numpy.empty((N, n_sp))

    if d == 1:
        vec_eval_cubic_multi_spline_1(a, b, orders, mcoefs, points, values)

    elif d == 2:
        vec_eval_cubic_multi_spline_2(a, b, orders, mcoefs, points, values)

    elif d == 3:
        vec_eval_cubic_multi_spline_3(a, b, orders, mcoefs, points, values)

    return values




def vec_eval_cubic_multi_spline_d(a, b, orders, mcoefs, points, values=None, dvalues=None):

    pass
