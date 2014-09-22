from __future__ import absolute_import
from __future__ import division

import numpy

from ..filter_cubic_splines import filter_coeffs

from ..eval_cubic_splines import vec_eval_cubic_multi_spline

from ..splines import MultivariateCubicSplines

from dolo.numeric.misc import mlinspace


def test_splines_filter():

    a = numpy.array([0.0, 0.0])
    b = numpy.array([1.0, 1.0])
    orders = numpy.array([10, 10])

    grid = mlinspace(a, b, orders)

    fun = lambda x, y: numpy.sin(x ** 2 + y ** 2)

    raw_vals = fun(grid[:, 0], grid[:, 1])
    raw_vals = raw_vals[:, None]


    coeffs = filter_coeffs(a, b, orders, raw_vals)

    return [a, b, orders, raw_vals, coeffs]


def test_eval_splines_2():

    import time

    [a, b, orders, raw_vals, coeffs] = test_splines_filter()

    fine_grid = mlinspace(a, b, [10000, 10000])


    # print(output.flags)
    N = fine_grid.shape[0]

    output = numpy.zeros((N, 1))

    cc = numpy.ascontiguousarray(coeffs[None,:,:])

    import time
    t1 = time.time()
    vec_eval_cubic_multi_spline(a, b, orders, cc, fine_grid, output)
    t2 = time.time()
    print(t2-t1)

    t1 = time.time()
    vec_eval_cubic_multi_spline(a, b, orders, cc, fine_grid, output)
    t2 = time.time()
    print(t2-t1)


def test_eval_splines_3():

    import time

    [a, b, orders, raw_vals, coeffs] = test_splines_filter()

    mvals = raw_vals[None,:]

    csp = MultivariateCubicSplines(a,b,orders)

    csp.set_mvalues(mvals)

    fine_grid = mlinspace(a, b, [10, 10])


    values = csp(fine_grid)


if __name__ == '__main__':

    import nose
    nose.run(argv=[__file__, '--with-doctest', '-vv'])
