from __future__ import division

import numpy
from splines_cython import *

d = 3
N = 50
smin = np.array( [0] * d )
smax = np.array( [1] * d )
orders = np.array( [N]*d )
n_v = 2

from itertools import product
grid = np.row_stack( product( *[np.linspace(smin[i],smax[i], orders[i]) for i in range(d)] )).T
grid = numpy.ascontiguousarray(grid)

#f = lambda x, y, z: x*numpy.sin(y)*numpy.cos(z)
#values = f(grid[0,:], grid[1,:], grid[2,:])
if d==1:
    f = lambda x: np.row_stack([numpy.sin(x)]*n_v)
    values = f(grid[0,:])
elif d==2:
    f = lambda x, y: np.row_stack([x*numpy.sin(y)]*n_v)
    values = f(grid[0,:], grid[1,:])

elif d==3:
    f = lambda x, y, z: np.row_stack([x*numpy.sin(y)*numpy.cos(z)]*n_v)
    values = f(grid[0,:], grid[1,:], grid[2,:])

minterp = MultivariateSplines(smin, smax, orders)

#interp = MSplines3d(smin,smax, orders, n_splines=n_v)

#if d == 1:
#    minterp = MSplines1d(smin, smax, orders, n_splines=n_v)
#elif d == 2:
#    minterp = MSplines2d(smin, smax, orders, n_splines=n_v)
#elif d == 3:
#    interp = Splines3d(smin, smax, orders)
#    minterp = MSplines3d(smin, smax, orders, n_splines=n_v)

import numpy.random

#interp.set_values(values.flatten())
minterp.set_values(values)
#interp.set_values(values)


K = 100*100*100
points = numpy.random.random( (d, K) )

#points = numpy.maximum( points, 0.0001)
#points = numpy.minimum( points, 0.9999)

import time
#
#s1 = time.time()
#for i in range(10):
#    [val1,dval1] = interp.interpolate(points)
#t1 = time.time()


s2 = time.time()
for i in range(10):
    [val, dval] = minterp.interpolate(points, with_derivatives=True)
t2 = time.time()

print('ok')
#
#from splines_c import MultivariateSpline as TestSplines
#ts = TestSplines(smin,smax,orders)
#ts.set_values(values)
#[test,dtest] = ts.interpolate(points, with_derivatives=True)
#
#ts.set_values(values)
#test = ts(points)
#
#
#print(abs(test-val).max())
#print('derivatives')
#print(dtest-dval)
#
#print('Error on derivatives {}'.format(abs(dtest-dval).max()))


#print('Elapsed (single): {}'.format((t1-s1)/10))
print(dval.shape)

print('Elapsed (multiple): {}'.format((t2-s2)/10/n_v))

err = val - f(*points)
print('Error : {}'.format(abs(err).max()))
