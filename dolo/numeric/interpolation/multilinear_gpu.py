# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
#
from __future__ import division
#
from numpy import *
import numpy as np

import numpy
from multilinear_cython import multilinear_interpolation as mlininterp_cpu

real_type = numpy.single

sourcecode = '/home/pablo/Programmation/bigeco//dolo/dolo/numeric/interpolation/multilinear_gpu.c'

with file(sourcecode) as f:
    txt = f.read()

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda.gpuarray import to_gpu
mod = SourceModule(txt)
multilinear_kernels = []
for i in range(1,4):
    funname = "multilinear_interpolation_{}d".format(i)
    kernel = mod.get_function(funname)
    multilinear_kernels.append(kernel)


def multilinear_interpolation( smin, smax, orders, x, y):

    if x.ndim > 1:
        return numpy.row_stack([multilinear_interpolation( smin, smax, orders, xx, y) for xx in x])

    n_s = len(smin)


    NN = y.shape[1]

    N = numpy.int32(y.shape[1])
    smin = array(smin,dtype=real_type)
    smax = array(smax,dtype=real_type)
    orders = array(orders,dtype=numpy.int32)


    x = array(x,dtype=real_type)
    y = array(y,dtype=real_type)
    
    values = array(x,dtype=real_type)
    dest = zeros_like(y[0,:],dtype=real_type)

    mfun = multilinear_kernels[n_s-1]

    gpu_points = to_gpu(y)
    gpu_dest = to_gpu(dest)

    N_blocks = int( numpy.ceil(NN/96) )
    
    args = [drv.In(smin), drv.In(smax), drv.In(orders), drv.In(values), N, gpu_points, gpu_dest]
    mfun( *args, block=(N_blocks,1,1), grid=(96,1))

    dest = gpu_dest.get()
    
    return dest

    
    
if __name__ == '__main__':
    N_grid = 10
#    N_fine_grid = 96*100
    N_fine_grid = 96*1000
    d = 3
    smin = array([0.5]*d)
    smax = array([1.8]*d)
    #orders = [N_grid]*d
    orders = array([100]*d)
    d = len(orders)

    from itertools import product
    grid = np.row_stack( product( *[np.linspace(smin[i],smax[i], orders[i]) for i in range(d)] )).T
    
    if d == 1:
        f = lambda x: np.row_stack( [np.cos(x), np.sin(x)] )
        v = f(grid[0,:])
    elif d == 2:
        f = lambda x,y: np.row_stack( [np.sin(x)*y, np.sin(y)*x] )
        v = f(grid[0,:], grid[1,:])
    elif d == 3:
        f = lambda x,y,z: np.row_stack( [np.sin(x)*y, np.sin(y)*x*z] )
        v = f(grid[0,:], grid[1,:], grid[2,:])
   
    values = v
    
    import numpy.random
    points = numpy.random.random((d,N_fine_grid))
    #points = atleast_2d( linspace( smin[0] + 0.001, smax[0] - 0.001, N_fine_grid) )
    
    points = minimum( smax[:,None]-0.001, points)
    points = maximum( smin[:,None]+0.001, points)
    
    import time
    
    t = time.time()
    
    dest = multilinear_interpolation( smin, smax, orders, values, points )
    
    s = time.time()

    u = time.time()
    check = mlininterp_cpu( smin, smax, orders, atleast_2d(values), points )
    v = time.time()
        
    
    print('Error : {}'.format( abs(check - dest).max() ) )    
    
    print('Elapsed (CUDA) : {}'.format(s-t))
    print('Elapsed (NUMPY) : {}'.format(v-u))
    
    # <codecell>
    
