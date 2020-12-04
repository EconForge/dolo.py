from numba import jit

import numpy
from math import sqrt

# from matplotlib import pyplot as plt
#
# %matplotlib inline
from numpy import linspace

from numba import guvectorize

ttol = 1e-10


@jit
def PhiFB(a, b):
    return a + b - sqrt(a ** 2 + b ** 2)


@jit
def smooth_FB(f, x, a, b):
    return PhiFB(-PhiFB(f, x - a), b - x)


@jit
def PhiFB_(a, b):
    sq = sqrt(a ** 2 + b ** 2)
    x = a + b - sq
    if sq != 0.0:
        dx_da = 1.0 - a / sq
        dx_db = 1.0 - b / sq
    else:
        dx_da = 0.0
        dx_db = 0.0
    return (x, dx_da, dx_db)


@jit
def smooth_FB_(f, x, a, b, df):
    # P1 = Phi(f, x-a)
    # P2 = Phi(-P1, b-x)
    # f>=0 x-a>=04
    v, dv_df, dv_dx = PhiFB_(f, x - a)
    P1, dP1_dx = (v, dv_df * df + dv_dx)
    v, dv_df, dv_dx = PhiFB_(-P1, b - x)
    P2, dP2_dx = (v, -dv_df * dP1_dx - dv_dx)
    # phi, d_phi_d_x
    return (P2, dP2_dx)


import numpy as np


@jit
def smooth(f, x, a, b, df):
    n_ms, N, n_x = f.shape
    V = np.zeros_like(f)
    dV = df.copy()
    for i in range(n_ms):
        for n in range(N):
            for j in range(n_x):
                v, dv = smooth_FB_(
                    f[i, n, j], x[i, n, j], a[i, n, j], b[i, n, j], df[i, n, j, j]
                )
                V[i, n, j] = v
                # dv[i,n,j] =
    return V


#


@jit
def smooth_(f, x, a, b):
    n_ms, N, n_x = f.shape
    V = np.zeros_like(f)
    for i in range(n_ms):
        for n in range(N):
            for j in range(n_x):
                v = smooth_FB(f[i, n, j], x[i, n, j], a[i, n, j], b[i, n, j])
                V[i, n, j] = v
                # dv[i,n,j] =
    return V


#

# @jit
# def smooth(f,x,a,b,df):
#     n_ms, N, n_x = f.shape
#     V = np.zeros_like(f)
#     dV = df.copy()
#     for i in range(n_ms):
#         for n in range(N):
#             for j in range(n_x):
#                 v,dv = smooth_FB_(f[i,n,j], x[i,n,j], a[i,n,j], b[i,n,j], df[i,n,j,j])
#                 V[i,n,j] = v
#                 # dv[i,n,j] =
#     return V


#
#
# import numpy as np
#
# xvec = numpy.linspace(-0.1,1.1,2000)
# f = lambda x: -(x-1.2)*0.2
# fvec = f(xvec)
# dfvec = xvec*0.0-0.2
#
# phi_vec = np.array( [smooth_FB(fvec[i], xvec[i], 0.0, 1.0, dfvec[i]) for i in range(len(xvec))] )
# phi_vec_1 = np.array( [smooth_FB_(fvec[i], xvec[i], 0.0, 1.0, dfvec[i])[0] for i in range(len(xvec))] )
# d_phi = np.array( [smooth_FB_(fvec[i], xvec[i], 0.0, 1.0, dfvec[i])[1] for i in range(len(xvec))] )
#
# emp_diff = (phi_vec[1:]-phi_vec[:-1])/(xvec[1:]-xvec[:-1])
#
#
# plt.subplot(211)
# plt.plot(xvec, fvec)
# plt.plot(xvec,phi_vec)
# plt.plot(xvec, xvec*0, linestyle='--')
# plt.subplot(212)
# plt.plot(xvec, d_phi)
# plt.plot(xvec[:-1], emp_diff)
