from __future__ import division

import numpy as np
from cython import double, float

ctypedef fused floating:
    float
    double

A44d = np.array([
   [-1.0/6.0,  3.0/6.0, -3.0/6.0, 1.0/6.0],
   [ 3.0/6.0, -6.0/6.0,  0.0/6.0, 4.0/6.0],
   [-3.0/6.0,  3.0/6.0,  3.0/6.0, 1.0/6.0],
   [ 1.0/6.0,  0.0/6.0,  0.0/6.0, 0.0/6.0]
])

dA44d = np.zeros((4,4))
for i in range(1,4):
    dA44d[:,i] = A44d[:,i-1]*(4-i)


d2A44d = np.zeros((4,4))
for i in range(1,4):
    d2A44d[:,i] = dA44d[:,i-1]*(4-i)


import cython
from libc.math cimport floor
from cython.parallel import parallel, prange
from cython import nogil



@cython.boundscheck(False)
@cython.wraparound(False)
def vec_eval_cubic_multi_spline_1( floating[:] smin, floating[:] smax, long[:] orders,  floating[:,::1] coefs, floating[:,::1] svec, floating[:,::1] vals):


    cdef int M0 = orders[0+1]
    cdef floating start0 = smin[0]
    cdef floating dinv0 = (orders[0]-1.0)/(smax[0]-smin[0])

    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32

    cdef int N = svec.shape[0]
    cdef int n

    cdef int n_x = coefs.shape[0]
    cdef int k


    cdef floating[:,::1] Ad = np.array(A44d, dtype=dtype)
    cdef floating[:,::1] dAd = np.array(dA44d, dtype=dtype)

    cdef int i0
    cdef floating x0
    cdef floating u0
    cdef floating t0
    cdef floating extrap0

    cdef floating Phi0_0, Phi0_1, Phi0_2, Phi0_3
    cdef floating tp0_0, tp0_1, tp0_2, tp0_3

    #cdef floating [::1] C = coefs
    #cdef floating [:] vals = np.zeros(N, dtype=dtype)

    cdef floating tpx_0, tpx_1, tpx_2, tpx_3
    cdef floating tpy_0, tpy_1, tpy_2, tpy_3

    with nogil, parallel():

        for n in prange(N):

            x0 = svec[n,0]
            u0 = (x0 - start0)*dinv0
            i0 = <int> u0
            i0 = max( min(i0,M0-2), 0 )
            t0 = u0-i0


            # 
            # extrap0 = 0 if (t0 < 0 or t0 >= 1) else 1
            # 

            tp0_0 = t0*t0*t0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;

            if t0 < 0:
                Phi0_0 = dAd[0,3]*t0 + Ad[0,3]
                Phi0_1 = dAd[1,3]*t0 + Ad[1,3]
                Phi0_2 = dAd[2,3]*t0 + Ad[2,3]
                Phi0_3 = dAd[3,3]*t0 + Ad[3,3]
            elif t0 > 1:
                Phi0_0 = (3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t0-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3])
                Phi0_1 = (3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t0-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3])
                Phi0_2 = (3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t0-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3])
                Phi0_3 = (3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t0-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3])
            else:
                Phi0_0 = (Ad[0,0]*tp0_0 + Ad[0,1]*tp0_1 + Ad[0,2]*tp0_2 + Ad[0,3]*tp0_3)
                Phi0_1 = (Ad[1,0]*tp0_0 + Ad[1,1]*tp0_1 + Ad[1,2]*tp0_2 + Ad[1,3]*tp0_3)
                Phi0_2 = (Ad[2,0]*tp0_0 + Ad[2,1]*tp0_1 + Ad[2,2]*tp0_2 + Ad[2,3]*tp0_3)
                Phi0_3 = (Ad[3,0]*tp0_0 + Ad[3,1]*tp0_1 + Ad[3,2]*tp0_2 + Ad[3,3]*tp0_3)

            for k in range(n_x):
                vals[n, k] = Phi0_0*(coefs[k,i0+0]) + Phi0_1*(coefs[k,i0+1]) + Phi0_2*(coefs[k,i0+2]) + Phi0_3*(coefs[k,i0+3])



@cython.boundscheck(False)
@cython.wraparound(False)
def vec_eval_cubic_multi_spline_2( floating[:] smin, floating[:] smax, long[:] orders,  floating[:,:,::1] coefs, floating[:,::1] svec, floating[:,::1] vals):


    cdef int M0 = orders[0+1]
    cdef floating start0 = smin[0]
    cdef floating dinv0 = (orders[0]-1.0)/(smax[0]-smin[0])
    cdef int M1 = orders[1+1]
    cdef floating start1 = smin[1]
    cdef floating dinv1 = (orders[1]-1.0)/(smax[1]-smin[1])

    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32

    cdef int N = svec.shape[0]
    cdef int n

    cdef int n_x = coefs.shape[0]
    cdef int k


    cdef floating[:,::1] Ad = np.array(A44d, dtype=dtype)
    cdef floating[:,::1] dAd = np.array(dA44d, dtype=dtype)

    cdef int i0, i1
    cdef floating x0, x1
    cdef floating u0, u1
    cdef floating t0, t1
    cdef floating extrap0, extrap1

    cdef floating Phi0_0, Phi0_1, Phi0_2, Phi0_3, Phi1_0, Phi1_1, Phi1_2, Phi1_3
    cdef floating tp0_0, tp0_1, tp0_2, tp0_3, tp1_0, tp1_1, tp1_2, tp1_3

    #cdef floating [:,::1] C = coefs
    #cdef floating [:] vals = np.zeros(N, dtype=dtype)

    cdef floating tpx_0, tpx_1, tpx_2, tpx_3
    cdef floating tpy_0, tpy_1, tpy_2, tpy_3

    with nogil, parallel():

        for n in prange(N):

            x0 = svec[n,0]
            x1 = svec[n,1]
            u0 = (x0 - start0)*dinv0
            i0 = <int> u0
            i0 = max( min(i0,M0-2), 0 )
            t0 = u0-i0
            u1 = (x1 - start1)*dinv1
            i1 = <int> u1
            i1 = max( min(i1,M1-2), 0 )
            t1 = u1-i1


            # 
            # extrap0 = 0 if (t0 < 0 or t0 >= 1) else 1
            # 
            # extrap1 = 0 if (t1 < 0 or t1 >= 1) else 1
            # 

            tp0_0 = t0*t0*t0;  tp0_1 = t0*t0;  tp0_2 = t0;  tp0_3 = 1.0;
            tp1_0 = t1*t1*t1;  tp1_1 = t1*t1;  tp1_2 = t1;  tp1_3 = 1.0;

            if t0 < 0:
                Phi0_0 = dAd[0,3]*t0 + Ad[0,3]
                Phi0_1 = dAd[1,3]*t0 + Ad[1,3]
                Phi0_2 = dAd[2,3]*t0 + Ad[2,3]
                Phi0_3 = dAd[3,3]*t0 + Ad[3,3]
            elif t0 > 1:
                Phi0_0 = (3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t0-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3])
                Phi0_1 = (3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t0-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3])
                Phi0_2 = (3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t0-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3])
                Phi0_3 = (3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t0-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3])
            else:
                Phi0_0 = (Ad[0,0]*tp0_0 + Ad[0,1]*tp0_1 + Ad[0,2]*tp0_2 + Ad[0,3]*tp0_3)
                Phi0_1 = (Ad[1,0]*tp0_0 + Ad[1,1]*tp0_1 + Ad[1,2]*tp0_2 + Ad[1,3]*tp0_3)
                Phi0_2 = (Ad[2,0]*tp0_0 + Ad[2,1]*tp0_1 + Ad[2,2]*tp0_2 + Ad[2,3]*tp0_3)
                Phi0_3 = (Ad[3,0]*tp0_0 + Ad[3,1]*tp0_1 + Ad[3,2]*tp0_2 + Ad[3,3]*tp0_3)

            if t1 < 0:
                Phi1_0 = dAd[0,3]*t1 + Ad[0,3]
                Phi1_1 = dAd[1,3]*t1 + Ad[1,3]
                Phi1_2 = dAd[2,3]*t1 + Ad[2,3]
                Phi1_3 = dAd[3,3]*t1 + Ad[3,3]
            elif t1 > 1:
                Phi1_0 = (3*Ad[0,0] + 2*Ad[0,1] + Ad[0,2])*(t1-1) + (Ad[0,0]+Ad[0,1]+Ad[0,2]+Ad[0,3])
                Phi1_1 = (3*Ad[1,0] + 2*Ad[1,1] + Ad[1,2])*(t1-1) + (Ad[1,0]+Ad[1,1]+Ad[1,2]+Ad[1,3])
                Phi1_2 = (3*Ad[2,0] + 2*Ad[2,1] + Ad[2,2])*(t1-1) + (Ad[2,0]+Ad[2,1]+Ad[2,2]+Ad[2,3])
                Phi1_3 = (3*Ad[3,0] + 2*Ad[3,1] + Ad[3,2])*(t1-1) + (Ad[3,0]+Ad[3,1]+Ad[3,2]+Ad[3,3])
            else:
                Phi1_0 = (Ad[0,0]*tp1_0 + Ad[0,1]*tp1_1 + Ad[0,2]*tp1_2 + Ad[0,3]*tp1_3)
                Phi1_1 = (Ad[1,0]*tp1_0 + Ad[1,1]*tp1_1 + Ad[1,2]*tp1_2 + Ad[1,3]*tp1_3)
                Phi1_2 = (Ad[2,0]*tp1_0 + Ad[2,1]*tp1_1 + Ad[2,2]*tp1_2 + Ad[2,3]*tp1_3)
                Phi1_3 = (Ad[3,0]*tp1_0 + Ad[3,1]*tp1_1 + Ad[3,2]*tp1_2 + Ad[3,3]*tp1_3)

            for k in range(n_x):
                vals[n, k] = Phi0_0*(Phi1_0*(coefs[k,i0+0,i1+0]) + Phi1_1*(coefs[k,i0+0,i1+1]) + Phi1_2*(coefs[k,i0+0,i1+2]) + Phi1_3*(coefs[k,i0+0,i1+3])) + Phi0_1*(Phi1_0*(coefs[k,i0+1,i1+0]) + Phi1_1*(coefs[k,i0+1,i1+1]) + Phi1_2*(coefs[k,i0+1,i1+2]) + Phi1_3*(coefs[k,i0+1,i1+3])) + Phi0_2*(Phi1_0*(coefs[k,i0+2,i1+0]) + Phi1_1*(coefs[k,i0+2,i1+1]) + Phi1_2*(coefs[k,i0+2,i1+2]) + Phi1_3*(coefs[k,i0+2,i1+3])) + Phi0_3*(Phi1_0*(coefs[k,i0+3,i1+0]) + Phi1_1*(coefs[k,i0+3,i1+1]) + Phi1_2*(coefs[k,i0+3,i1+2]) + Phi1_3*(coefs[k,i0+3,i1+3]))

