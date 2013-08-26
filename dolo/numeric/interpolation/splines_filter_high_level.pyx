
from __future__ import division

import numpy as np
from cython import double

import cython
from libc.math cimport floor
from cython.parallel import parallel, prange
from cython import nogil


cdef double[:] find_coefs_1d(delta_inv, M, data_in):

    import scipy
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    data = np.array(data_in).copy()
    N = M + 2
    if len(data) == N:
#        data = data[0:N-2]
        data = data[1:N-1]

    basis = np.array([1.0/6.0, 2.0/3.0, 1.0/6.0])
    rhs = np.concatenate( [[0], data, [0]])

    vals = np.tile( basis, M )
    co_x = np.repeat( np.arange(M) , 3 ) + 1
    co_y = np.tile( np.arange(3), M ) + co_x - 1

    db = 4
    initial = np.array( [1,-2,1,0] )*delta_inv*delta_inv
    final = np.array( [0,1,-2,1] )*delta_inv*delta_inv
#    final += np.array( [-2, 3, 0, -1] )*delta_inv*delta_inv*delta_inv
#    initial += np.array( [-1, 0, 3, -2] )*delta_inv*delta_inv*delta_inv

    vals = np.concatenate( [initial, vals, final] )
    co_x = np.concatenate( [np.zeros(db), co_x, np.ones(db)*(M+1)] )
    co_y = np.concatenate( [np.arange(db), co_y, np.arange(M+2-db,M+2)] )

    spmat = coo_matrix( (vals, (co_x, co_y)), shape=(N,N) )
    spmat = spmat.tocsr()

    res = spsolve( spmat, rhs )

    return res


def filter_coeffs_1d(double[:] dinv, double[:] data):

  M = data.shape[0]
  N = M+2
  coefs = np.zeros(N)
  coefs[:] = find_coefs_1d(dinv[0], M, data)

  return coefs

def filter_coeffs_2d(double[:] dinv, double[:,:] data):
    Mx = data.shape[0]
    My = data.shape[1]

    Nx = Mx+2
    Ny = My+2

    cdef double [:,:] coefs = np.zeros((Nx,Ny))

    cdef int iy, ix

    # First, solve in the X-direction
    for iy in range(My):
#        find_coefs_1d(dinv[0], Mx, data[:,iy], coefs[:,iy])
        coefs[:,iy+1] = find_coefs_1d(dinv[0], Mx, data[:,iy])


    # Now, solve in the Y-directiona
    for ix in range(Nx):
        coefs[ix,:] =  find_coefs_1d(dinv[1], My, coefs[ix,:])
    return coefs

def filter_coeffs_3d(double[:] dinv, double[:,:,:] data):

    Mx = data.shape[0]
    My = data.shape[1]
    Mz = data.shape[2]

    Nx = Mx+2
    Ny = My+2
    Nz = Mz+2

    cdef double [:,:,:] coefs = np.zeros((Nx,Ny,Nz))

    cdef int iy, ix, iz

    # First, solve in the X-direction
    for iy in range(My):
        for iz in range(Mz):
            coefs[:,iy+1,iz+1] = find_coefs_1d(dinv[0], Mx, data[:,iy,iz])
    # Now, solve in the Y-direction
    for ix in range(Nx):
        for iz in range(Mz):
            coefs[ix,:,iz+1] = find_coefs_1d(dinv[1], My, coefs[ix,:,iz+1])

    # Now, solve in the Z-direction
    for ix in range(Nx):
        for iy in range(Ny):
            coefs[ix,iy,:] = find_coefs_1d(dinv[2], Mz, coefs[ix,iy,:])

    return coefs

def filter_coeffs_4d(double[:] dinv, double[:,:,:,:] data):

    M0 = data.shape[0]
    M1 = data.shape[1]
    M2 = data.shape[2]
    M3 = data.shape[3]

    N0 = M0+2
    N1 = M1+2
    N2 = M2+2
    N3 = M3+2


    N0 = M0+2;
    N1 = M1+2;
    N2 = M2+2;
    N3 = M3+2;


    cdef double [:,:,:,:] coefs = np.zeros((N0,N1,N2,N3))

    cdef int i0, i1, i2, i3

    # First, solve in the X-direction
    for i1 in range(M1):
        for i2 in range(M2):
            for i3 in range(M3):
                coefs[:,i1+1,i2+1,i3+1] = find_coefs_1d(dinv[0], M0, data[:,i1,i2,i3])
    for i0 in range(N0):
        for i2 in range(M2):
            for i3 in range(M3):
                coefs[i0,:,i2+1,i3+1] = find_coefs_1d(dinv[1], M1, coefs[i0,:,i2+1,i3+1])

    for i0 in range(N0):
        for i1 in range(N1):
            for i3 in range(M3):
                coefs[i0,i1,:,i3+1] = find_coefs_1d(dinv[2], M2, coefs[i0,i1,:,i3+1])

    for i0 in range(N0):
        for i1 in range(N1):
            for i2 in range(N2):
                coefs[i0,i1,i2,:] = find_coefs_1d(dinv[3], M3, coefs[i0,i1,i2,:])

    return coefs



def filter_data(dinv, data):
    if len(dinv) == 1:
        return filter_coeffs_1d(dinv,data)
    elif len(dinv) == 2:
        return filter_coeffs_2d(dinv,data)
    elif len(dinv) == 3:
        return filter_coeffs_3d(dinv,data)
    elif len(dinv) == 4:
        return filter_coeffs_4d(dinv,data)
