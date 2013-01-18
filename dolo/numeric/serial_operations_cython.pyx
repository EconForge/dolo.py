import numpy as np
cimport numpy as np
from numpy.linalg import inv


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cimport cython



@cython.boundscheck(False)
@cython.wraparound(False)
def serial_multiplication(np.ndarray[DTYPE_t, ndim=3] A, np.ndarray[DTYPE_t, ndim=3] B):

    cdef int i,k,j,n,I,K,J,N
    I = A.shape[0]
    J = A.shape[1]
    N = A.shape[2]
    K = B.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=3] C = np.zeros( (I,K,N), dtype = np.float64 )

    cdef double* A_d = <double*> A.data
    cdef double* B_d = <double*> B.data
    cdef double* C_d = <double*> C.data

    cdef int rg, rm, rd

    for i in range(I):
        for k in range(K):
            for j in range(J):
                rg = i*N*K + k*N
                rm = i*N*J + j*N
                rd = j*N*K + k*N
                for n in range(N):
                    C_d[rg] += A_d[rm]*B_d[rd]
                    rg +=1
                    rm +=1
                    rd +=1

    return C




@cython.boundscheck(False)
@cython.wraparound(False)
def serial_dot_21(np.ndarray[DTYPE_t, ndim=3] A, np.ndarray[DTYPE_t, ndim=2] B):

    cdef int i,k,j,n,I,K,J,N
    I = A.shape[0]
    J = A.shape[1]
    N = A.shape[2]
#    K = B.shape[1]  : B.shape should be [J,N]

    #    assert(B.shape[0]==J)
    #    assert(B.shape[2]==N)

    cdef np.ndarray[DTYPE_t, ndim=2] resp
    #    cdef np.ndarray[DTYPE_t, ndim=1] T

    resp = np.zeros( (I,N), dtype = np.float64 )
    for i in range(I):
        for j in range(J):
            for n in range(N):
                resp[i,n] += A[i,j,n]*B[j,n]

    return resp

@cython.boundscheck(False)
@cython.wraparound(False)
def serial_dot_11(np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=2] B):

    cdef int i,k,j,n,I,K,J,N
    I = A.shape[0]
    N = A.shape[1]
    #    K = B.shape[1]  : B.shape should be [I,N]

    assert(B.shape[0]==I)
    assert(B.shape[1]==N)

    cdef np.ndarray[DTYPE_t, ndim=2] resp
    #    cdef np.ndarray[DTYPE_t, ndim=1] T

    resp = np.zeros( (I,N), dtype = np.float64 )
    for i in range(I):
        for n in range(N):
            resp[i,n] += A[i,n]*B[i,n]

    return resp


@cython.boundscheck(False)
@cython.wraparound(False)
def serial_inversion(np.ndarray[DTYPE_t, ndim=3] M):
    '''

    :param M: a pxpxN array
    :return: a pxpxN array T such that T(:,:,i) is the inverse of M(:,:,i)
    '''

    import numpy
    from numpy.linalg import inv

    cdef np.ndarray[DTYPE_t, ndim=3] T
    cdef np.ndarray[DTYPE_t, ndim=2] tmp
    cdef int i

    p = M.shape[0]
    assert(M.shape[1] == p)
    N = M.shape[2]

    M = np.asfortranarray(M)

    T = np.zeros((p,p,N),order='F')

    for i in range(N):
        tmp = M[:,:,i]
        T[:,:,i] = inv(tmp)

    T = numpy.ascontiguousarray(T)

    return T