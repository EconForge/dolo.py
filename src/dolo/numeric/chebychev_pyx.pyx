import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float_t DTYPE_t

def cheb_extrema(n):
    jj = np.arange(1.0,n+1.0)
    zeta =  np.cos( np.pi * (jj-1) / (n-1 ) )
    return zeta

@cython.boundscheck(False)
def chebychev(np.ndarray[DTYPE_t, ndim=2] x, int N):
    # computes the chebychev polynomials of the first kind
    cdef int K = x.shape[0]
    cdef int L = x.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=3] results
    results = np.zeros((N+1,K,L))
    results[0,...] = np.ones((K,L))
    results[1,...] = x
    cdef unsigned int i
    cdef unsigned int k
    cdef unsigned int l
    for i in range(2,N+1):
        for k in range(K):
            for l in range(L):
                results[i,k,l] = 2 * x[k,l] * results[i-1,k,l] - results[i-2,k,l]
    return results

@cython.boundscheck(False)
def chebychev2(np.ndarray[DTYPE_t, ndim=2] x, int N):
    # computes the chebychev polynomials of the first kind
    cdef int K = x.shape[0]
    cdef int L = x.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=3] results
    results = np.zeros((N+1,K,L))
    results[0,...] = np.ones((K,L))
    results[1,...] = 2*x
    cdef unsigned int i
    cdef unsigned int k
    cdef unsigned int l
    for i in range(2,N+1):
        for k in range(K):
            for l in range(L):
                results[i,k,l] = 2 * x[k,l] * results[i-1,k,l] - results[i-2,k,l]
    return results