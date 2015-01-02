from itertools import product

import numpy as np

def numdiff1(f, x0, dv=1e-8):
    '''Returns the derivative of f w.r.t. to multidimensional vector x0
If x0 is of dimension R1 x ... x Rd x Rn dimension of f is assumed to be
in the form S1 x ... x Sf x Rn. The last dimension corresponds to various
observations. The value returned is of dimension :
S1 x ... x Sf x R1 x ... x Rd x Rn
    '''
    in_shape = x0.shape
    nobs = in_shape[-1]
    dd = in_shape[:-1]
    f0 = f(x0)
    assert(f0.shape[-1] == nobs)
    f_shape = f0.shape[:-1]

    out_shape = f_shape + dd + (nobs,)
    ret = np.zeros(out_shape)

    for ind in product( *[range(i) for i in dd] ):
        sl = ind + (slice(None, None, None), )
        x = x0.copy()
        x[sl] += dv
        x2 = x0.copy()
        x2[sl] -= dv
        df = (f(x) - f(x2))/dv/2.0
        obj = [ Ellipsis] +  list(ind) + [slice(None, None, None)]
        obj = tuple(obj)
        ret[obj] = df

    return ret


def numdiff2(f, x0, dv=1e-8):
    '''Returns the derivative of f w.r.t. to multidimensional vector x0
If x0 is of dimension R1 x ... x Rd dimension of f is assumed to be
in the form S1 x ... x Sf x Rn. The last dimension corresponds to various
observations. The value returned is of dimension :
S1 x ... x Sf x R1 x ... x Rd x Rn
    '''

    dd = x0.shape
    f0 = f(x0)
    nobs = f0.shape[-1]
    f_shape = f0.shape[:-1]

    out_shape = f_shape + dd + (nobs,)
    ret = np.zeros(out_shape)


    for ind in product( *[range(i) for i in dd] ):
        x = x0.copy()
        x[ind] += dv
        x2 = x0.copy()
        x2[ind] -= dv
        df = (f(x) - f(x2))/dv/2.0
        obj = [ Ellipsis] +  list(ind) + [slice(None, None, None)]
        #obj = tuple(obj)
        ret[obj] = df

    return ret

import numpy
from numba import jit

@jit
def serial_multiplication(A,B):

    if A.ndim == 2 and B.ndim == 2:
        return numpy.dot(A,B)

    I = A.shape[1]
    J = A.shape[2]
    N = A.shape[0]
    K = B.shape[2]

    assert(B.shape[1]==J)
    assert(B.shape[0]==N)

    v = 0.0
    resp = np.zeros( (N,I,K) )
    for n in range(N):
        for i in range(I):
            for k in range(K):
                v = 0
                for j in range(J):
                    v += A[n,i,j]*B[n,j,k]
                resp[n,i,k] = v

    return resp


# def serial_multiplication_vector(A,X):
#
#     I = A.shape[0]
#     J = A.shape[1]
#     N = A.shape[2]
#
#     assert(X.shape[0]==J)
#     assert(X.shape[1]==N)
#
#     resp = np.zeros( (I,N) )
#     for i in range(I):
# #        T = np.zeros( N )
#         for j in range(J):
# #            T += A[i,j,:]*B[j,k,:]
#             resp[i,:] += A[i,j,:]*X[j,:]
#     return resp


if __name__ == "__main__":


    import numpy.random
    A = numpy.random.random((I,J, N))
    B = numpy.random.random((J,K, N))

    import time
    r = time.time()
    C0 = serial_multiplication(A,B)
    s = time.time()
    t = time.time()
    C2 = smult(A,B)
    u = time.time()


    AA = numpy.rollaxis(A,2).copy()
    BB = numpy.rollaxis(B,2).copy()

    x = time.time()
    CC = numpy.zeros( (A.shape[2], A.shape[0], B.shape[1]))
    serial_mult_numba(AA,BB,CC)
    y = time.time()

    print('Py : {}'.format(s-r))
    print('Cython : {}'.format(u-t))
    print('Numba : {}'.format(y-x))

    print(abs(C2-C0).max())
