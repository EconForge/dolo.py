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

def serial_multiplication(A,B):

    if A.ndim == 2 and B.ndim == 2:
        return numpy.dot(A,B)

    I = A.shape[1]
    J = A.shape[2]
    N = A.shape[0]
    K = B.shape[2]

    assert(B.shape[1]==J)
    assert(B.shape[0]==N)

    resp = np.zeros( (N,I,K) )
    for i in range(I):
        for k in range(K):
            T = np.zeros( N )
            for j in range(J):
                T += A[:,i,j]*B[:,j,k]
            resp[:,i,k] = T
    return resp


def serial_multiplication_vector(A,X):

    I = A.shape[0]
    J = A.shape[1]
    N = A.shape[2]

    assert(X.shape[0]==J)
    assert(X.shape[1]==N)

    resp = np.zeros( (I,N) )
    for i in range(I):
#        T = np.zeros( N )
        for j in range(J):
#            T += A[i,j,:]*B[j,k,:]
            resp[i,:] += A[i,j,:]*X[j,:]
    return resp

strange_tensor_multiplication = serial_multiplication

def serial_dot_3_2(A,X):

    I = A.shape[0]
    J = A.shape[1]
    N = A.shape[2]

    assert(X.shape[0]==J)
    assert(X.shape[1]==N)

    resp = np.zeros( (I,N) )
    for i in range(I):
    #        T = np.zeros( N )
        for j in range(J):
        #            T += A[i,j,:]*B[j,k,:]
            resp[i,:] += A[i,j,:]*X[j,:]
    return resp

def serial_dot_2_2(A,B):
    # scalar_products

    resp = (A*B).sum(axis=0)

    return resp

def serial_dot(A,B):

    if A.ndim == 3 and B.ndim == 2:
        return serial_dot_3_2(A,B)

    if A.ndim == 2 and B.ndim == 2:
            return serial_dot_2_2(A,B)

    nobs = A.shape[-1]
    test = np.dot( A[...,0], B[...,0] )
    sh = test.shape + (nobs,)

    resp = np.zeros( sh ) #empty?
    for i in range(nobs):
        resp[...,i] = np.dot( A[...,i], B[...,i] )
    return resp


def serial_solve(M,Y, debug=False):
    '''
    :param M: a pxpxN array
    :param Y: a pxN array
    :return X: a pxN array X such that M(:,:,i)*X(:,i) = Y(:,:,i)
    '''


    debug = True
    ## surprisingly it is slower than inverting M and doing a serial multiplication !

    import numpy
    from numpy.linalg import solve


    p = M.shape[0]
    assert(M.shape[1] == p)
    N = M.shape[2]
    assert(Y.shape == (p,N) )

    X = numpy.zeros((p,N))

    if not debug:
        for i in range(N):
            X[:,i] = solve(M[:,:,i],Y[:,i])
    else:
        for i in range(N):
            try:
                X[:,i] = solve(M[:,:,i],Y[:,i])
            except Exception as e:
                print('Derivative {}'.format(i))
                print(M[:,:,i])
                raise Exception('Error while solving point {}'.format(i))

    return X

def serial_inversion(M):
    '''

    :param M: a pxpxN array
    :return: a pxpxN array T such that T(:,:,i) is the inverse of M(:,:,i)
    '''

    import numpy
    from numpy.linalg import inv

#    MM = numpy.ascontiguousarray(M.swapaxes(0,2))

    p = M.shape[0]
    assert(M.shape[1] == p)
    N = M.shape[2]
    T = numpy.zeros((p,p,N))


    for i in range(N):
        T[:,:,i] = inv(M[:,:,i])

    return T



try:
    from dolo.numeric.serial_operations_cython import serial_multiplication as smult
except:
    pass

strange_tensor_multiplication_vector = serial_multiplication_vector
#strange_tensor_multiplication = serial_multiplication

################################################################################    

# from numbapro import guvectorize, void, f8, f4
#
# @guvectorize([ void(f4[:,:], f4[:,:], f4[:,:]), void(f8[:,:], f8[:,:], f8[:,:])], '(I,J),(J,K)->(I,K)', backend='ast')#, target='gpu')
# def serial_mult_numba(A,B,C):
#     m, n = A.shape
#     n, p = B.shape
#     for i in range(m):
#         for j in range(p):
#             C[i, j] = 0
#             for k in range(n):
#                 C[i, j] += A[i, k] * B[k, j]
#
#
# from numpy.linalg import inv
# @guvectorize([ void(f4[:,:], f4[:,:]), void(f8[:,:], f8[:,:])], '(I,I)->(I,I)', backend='ast')#, target='gpu')
# def serial_solve_numba(A,B):
#     m, m = A.shape
#     m, m = B.shape
#     B[...] = inv(A)



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

