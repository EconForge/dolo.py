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

def serial_multiplication(A,B):

    I = A.shape[0]
    J = A.shape[1]
    N = A.shape[2]
    K = B.shape[1]

    assert(B.shape[0]==J)
    assert(B.shape[2]==N)

    resp = np.zeros( (I,K,N) )
    for i in range(I):
        for k in range(K):
            T = np.zeros( N )
            for j in range(J):
                T += A[i,j,:]*B[j,k,:] 
            resp[i,k,:] = T
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

def serial_dot(A,B):
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
    from dolo.numeric.serial_operations_c import serial_multiplication
except:
    pass


#try:
#import pyximport
#pyximport.install()
#from dolo.numeric.serial_operations_cython import serial_dot_21, serial_dot_11
#except:
#    pass

#
#try:
#    from dolo.numeric.serial_operations_c import serial_multiplication
#except:
#    pass


strange_tensor_multiplication_vector = serial_multiplication_vector
#strange_tensor_multiplication = serial_multiplication

################################################################################    

if __name__ == "__main__":
    
    def test(X):    
        ret = np.zeros_like(X)
        for n in range(X.shape[-1]):
            x = X[..., n]
            ret[..., n] = x**2
        return ret
    
    X0 = np.ones( (3, 2, 5) ) /2
    resp = numdiff1(test, X0)
    print resp.shape
    print resp[:, :, 1, 1, 0]

