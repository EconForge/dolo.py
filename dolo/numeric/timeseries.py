def asymptotic_variance( A, B, sigma, T=100 ):
    '''Computes asymptotic variance of the AR(1) process defined by:
    $X_t = A X_{t-1} + B \epsilon_t$
    where the $\epsilon_t$ follows a random law with covariance sigma.
    '''
    import numpy
    p = A.shape[0] # number of variables
    Q = numpy.zeros( (p,p) )
    for i in range(T):
        Q = numpy.dot( A, numpy.dot( Q, A.T)) + numpy.dot( B, numpy.dot( sigma, B.T))
    return Q