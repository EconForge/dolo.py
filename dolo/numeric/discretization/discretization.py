"""
Discretization of continuous processes as markov chain
"""


import scipy as sp
import scipy.stats
import numpy as np
import math

# univariate tauchen code is taken from dpsolve ( https://bitbucket.org/stevejb/dpsolve )

def tauchen(N, mu, rho, sigma, m=2):
    """
    Approximate an AR1 process by a finite markov chain using Tauchen's method.

    :param N: scalar, number of nodes for Z
    :param mu: scalar, unconditional mean of process
    :param rho: scalar
    :param sigma: scalar, std. dev. of epsilons
    :param m: max +- std. devs.
    :returns: Z, N*1 vector, nodes for Z. Zprob, N*N matrix, transition probabilities

    SJB: This is a port of Martin Floden's 1996 Matlab code to implement Tauchen 1986 Economic Letters method The following comments are Floden's. Finds a Markov chain whose sample paths approximate those of the AR(1) process z(t+1) = (1-rho)*mu + rho * z(t) + eps(t+1) where eps are normal with stddev sigma.
    """
    Z     = np.zeros((N,1))
    Zprob = np.zeros((N,N))
    a     = (1-rho)*mu

    Z[-1]  = m * math.sqrt(sigma**2 / (1 - (rho**2)))
    Z[0]  = -1 * Z[-1]
    zstep = (Z[-1] - Z[0]) / (N - 1)

    for i in range(1,N):
        Z[i] = Z[0] + zstep * (i)

    Z = Z + a / (1-rho)

    for j in range(0,N):
        for k in range(0,N):
            if k == 0:
                Zprob[j,k] = sp.stats.norm.cdf((Z[0] - a - rho * Z[j] + zstep / 2) / sigma)
            elif k == (N-1):
                Zprob[j,k] = 1 - sp.stats.norm.cdf((Z[-1] - a - rho * Z[j] - zstep / 2) / sigma)
            else:
                up = sp.stats.norm.cdf((Z[k] - a - rho * Z[j] + zstep / 2) / sigma)
                down = sp.stats.norm.cdf( (Z[k] - a - rho * Z[j] - zstep / 2) / sigma)
                Zprob[j,k] = up - down

    return( (Z, Zprob) )


def rouwenhorst(rho, sigma, N):
    """
    Approximate an AR1 process by a finite markov chain using Rouwenhorst's method.

    :param rho: autocorrelation of the AR1 process
    :param sigma: conditional standard deviation of the AR1 process
    :param N: number of states
    :return [nodes, P]: equally spaced nodes and transition matrix
    """

    from numpy import sqrt, linspace, array,zeros

    sigma = float(sigma)

    if N == 1:
      nodes = array([0.0])
      transitions = array([[1.0]])
      return [nodes, transitions]

    p = (rho+1)/2
    q = p
    nu = sqrt( (N-1)/(1-rho**2) )*sigma

    nodes = linspace( -nu, nu, N)
    sig_a = sigma
    n = 1
    #    mat0 = array( [[1]] )
    mat0 = array([[p,1-p],[1-q,q]])
    if N == 2:
        return [nodes,mat0]
    for n in range(3,N+1):
        mat = zeros( (n,n) )
        mat_A = mat.copy()
        mat_B = mat.copy()
        mat_C = mat.copy()
        mat_D = mat.copy()
        mat_A[:-1,:-1] = mat0
        mat_B[:-1,1:] = mat0
        mat_C[1:,:-1] = mat0
        mat_D[1:,1:] = mat0

        mat0 = p*mat_A + (1-p)*mat_B + (1-q)*mat_C + q*mat_D
        mat0[1:-1,:] = mat0[1:-1,:]/2
    P = mat0
    return [nodes, P]



def multidimensional_discretization(rho, sigma, N=2, method='rouwenhorst', m=2):
    """
    Discretize an VAR(1) into a markov chain. The autoregression matrix is supposed to be a scalar.

    :param rho:
    :param sigma:
    :param N:
    :param method:
    :param m:
    :return:
    """

    # rho is assumed to be a scalar
    # sigma is a positive symmetric matrix
    # N number of points in each non-degenerate dimension
    # m : standard deviations to approximate

    import scipy.linalg
    from itertools import product

    d = sigma.shape[1]

    sigma = sigma.copy()

    zero_columns = np.where(sigma.sum(axis=0)==0)[0]
    for i in zero_columns:
        sigma[i,i] = 1

    L = scipy.linalg.cholesky(sigma)

    N = int(N)

    if method=='tauchen':
        [nodes_1d, probas_1d] = tauchen(N, 0, rho, 1, m=m)
    elif method=='rouwenhorst':
        [nodes_1d, probas_1d] = rouwenhorst(rho, 1, N)

    markov_nodes = np.array( list( product( *([nodes_1d]*d))) ).T
    markov_indices = np.array( list( product( *([range(N)]*d)  )  ) ).T

    markov_nodes = np.dot(L, markov_nodes)


    transition_matrix = 1
    for i in range(d):
        transition_matrix = np.kron(transition_matrix, probas_1d)

    markov_nodes = np.ascontiguousarray(markov_nodes.T)

    for i in zero_columns:
        markov_nodes[:,i] = 0

    return [markov_nodes, transition_matrix]

def tensor_markov( *args ):
    """Computes the product of two independent markov chains.

    :param m1: a tuple containing the nodes and the transition matrix of the first chain
    :param m2: a tuple containing the nodes and the transition matrix of the second chain
    :return: a tuple containing the nodes and the transition matrix of the product chain
    """
    if len(args) > 2:

        m1 = args[0]
        m2 = args[1]
        tail = args[2:]
        prod = tensor_markov(m1,m2)
        return tensor_markov( prod, tail )

    elif len(args) == 2:

        m1,m2 = args
        n1, t1 = m1
        n2, t2 = m2

        n1 = np.array(n1, dtype=float)
        n2 = np.array(n2, dtype=float)
        t1 = np.array(t1, dtype=float)
        t2 = np.array(t2, dtype=float)

        assert(n1.shape[0] == t1.shape[0] == t1.shape[1])
        assert(n2.shape[0] == t2.shape[0] == t2.shape[1])

        t = np.kron(t1, t2)

        p = t1.shape[0]
        q = t2.shape[0]

        np.tile( n2, (1,p))
        # n = np.row_stack([
        #     np.repeat(n1, q, axis=1),
        #     np.tile( n2, (1,p))
        # ])
        n = np.column_stack([
            np.repeat(n1, q, axis=0),
            np.tile( n2, (p,1))
        ])
        return [n,t]

    else:
        raise Exception("Incorrect number of arguments. Expected at least 2. Found {}.".format(len(args)))




#
# quantization_data = '/home/pablo/quantization_grids/'
#
#
# def quantization_nodes(N,sigma):
#     import numpy
#     import numpy.linalg
#     assert( len(sigma.shape) == 2 )
#     var = numpy.diag(sigma)
#     d = sigma.shape[0]
#     [w, x] = standard_quantization_weights(N,d )
#     A = numpy.linalg.cholesky(sigma)
#     x = numpy.dot(A, x)
#     return [x,w]
#
#
# def quantization_weights(N,sigma):
#     [x,w] = quantization_nodes(N,sigma)
#     return [w,x]
#
# def standard_quantization_weights(N,d):
#     filename = quantization_data + '{0}_{1}_nopti'.format(N,d)
#     import numpy
#
#     try:
#         G = numpy.loadtxt(filename)
#     except Exception as e:
#         raise e
#
#     w = G[:N,0]
#     x = G[:N,1:d+1]
#
#     s = numpy.dot(w, x)
#     x = x - numpy.outer(w,s)
#
#     return [w,x.T]

if __name__ == '__main__':

    [Z,Zprob] = tauchen( 5,0,0.8,0.1,1.5 )

    import numpy
    sigma = numpy.diag([0.1, 0.1])**2
    sigma[0,1]  = numpy.sqrt( 0.5*sigma[0,0]*sigma[1,1] )
    sigma[1,0]  = numpy.sqrt( 0.5*sigma[0,0]*sigma[1,1] )
    rho = 0.9

    [nodes, transition] = multidimensional_discretization(rho, sigma, 2, method='rouwenhorst')

    transition0 = transition.copy()*0
    transition0[1,1] = 1.0

    print(nodes.shape)

    print(transition.shape)

    [nodes, transition] = tensor_markov( (nodes, transition0), (nodes, transition) )

    print(nodes.shape)
    print(transition.shape)
