import numpy as np

import numpy


# from dolo import *
import pickle

# should be moved to markov
from numba import jit, njit

@njit
def choice(x, n, cumul):
    i = 0
    running = True
    while i<n and running:
        if x < cumul[i]:
            running = False
        else:
            i += 1
    return i


@jit
def simulate_markov_chain(nodes, transitions, i_0, n_exp, horizon):

    n_states = nodes.shape[0]

#    start = numpy.array( (i_0,)*n_exp )
    simul = numpy.zeros( (horizon, n_exp), dtype=int)
    simul[0,:] = i_0
    rnd = numpy.random.rand(horizon* n_exp).reshape((horizon,n_exp))

    cumuls = transitions.cumsum(axis=1)
    cumuls = numpy.ascontiguousarray(cumuls)

    for t in range(horizon-1):
        for i in range(n_exp):
            s = simul[t,i]
            p = cumuls[s,:]
            simul[t+1,i] = choice(rnd[t,i], n_states, p)

    res = numpy.row_stack(simul)

    return res

class Process:
    pass

class IIDProcess(Process):
    pass

class DiscreteProcess(Process):
    pass

class ContinuousProcess(Process):
    pass

# class VAR1(ContinuousProcess):
#     pass

class DiscretizedProcess:
    pass

class DiscretizedIIDProcess(DiscretizedProcess):

    def __init__(self, nodes, weights):
        self.integration_nodes = nodes
        self.integration_weights = weights

    @property
    def grid(self):
        return None

    def n_nodes(self): # integer
        return 1

    def node(self, i:int): # vector
        return self.integration_nodes[0,:]*0

    def n_inodes(self, i:int): # integer
        return self.integration_nodes.shape[0]

    def inode(self, i:int, j:int): # vector
        return self.integration_nodes[j,:]

    def iweight(self, i:int, j:int): # scalar
        return self.integration_weights[j]


class MvNormal(IIDProcess):

    def __init__(self, sigma=None, orders=None, mu=None):

        self.sigma = np.atleast_2d( np.array(sigma, dtype=float) )
        self.d = len(self.sigma)
        if orders is None:
            self.orders = np.array([5]*self.d)
        else:
            self.orders = np.array(orders, dtype=int)
        if mu is None:
            self.mu = np.array([0.0]*self.d)
        else:
            self.mu = np.array(mu, dtype=float)
        assert(self.sigma.shape[0] == self.d)
        assert(self.sigma.shape[0] == self.d)
        assert(self.orders.shape[0] == self.d)

    def discretize(self, orders=None):
        if orders is None:
            orders = self.orders
        from dolo.numeric.discretization.quadrature import gauss_hermite_nodes
        [x, w] = gauss_hermite_nodes(orders, self.sigma, mu=self.mu)
        return DiscretizedIIDProcess(x,w)

    def simulate(self, N, T, m0=None, stochastic=True):
        from numpy.random import multivariate_normal
        sigma = self.sigma
        mu = self.mu
        if stochastic:
            sim = multivariate_normal(mu, sigma, N*T)
        else:
            sim = mu[None,len(mu)].repeat(T*N,axis=0)
        return sim.reshape((T,N,len(mu)))

    def response(self, T, impulse):
        from numpy.random import multivariate_normal
        sigma = self.sigma
        mu = self.mu
        irf = numpy.zeros((T,len(mu)))
        irf[1,:] = impulse[None,:]
        return irf



class DiscreteMarkovProcess(DiscretizedProcess):

    # transitions: 2d float array
    # values: 2d float array

    def __init__(self, transitions, values):

        self.transitions = np.array(transitions, dtype=np.float64)
        self.values = np.array(values, dtype=np.float64)

    def discretize(self):
        return self

    @property
    def grid(self):
        return self.values

    def n_nodes(self): # integer
        return self.values.shape[0]

    def node(self, i:int): # vector
        return self.values[i,:]

    def n_inodes(self, i:int): # integer
        return self.transitions.shape[1]

    def inode(self, i:int, j:int): # vector
        return self.values[j,:]

    def iweight(self, i:int, j:int): # scalar
        return self.transitions[i,j]

    def simulate(self, N, T, i0=0, stochastic=True):
        inds = simulate_markov_chain(self.values, self.transitions, i0, N, T)
        return self.values[inds]
#
# class MarkovChain(list):
#
#     def __init__(self, P=None, Q=None):
#
#         import numpy
#         P = numpy.array(P, dtype=float)
#         Q = numpy.array(Q, dtype=float)
#         self.states = P
#         self.transitions = Q
#         self.extend([P, Q]) # compatibility fix
# #
class VAR1(DiscreteMarkovProcess):

    def __init__(self, rho=None, sigma=None, N=2):

        self.rho = rho
        self.sigma = sigma
        self.N = N

    def discretize(self):

        N = self.N
        rho = self.rho
        sigma = self.sigma

        rho_array = np.array(rho, dtype=float)
        sigma_array = np.atleast_2d(np.array(sigma, dtype=float))
        try:
            assert(rho_array.ndim <= 1)
        except:
            raise Exception("When discretizing a Vector AR1 process, the autocorrelation coefficient must be as scalar. Found: {}".format(rho_array))
        try:
            assert(sigma_array.shape[0] == sigma_array.shape[1])
        except:
            raise Exception("The covariance matrix for a Vector AR1 process must be square. Found: {}".format())
        from dolo.numeric.discretization import multidimensional_discretization

        [P,Q] = multidimensional_discretization(rho_array, sigma_array)

        return DiscreteMarkovProcess(transitions=P, values=Q)
