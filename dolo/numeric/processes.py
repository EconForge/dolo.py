import numpy as np
import numpy
import pickle
from typing import List, Optional
from numba import jit, njit
from dolo.numeric.grids import EmptyGrid, CartesianGrid, UnstructuredGrid

from dolo.compiler.language import greek_tolerance
from dolo.compiler.language import language_element

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

    def response(self, T, impulse):
        irf = numpy.zeros((T,self.d))
        irf[1,:] = impulse[None,:]
        return irf

class DiscreteProcess(Process):
    pass

class ContinuousProcess(Process):
    pass


@language_element
class Product:

    def __init__(self, *l):
        self.processes = l

    def discretize(self, to='mc', options={}):

        if isinstance(options, dict):
            kwargs = [options]*len(self.processes)
        else:
            assert(len(options)==len(self.processes))
        if to=='mc':
            return MarkovProduct(*[e.discretize(to='mc', **kwargs[i]) for i,e in enumerate(self.processes)]).discretize(to='mc')
        else:
            raise Exception("Not implemented.")



@language_element
class ConstantProcess(Process):

    signature = {'μ': 'Vector'}

    @greek_tolerance
    def __init__(self, μ=None):

        if isinstance(μ, (float, int)):
             μ = [μ]
        self.μ = np.array(μ)
        assert(self.μ.ndim==1)
        self.d = len(self.μ)

    def discretize(self, to='mc'):

        if to!='mc':
            raise Exception("Not implemented")
        else:
            nodes = self.μ[None,:]
            transitions = np.array([[1.0]])
            return DiscreteMarkovProcess(transitions, nodes)

@language_element
class AggregateProcess(ConstantProcess):

    # that is a dummy class
    pass
# class VAR1(ContinuousProcess):
#     pass

class DiscretizedProcess:

    def iteritems(self, i, eps=1e-16):
        for j in range(self.n_inodes(i)):
            w = self.iweight(i,j)
            if w>eps:
                x = self.inode(i,j)
                yield (w,x)


class GDP(DiscretizedProcess):

    def __init__(self, nodes, inodes, iweights, grid=None):
        self.__nodes__ = nodes
        self.__inodes__ = inodes
        self.__iweights__= iweights
        self.__grid__=grid

    #def discretize_gdp(self):
    #    return self
    @property
    def grid(self):
        return self.__grid__

    def n_nodes(self)->int:
        return self.__nodes__.shape[0]

    def node(self, i: int): #->List:
        return self.__nodes__[i,:]

    def nodes(self):
        return self.__nodes__

    def n_inodes(self, i: int): #->int:
        return self.__inodes__.shape[1]

    def inode(self, i, j): #->List:
        return self.__inodes__[i,j]

    def iweight(self, i, j): #->float:
        return self.__iweights__[i,j]


class DiscretizedIIDProcess(DiscretizedProcess):

    def __init__(self, nodes, weights):
        self.integration_nodes = nodes
        self.integration_weights = weights

    @property
    def grid(self):
        return EmptyGrid()

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


@language_element
class MvNormal(IIDProcess):

    signature = {"Σ": 'Matrix', 'μ':  'Optional[Vector]'}

    @greek_tolerance
    def __init__(self, Σ=None, μ=None):

        Sigma = Σ
        mu = μ

        self.Σ = np.atleast_2d( np.array(Sigma, dtype=float) )
        self.d = len(self.Σ)
        if mu is None:
            self.μ = np.array([0.0]*self.d)
        else:
            self.μ = np.array(mu, dtype=float)
        assert(self.Σ.shape[0] == self.d)
        assert(self.Σ.shape[0] == self.d)

    def discretize(self, to='iid', N=None):
        if to!="iid":
            raise Exception("Not Implemented")
        if N is None:
            N  = 5
        if isinstance(N,int):
            N = [N]*self.d
        from dolo.numeric.discretization.quadrature import gauss_hermite_nodes
        [x, w] = gauss_hermite_nodes(N, self.Σ, mu=self.μ)
        x = np.row_stack( [(e + self.μ) for e in x] )
        return DiscretizedIIDProcess(x,w)

    def simulate(self, N, T, m0=None, stochastic=True):
        from numpy.random import multivariate_normal
        Sigma = self.Σ
        mu = self.μ
        if stochastic:
            sim = multivariate_normal(mu, Sigma, N*T)
        else:
            sim = mu[None,len(mu)].repeat(T*N,axis=0)
        return sim.reshape((T,N,len(mu)))

@language_element
class Normal(MvNormal):
    pass

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
        return UnstructuredGrid(self.values)

    def n_nodes(self): # integer
        return self.values.shape[0]

    def node(self, i:int): # vector
        return self.values[i,:]

    def nodes(self):
        return self.values

    def n_inodes(self, i:int): # integer
        return self.transitions.shape[1]

    def inode(self, i:int, j:int): # vector
        return self.values[j,:]

    def iweight(self, i:int, j:int): # scalar
        return self.transitions[i,j]

    def simulate(self, N, T, i0=0, stochastic=True):
        if stochastic:
            inds = simulate_markov_chain(self.values, self.transitions, i0, N, T)
        else:
            inds = np.zeros((T,N), dtype=int) + i0
        return self.values[inds]


@language_element
class MarkovChain(DiscreteMarkovProcess):

    signature = {'transitions': 'Matrix', 'values': 'Matrix'}

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

class MarkovProduct(DiscreteMarkovProcess):

    def __init__(self, *args):

        self.M = args

    def discretize(self, to='mc'):

        M = [(m.values, m.transitions) for m in self.M]
        from dolo.numeric.discretization import tensor_markov
        [P, Q] = tensor_markov( *M )
        return DiscreteMarkovProcess(Q,P)

class GDPProduct(DiscreteMarkovProcess):

    def __init__(self, *args):

        self.M = args

    def discretize(self, to='gdp'):

        pass


@language_element
class VAR1(ContinuousProcess):

    signature = {"ρ": 'float','Σ': 'Matrix', 'μ': 'Optional[Vector]'}

    @greek_tolerance
    def __init__(self, ρ=None, Σ=None, μ=None):

        rho = ρ
        Sigma = Σ
        mu = μ

        self.Sigma = np.atleast_2d(Sigma)
        d = self.Sigma.shape[0]
        rho = np.array(rho)
        if rho.ndim == 0:
            self.rho = np.eye(d)*rho
        elif rho.ndim ==1:
            self.rho = np.diag(rho)
        else:
            self.rho = rho
        if mu is None:
            self.mu = np.zeros(d)
        else:
            self.mu = np.array(mu, dtype=float)
        self.d = d

    def discretize(self, N=3, to='mc', **kwargs):
        if to=='mc':
            return self.discretize_mc(N=N, **kwargs)
        elif to=='gdp':
            return self.discretize_gdp(N=N, **kwargs)

    def discretize_mc(self, N=3):

        rho = self.rho
        Sigma = self.Sigma

        try:
            assert(abs(np.eye(rho.shape[0])*rho[0,0]-rho).max() <= 1)
        except:
            raise Exception("When discretizing a Vector AR1 process, the autocorrelation coefficient must be as scalar. Found: {}".format(rho_array))

        from dolo.numeric.discretization import multidimensional_discretization

        [P,Q] = multidimensional_discretization(rho[0,0], Sigma, N=N)

        P += self.mu[None,:]

        return DiscreteMarkovProcess(values=P, transitions=Q)

    def discretize_gdp(self, N=3):

        Σ = self.Sigma
        ρ = self.rho

        n_nodes = N
        n_std = 2.5
        n_integration_nodes = 5

        try:
            assert(Σ.shape[0]==1)
        except:
            raise Exception("Not implemented.")

        try:
            assert(ρ.shape[0]==ρ.shape[1]==1)
        except:
            raise Exception("Not implemented.")

        ρ = ρ[0,0]
        σ = np.sqrt(Σ[0,0])


        from dolo.numeric.discretization import gauss_hermite_nodes

        epsilons, weights = gauss_hermite_nodes([n_integration_nodes], Σ)

        min = -n_std*(σ/(np.sqrt(1-ρ**2)))
        max = n_std*(σ/(np.sqrt(1-ρ**2)))

        from .grids import CartesianGrid
        grid = CartesianGrid([min],[max],[n_nodes])

        nodes = np.linspace(min,max,n_nodes)[:,None]
        iweights = weights[None,:].repeat(n_nodes,axis=0)
        integration_nodes = np.zeros((n_nodes, n_integration_nodes))[:,:,None]
        for i in range(n_nodes):
            for j in range(n_integration_nodes):
                integration_nodes[i,j,:] =  ρ*nodes[i,:] + epsilons[j]

        return GDP(nodes,integration_nodes,iweights,grid=grid)
        #return (nodes,integration_nodes,iweights)


    def simulate(self, N, T, m0=None, stochastic=True):

        d = self.d

        if m0 is None:
            m0 = np.zeros(d)
        from numpy.random import multivariate_normal
        Sigma = self.Sigma
        mu = self.mu*0
        if stochastic:
            innov = multivariate_normal(mu, Sigma, N*T)
        else:
            innov = mu[None,len(mu)].repeat(T*N,axis=0)
        innov = innov.reshape((T,N,d))
        rho = self.rho
        sim = np.zeros((T,N,d))
        sim[0,:,:] = m0[None,:]
        for t in range(1,sim.shape[0]):
            sim[t,:,:] = sim[t-1,:,:]@rho.T + innov[t,:,:]

        sim += self.mu[None,None,:]

        return sim

    def response(self, T, impulse):
        d = self.d
        irf = np.zeros((T,d))
        irf[0,:] = impulse
        rho = self.rho
        for t in range(1,irf.shape[0]):
            irf[t,:] = rho@irf[t-1,:]
        irf += self.mu[None,:]
        return irf
