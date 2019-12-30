from dolo.compiler.language import greek_tolerance, language_element


## Useful Links

## Common probability distributions:
## https://blog.cloudera.com/blog/2015/12/common-probability-distributions-the-data-scientists-crib-sheet/
##
## Distributions.jl:(args, kwargs))
## https://juliastats.github.io/Distributions.jl/stable/
##
## Scipy.stats:
## https://docs.scipy.org/doc/scipy-0.17.1/reference/stats.html
##
## Quantecon/rvlib list:
## https://github.com/QuantEcon/rvlib/tree/multivariate
##
## Quantecon/rvlib univarite:
## https://github.com/QuantEcon/rvlib/blob/multivariate/rvlib/univariate.py
##
## Hark/utilities.py: (all univariate?)
## https://github.com/econ-ark/HARK/blob/d99393973554b1cf830c6285e6da59d98ff242ff/HARK/utilities.py
##
## Dolo processes.py:
## https://github.com/EconForge/dolo/blob/master/dolo/numeric/processes.py
##
## Dolo processes.jl:
## https://github.com/EconForge/Dolo.jl/blob/master/src/numeric/processes.jl


## This code

# Here we have the list of classes implemented by Rvlib
  # They only have Mv normal in multivariate (add at least log-normal)

# Do we sepearte mv and univariate?
  # If not create a dict with small and capital letters denoting uni and mv cases
  # Then accept both and convert automatically to do operations for mv case ?
  # For which cases we have mv?

# Also implement greek tolerance
  # (draft code below)

# If parameters not defined, do we define what to take as default?
  # example:log-normal --> if not parameters, mean-one log-normal

# Parameter names are (so far) used as they appear in Distributions.jl
  # This seems like the richest source for distributions with very clear documentation



from scipy.stats import norm, uniform, lognorm, beta
from matplotlib import pyplot as plt
import numpy as np
from dataclasses import dataclass

from dolo.numeric.processes import Process, IIDProcess, DiscretizedProcess, DiscretizedIIDProcess


class UnivariateIIDProcess(IIDProcess):
    d = 1

    def discretize(self,  N=5, method='equiprobable', mass_point = "median" ,to='iid'):
        if to !='iid':
            raise Exception("Not implemented")

        if method=='gauss-hermite':
            return self.__discretize_gh__(N=N)
        elif method=='equiprobable':
            return self.__discretize_ep__(N=N, mass_point=mass_point)
        else:
            raise Exception("Unknown discretization method.")

    def __discretize_ep__(self, N=5, mass_point="median"): #Equiprobable
        if mass_point == "median":
            p = np.linspace(0.5/N,1-0.5/N,N)
            q = self.ppf(p)
        elif mass_point == "left":
            p = np.linspace(0,1-1/N,N)
            q = self.ppf(p)
        elif mass_point == "middle":
            p = np.linspace(0.,1,N+1)
            q = self.ppf(p)
            q = 0.5*(q[1:]+q[:-1])
        elif mass_point == "right":
            p = np.linspace(1/N,1,N)
            q = self.ppf(p)
        else:
            raise Exception("Not implemented")

        w = (1/(N))*np.ones(N)

        return DiscretizedIIDProcess(q[:,None], w)

@language_element
@dataclass
class Bernouilli(UnivariateIIDProcess):

    π: float=0.5

    signature = {'π': 'float'} # this is redundant for now

    @greek_tolerance
    def __init__(self, π:float=None):
        self.π = float(π)

    def discretize(self, to='iid'):
        if to !='iid':
            raise Exception("Not implemented")
        x = np.array([[0],[1]])
        w = np.array([1-self.π, self.π])
        return DiscretizedIIDProcess(x, w)

    def simulate(self, N, T, stochastic=True):

        from numpy.random import choice
        ch = np.array([0, 1])
        p = np.array([1-self.π, self.π])
        sim = choice(ch, size=T*N, p=p)
        return sim.reshape((T,N,1))


class IIDMixture(IIDProcess):

    def __init__(self, index, distributions):
        # index is a distribution which takes discrete values
        # distributions is a map from each of these values to a distribution
        self.index = index
        self.distributions = distributions
        ds = [e.d for e in self.distributions.values()]
        assert(len(set(ds))==1)
        self.d = self.distributions[0].d

    def discretize(self, to='iid'):

        if to !='iid':
            raise Exception("Not implemented")

        inddist = self.index.discretize()
        nodes = []
        weights = []
        for i in range(inddist.n_inodes(0)):
            wind = inddist.iweight(0,i)
            xind = inddist.inode(0,i)
            dist = self.distributions[int(xind)].discretize(to='iid')
            for j in range(dist.n_inodes(0)):
                w = dist.iweight(0,j)
                x = dist.inode(0,j)
                nodes.append(x)
                weights.append(wind*w)
        nodes = np.concatenate([e[None,:] for e in nodes], axis=0)
        weights = np.array(weights)
        return DiscretizedIIDProcess(nodes, weights)

    def simulate(self, N, T):

        # stupid approach
        choices = self.index.simulate(N,T)
        draws = [dist.simulate(N,T) for dist in self.distributions.values()]
        draw = sum([(i==choices)*draws[i] for i in range(len(draws))])
        return draw



@language_element
def Mixture(index=None, distributions=None):

    for dist in distributions.values():
        if not (isinstance(dist, IIDProcess)):
            raise Exception("Only mixtures of iid processes are supported so far.")
    return IIDMixture(index, distributions)
    # not clear what we might do with non-iid

Mixture.signature = {'index': 'intprocess', 'distributions': 'Dict[int,IIDProcesses]'}


@language_element
@dataclass
class UNormal(UnivariateIIDProcess):

    μ: float=0.0
    σ: float=1.0
    signature = {'μ': 'Optional[float]', 'σ': 'float'} # this is redundant for now

    @greek_tolerance
    def __init__(self, σ:float=None, μ:float=None):
        self.σ = float(σ)
        self.μ = 0.0 if μ is None else float(μ)

    def ppf(self, quantiles):
        x = norm.ppf(quantiles, loc=self.μ, scale=self.σ)
        return x

    def discretize(self, N=5, method='gauss-hermite', to='iid'):

        if to !='iid':
            raise Exception("Not implemented")

        if method=='gauss-hermite':
            return self.__discretize_gh__(N=N)
        elif method=='equiprobable':
            return self.__discretize_ep__(N=N)
        else:
            raise Exception("Unknown discretization method.")


    def __discretize_gh__(self, N=5): # Gauss-Hermite

        from dolo.numeric.discretization.quadrature import gauss_hermite_nodes
        [x, w] = gauss_hermite_nodes(N, np.array([[self.σ**2]]), mu=self.μ)
        x = np.row_stack( [(e + self.μ) for e in x] )
        return DiscretizedIIDProcess(x, w)

    def simulate(self, N, T, m0=None, stochastic=True):

        from numpy.random import multivariate_normal
        Sigma = np.array([[self.σ**2]])
        mu = np.array([self.μ])
        if stochastic:
            sim = multivariate_normal(mu, Sigma, N*T)
        else:
            sim = mu[None,len(mu)].repeat(T*N,axis=0)
        return sim.reshape((T,N,len(mu)))

@language_element
@dataclass
class Uniform(UnivariateIIDProcess):

    # uniform distribution over an interval [a,b]
    a: float=0.0
    b: float=1.0
    signature = {'a': 'float', 'b': 'float'}

    def __init__(self, a:float=0.0, b:float=1.0):
        self.a = float(a)
        self.b = float(b)

    def ppf(self, quantiles):
        x = uniform.ppf(quantiles, loc=self.a, scale=(self.b-self.a))
        return x


    def simulate(self, N, T, m0=None, stochastic=True):
        from numpy.random import uniform
        mu = np.array([self.a+(self.b-self.a)/2])
        if stochastic:
            sim = uniform(self.a, self.b, N*T)
        else:
            sim = mu[None,len(mu)].repeat(T*N,axis=0)
        return sim.reshape((T,N,len(mu)))

        # method: In hark utilities
          #reference https://github.com/econ-ark/HARK/blob/d99393973554b1cf830c6285e6da59d98ff242ff/HARK/utilities.py

        #Returns
          #inodes ("X" in hark)
        #Discrete points for discrete probability mass function.
          #weights ("pmf" in Hark : Discrete points for discrete probability mass function.)
        #Probability associated with each point in grid (nodes)

@language_element
@dataclass
class LogNormal(UnivariateIIDProcess):



    # parametrization a lognormal random variable Y is in terms of
    # the mean, μ, and standard deviation, σ, of the unique normally distributed random variable X
    # such that exp(X) = Y.

    μ: float=0.0
    σ: float=1.0

    signature = {'μ': 'Optional[float]', 'σ': 'float'} # this is redundant for now

    @greek_tolerance
    def __init__(self, σ:float=0.0, μ:float=1.0):
        self.σ = float(σ)
        self.μ = float(μ)

    # From scipy: defined as lognorm.pdf(x, s, loc, scale)
    # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm

    # A common parametrization for a lognormal random variable Y is in terms of
    # the mean, mu, and standard deviation, sigma, of the unique normally distributed random variable X
    #such that exp(X) = Y.
    #This parametrization corresponds to setting s = sigma and scale = exp(mu).


    def ppf(self, quantiles):
        x = lognorm.ppf(quantiles, s=self.σ, loc=0, scale=np.exp(self.μ))
        return x


          # method: See hark
          # Makes an equiprobable distribution by
          # default, but user can optionally request augmented tails with exponentially
          # sized point masses.  This can improve solution accuracy in some models.
            #https://github.com/econ-ark/HARK/blob/d99393973554b1cf830c6285e6da59d98ff242ff/HARK/utilities.py

              #Returns
          #inodes
        #Discrete points for discrete probability mass function.
          #weights
        #Probability associated with each point in grid (nodes)

@language_element
@dataclass
class Beta(UnivariateIIDProcess):

    # If X∼Gamma(α) and Y∼Gamma(β) independently, then X/(X+Y)∼Beta(α,β).
    # Beta distribution with shape parameters α and β

    α: float
    β: float

    signature = {'α': 'float', 'β': 'float'}

    def __init__(self, α:float=None, β:float=None):
        self.α = float(α)
        self.β = float(β)

    def ppf(self, quantiles):
        x = beta.ppf(quantiles, self.α, self.β)
        return x

        # method: In hark utilities
          #reference https://github.com/econ-ark/HARK/blob/d99393973554b1cf830c6285e6da59d98ff242ff/HARK/utilities.py

        #Returns
          #inodes ("X" in hark)
        #Discrete points for discrete probability mass function.
          #weights ("pmf" in Hark : Discrete points for discrete probability mass function.)
        #Probability associated with each point in grid (nodes)
