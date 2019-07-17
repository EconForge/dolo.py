# GREEK TOLERANCE

greek_translation = {
   'sigma': 'σ',
   'rho': 'ρ',
   'mu': 'μ'
}

def greekify_dict(arg):
   dd = dict()
   for k in arg:
       if k in greek_translation:
           key = greek_translation[k]
       else:
           key = k
       if key in dd:
           raise Exception(f"key {key} defined twice")
       dd[key] = arg[k]
   return dd


def greek_tolerance(fun):

   def f(*pargs, **args):
       nargs = greekify_dict(args)
       return fun(*pargs, **nargs)

   return f


## Useful Links

## Common probability distributions:
## https://blog.cloudera.com/blog/2015/12/common-probability-distributions-the-data-scientists-crib-sheet/
##
## Distributions.jl:
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

    def discretize(self, N=5, method='equiprobable'):
        if method=='gauss-hermite':
            return self.__discretize_gh__(N=N)
        elif method=='equiprobable':
            return self.__discretize_ep__(N=N)
        else:
            raise Exception("Unknown discretization method.")

    def __discretize_ep__(self, N=5): #Equiprobable

        # this could be implemented once for all univariate processes which have a ppf
        xvec = np.linspace(0, 1, N+1)
        quantiles = 1/N/2 + xvec[:-1]
        x = self.ppf(quantiles)
        x = x[:,None]
        w = (1/(N))*np.ones(N)
        return DiscretizedIIDProcess(x, w)




@dataclass
class UNormal(UnivariateIIDProcess):

    μ: float=0.0
    σ: float=1.0


    @greek_tolerance
    def __init__(self, σ:float=None, μ:float=None):
        self.σ = float(σ)
        self.μ = 0.0 if μ is None else float(μ)

    def ppf(self, quantiles):
        x = norm.ppf(quantiles, loc=self.μ, scale=self.σ)
        return x

    def discretize(self, N=5, method='gauss-hermite'):
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

@dataclass
class Uniform(UnivariateIIDProcess):

    # uniform distribution over an interval [a,b]
    a: float
    b: float

    def __init__(self, a:float=None, b:float=None):
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

@dataclass
class LogNormal(UnivariateIIDProcess):



    # parametrization a lognormal random variable Y is in terms of
    # the mean, μ, and standard deviation, σ, of the unique normally distributed random variable X
    # such that exp(X) = Y.

    μ: float # log-mean μ
    σ: float # scale σ


    @greek_tolerance
    def __init__(self, σ:float=None, μ:float=None):
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


@dataclass
class Beta(UnivariateIIDProcess):

    # If X∼Gamma(α) and Y∼Gamma(β) independently, then X/(X+Y)∼Beta(α,β).
    # Beta distribution with shape parameters α and β

    α: float
    β: float


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
