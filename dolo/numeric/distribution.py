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
# For which cases we have mv? mostly Normal/UNormal

# Parameter names are (so far) used as they appear in Distributions.jl
# This seems like the richest source for distributions with very clear documentation


import numpy as np  # type: ignore
from scipy.stats import norm, uniform, lognorm, beta  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from dataclasses import dataclass  # type: ignore

from typing import List, TypeVar, Generic, Union, Any, Callable  # type: ignore
from typing import Iterator, Tuple  # type: ignore

from dolang.language import greek_tolerance, language_element  # type: ignore
from dolo.numeric.processes import IIDProcess, DiscretizedIIDProcess, MarkovChain  # type: ignore

Vector = List[float]
Matrix = List[Vector]
T = TypeVar("T")


class Distribution(IIDProcess):
    """
    A multivariate distribution.

    Attributes:
        d(int): number of dimensions.
        names(list[str], optional): variable names
    """

    d: int  # number of dimensions
    names: Union[None, Tuple[str, ...]]  # names of variables (optional)

    def draw(self, N: int) -> Matrix:
        "Compute `N` random draws. Returns an `N` times `d` matrix."

        raise Exception(
            f"Not Implemented (yet). Should be implemented by subclass {self.__class__}"
        )

    def integrate(self, f) -> float:
        "Computes the expectation $E_u f(u)$ for given function `f`"

        raise Exception(
            f"Not Implemented (yet). Should be implemented by subclass {self.__class__}"
        )


###
### Continuous Distributions
###


class ContinuousDistribution(Distribution):
    def discretize(self, **kwargs):  # ->DiscreteDistribution:

        raise Exception(
            f"Not Implemented (yet). Should be implemented by subclass {self.__class__}"
        )


###
### Discrete Distributions
###


class DiscreteDistribution(Distribution, DiscretizedIIDProcess):
    """
    A multivariate discrete distribution.

    Attributes:
        d(int): number of dimensions.
        names(list[str], optional): variable names
        n(int):  number of discretization points
        origin(distribution, optional): distribution that was discretized
    """

    n: int  # number of discretization points
    origin: Union[None, ContinuousDistribution]

    def point(self, i) -> Vector:
        "Returns i-th discretization point (a Vector)"
        raise Exception(
            f"Not Implemented (yet). Should be implemented by subclass {self.__class__}"
        )

    def weight(self, i) -> float:
        "Returns i-th discretization point (a float)"
        raise Exception(
            f"Not Implemented (yet). Should be implemented by subclass {self.__class__}"
        )

    def items(self) -> Iterator[Tuple[float, Vector]]:
        """Returns a generator yielding all points and weights.

        Example: sum( [ w*f(x) for (w,x) in discrete_dist.items() ] )
        """
        return ((self.weight(i), self.point(i)) for i in range(self.n))

    def integrate(self, fun: Callable[[Vector], T]) -> T:
        # alread documented by the ancestor
        return sum(w * fun(x) for (w, x) in self.items())


class EquiprobableDistribution(DiscreteDistribution):

    points: Vector

    def __init__(self, points: Vector = None, origin: Union[Distribution, None] = None):

        n, d = points.shape
        self.d = d
        self.n = n
        self.points = points
        self.origin = origin

    @property
    def weights(self) -> Vector:
        # so that it can behave like a FiniteDistribution (notably for graphs)
        w = np.ones(self.n)
        w /= self.n
        return w

    def point(self, i) -> float:

        return self.points[i, :]

    def weight(self, i) -> float:

        return 1 / self.n

    def draw(self, N: int) -> Matrix:
        import numpy.random

        inds = numpy.random.randint(low=0, high=self.n, size=N)
        return self.points[inds, :]

    def discretize(self, to="iid"):
        if to == "iid":
            return self
        elif to == "mc":
            return FiniteDistribution(self.points, self.weights).discretize(to="mc")
        else:
            raise Exception("Not implemented.")

    def __repr__(self):
        return f"EquiprobableDistribution(points={self.points.__repr__()}, origin={str(self.origin)})"

    def __str__(self):
        return f"EquiprobableDistribution(points={self.points}, origin={self.origin})"


# Special kind of Discrete distributions characterized
# by a list of points and a list of weights.
class FiniteDistribution(DiscreteDistribution):

    points: Vector
    weights: Vector

    def __init__(
        self,
        points: Vector = None,
        weights: Vector = None,
        origin: Union[Distribution, None] = None,
    ):

        n, d = points.shape
        self.d = d
        self.n = n
        assert len(weights) == n
        self.points = points
        self.weights = weights
        self.origin = origin

    def draw(self, N: int) -> Matrix:

        import numpy.random

        choices = numpy.random.choice(range(self.n), size=N, p=self.weights)
        return self.points[choices, :]

    def point(self, i) -> float:

        return self.points[i, :]

    def weight(self, i) -> float:

        return self.weights[i]

    def discretize(self, to="iid"):
        if to == "iid":
            return self
        elif to == "mc":
            from .processes import MarkovChain

            nodes = self.points
            N = len(nodes)
            transitions = np.array(
                [
                    self.weights,
                ]
                * N
            )
            return MarkovChain(transitions, nodes)
        else:
            raise Exception("Not implemented.")

    def __repr__(self):
        return f"FiniteDistribution(points={self.points.__repr__()}, weights={self.weights.__repr__()}, origin={str(self.origin)})"

    def __str__(self):
        return f"FiniteDistribution(points={self.points}, weights={self.weights}, origin={self.origin})"


def product_iid(iids: List[FiniteDistribution]) -> FiniteDistribution:

    from dolo.numeric.misc import cartesian

    nn = [len(f.weights) for f in iids]

    cart = cartesian([range(e) for e in nn])

    nodes = np.concatenate(
        [f.points[cart[:, i], :] for i, f in enumerate(iids)], axis=1
    )
    weights = iids[0].weights
    for f in iids[1:]:
        weights = np.kron(weights, f.weights)

    return FiniteDistribution(nodes, weights)


###
### Discrete Distributions
###


@language_element
##@dataclass
class Bernouilli(DiscreteDistribution):

    π: float = 0.5

    signature = {"π": "float"}  # this is redundant for now

    @greek_tolerance
    def __init__(self, π: float = None):
        self.π = float(π)

    def discretize(self, to="iid"):

        if to == "iid":
            x = np.array([[0], [1]])
            w = np.array([1 - self.π, self.π])
            return FiniteDistribution(x, w)
        elif to == "mc":
            fin_distr = self.discretize(to="iid")
            return fin_distr.discretize(to="mc")
        else:
            raise Exception("Not implemented.")

    def draw(self, N: int) -> Matrix:
        a = np.array([0, 1])
        c = np.random.choice(a, size=N)
        return c.reshape((N, 1))


@language_element
##@dataclass
class Binomial(DiscreteDistribution):

    π: float = 0.5
    n: int

    # TODO


###
### 1d Continuous Distributions
###


class UnivariateContinuousDistribution(ContinuousDistribution):
    """
    A univariate distribution.

    Attributes:
        d(int): number of dimensions.
        names(list[str], optional): variable names
    """

    d = 1

    def ppf(self, quantiles: Vector) -> Vector:
        "Percentage Point Function (inverse CDF)"
        raise Exception(
            f"Not Implemented (yet). Should be implemented by subclass {self.__class__}"
        )

    def cdf(self, quantiles: Vector) -> Vector:
        "Cumulative Distribution"
        raise Exception(
            f"Not Implemented (yet). Should be implemented by subclass {self.__class__}"
        )

    def discretize(self, to="iid", N=5, method="equiprobable", mass_point="median"):
        """
        Returns a discretized version of this process.

        Parameters
        ----------
        N : int
            Number of point masses in the discretized distribution.

        method : str
            'equiprobable' or 'gauss-hermite'

        mass_point : str
            'median', 'left', 'middle', or 'right'


        Returns:
        ------------
        process : DiscreteDistribution
            A discrete distribution.
        """
        if to == "iid":
            if method == "gauss-hermite":
                return self.__discretize_gh__(N=N)
            elif method == "equiprobable":
                return self.__discretize_ep__(N=N, mass_point=mass_point)
            else:
                raise Exception("Unknown discretization method.")
        elif to == "mc":
            discr_iid = self.discretize(to="iid")
            return discr_iid.discretize(to="mc")
        else:
            raise Exception("Not implemented (yet).")

    def __discretize_ep__(self, N=5, mass_point="median"):  # Equiprobable
        if mass_point == "median":
            p = np.linspace(0.5 / N, 1 - 0.5 / N, N)
            q = self.ppf(p)
        elif mass_point == "left":
            p = np.linspace(0, 1 - 1 / N, N)
            q = self.ppf(p)
        elif mass_point == "middle":
            p = np.linspace(0.0, 1, N + 1)
            q = self.ppf(p)
            q = 0.5 * (q[1:] + q[:-1])
        elif mass_point == "right":
            p = np.linspace(1 / N, 1, N)
            q = self.ppf(p)
        else:
            raise Exception("Not implemented")

        w = (1 / (N)) * np.ones(N)

        return EquiprobableDistribution(q[:, None], origin=self)


@language_element
##@dataclass
class UNormal(UnivariateContinuousDistribution):

    μ: float = 0.0
    σ: float = 1.0

    signature = {"μ": "Optional[float]", "σ": "float"}  # this is redundant for now

    @greek_tolerance
    def __init__(self, σ: float = None, μ: float = None):
        self.σ = float(σ)
        self.μ = 0.0 if μ is None else float(μ)

    def ppf(self, quantiles):
        x = norm.ppf(quantiles, loc=self.μ, scale=self.σ)
        return x

    def cdf(self, x):
        p = norm.cdf(x, loc=self.μ, scale=self.σ)
        return p

    def draw(self, N):

        from numpy.random import multivariate_normal

        Sigma = np.array([[self.σ ** 2]])
        mu = np.array([self.μ])
        sim = multivariate_normal(mu, Sigma, size=N)
        return sim.reshape((N, 1))

    def integrate(self, fun) -> float:

        # I don't think anybody should use that. It's just an example
        σ = self.σ
        μ = self.μ

        f = (
            lambda x: fun(x)
            / σ
            / np.sqrt(2 * np.pi)
            * np.exp(-1 / 2 * ((x - μ) / σ) ** 2)
        )
        import scipy.integrate

        return scipy.integrate.quad(f, -np.Inf, np.Inf)[0]

    def __discretize_gh__(self, N=5):  # Gauss-Hermite

        # Maybe we can avoid that one by inheriting from mvNormal
        from dolo.numeric.discretization.quadrature import gauss_hermite_nodes

        [x, w] = gauss_hermite_nodes(N, np.array([[self.σ ** 2]]), mu=self.μ)
        x += np.array([self.μ])[:, None]
        return FiniteDistribution(x, w, origin=self)


@language_element
##@dataclass
class Uniform(UnivariateContinuousDistribution):

    # uniform distribution over an interval [a,b]
    a: float = 0.0
    b: float = 1.0

    signature = {"a": "float", "b": "float"}  # this is redundant for now

    def __init__(self, a: float = 0.0, b: float = 1.0):
        self.a = float(a)
        self.b = float(b)

    def ppf(self, quantiles: Vector) -> Vector:
        x = uniform.ppf(quantiles, loc=self.a, scale=(self.b - self.a))
        return x

    def cdf(self, x: Vector) -> Vector:
        p = uniform.cdf(x, loc=self.a, scale=(self.b - self.a))
        return p

    def draw(self, N) -> Matrix:
        from numpy.random import uniform

        sim = uniform(self.a, self.b, N)
        return sim.reshape((N, 1))


@language_element
# @dataclass
class LogNormal(UnivariateContinuousDistribution):

    # parametrization a lognormal random variable Y is in terms of
    # the mean, μ, and standard deviation, σ, of the unique normally distributed random variable X
    # such that exp(X) = Y.

    μ: float = 0.0
    σ: float = 1.0

    signature = {"μ": "Optional[float]", "σ": "float"}  # this is redundant for now

    @greek_tolerance
    def __init__(self, σ: float = 0.0, μ: float = 1.0):
        self.σ = float(σ)
        self.μ = float(μ)

    # From scipy: defined as lognorm.pdf(x, s, loc, scale)
    # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm

    # A common parametrization for a lognormal random variable Y is in terms of
    # the mean, mu, and standard deviation, sigma, of the unique normally distributed random variable X
    # such that exp(X) = Y.
    # This parametrization corresponds to setting s = sigma and scale = exp(mu).

    def ppf(self, quantiles):
        x = lognorm.ppf(quantiles, s=self.σ, loc=0, scale=np.exp(self.μ))
        return x

        # method: See hark
        # Makes an equiprobable distribution by
        # default, but user can optionally request augmented tails with exponentially
        # sized point masses.  This can improve solution accuracy in some models.
        # https://github.com/econ-ark/HARK/blob/d99393973554b1cf830c6285e6da59d98ff242ff/HARK/utilities.py

        # Returns
        # inodes
        # Discrete points for discrete probability mass function.
        # weights
        # Probability associated with each point in grid (nodes)


class Beta(UnivariateContinuousDistribution):

    α: float
    β: float

    signature = {"α": "float", "β": "float"}

    def __init__(self, α: float = None, β: float = None):
        self.α = float(α)
        self.β = float(β)

    def ppf(self, quantiles):
        x = beta.ppf(quantiles, self.α, self.β)
        return x

        # method: In hark utilities
        # reference https://github.com/econ-ark/HARK/blob/d99393973554b1cf830c6285e6da59d98ff242ff/HARK/utilities.py

        # Returns
        # inodes ("X" in hark)
        # Discrete points for discrete probability mass function.
        # weights ("pmf" in Hark : Discrete points for discrete probability mass function.)
        # Probability associated with each point in grid (nodes)


#%


###
### nd continous distribution
###


@language_element
class Normal(ContinuousDistribution):

    Μ: Vector  # this is capital case μ, not M... 😭
    Σ: Matrix

    signature = {"Σ": "Matrix", "Μ": "Optional[Vector]"}

    @greek_tolerance
    def __init__(self, Σ=None, Μ=None):

        Sigma = Σ
        mu = Μ

        self.Σ = np.atleast_2d(np.array(Sigma, dtype=float))
        self.d = len(self.Σ)
        if mu is None:
            self.Μ = np.array([0.0] * self.d)
        else:
            self.Μ = np.array(mu, dtype=float)
        assert self.Σ.shape[0] == self.d
        assert self.Σ.shape[0] == self.d

        # this class wraps functionality from scipy
        import scipy.stats

        self._dist_ = scipy.stats.multivariate_normal(mean=self.Μ, cov=self.Σ)

    def draw(self, N: int) -> Matrix:

        res = self._dist_.rvs(size=N)
        if res.ndim == 1:
            # scipy returns a 1d object for 1d distribution
            res = res[:, None]

        return res

    def discretize(self, to="iid", N=None) -> FiniteDistribution:

        if to == "iid":
            if N is None:
                N = 5
            if isinstance(N, int):
                N = [N] * self.d
            from dolo.numeric.discretization.quadrature import gauss_hermite_nodes  # type: ignore

            [x, w] = gauss_hermite_nodes(N, self.Σ, mu=self.Μ)
            x = np.row_stack([(e + self.Μ) for e in x])
            return FiniteDistribution(x, w, origin=self)

        elif to == "mc":
            discr_iid = self.discretize(to="iid")
            return discr_iid.discretize(to="mc")

        else:
            raise Exception("Not implemented.")

    def __repr__(self):
        return f"Normal(Μ={self.Μ.__repr__()},Σ={self.Σ.__repr__()})"

    def __str__(self):
        return f"Normal(Μ={self.Μ},Σ={self.Σ})"


MvNormal = Normal


class ProductDistribution(ContinuousDistribution):

    distributions: List[Distribution]

    # def __new__(self, distributions: List[Distributions]):
    #     # if all distributions are normal we can interrupt the object
    #     # construction and return a multivariate normal object instead
    #     # of a product object

    def __init__(self, distributions: List[Distribution]):

        self.distributions = distributions
        self.d = sum([dis.d for dis in distributions])
        self.names = sum([dis.names for dis in self.distributions], tuple())

    def discretize(self, to="iid"):

        # TODO: pass some options
        fids = [dis.discretize(to=to) for dis in self.distributions]
        return product_iid(fids)

    def draw(self, N: int) -> Matrix:

        return np.concatenate([dis.draw(N) for dis in self.distributions], axis=1)


def product_iid(iids: List[FiniteDistribution]) -> FiniteDistribution:

    from dolo.numeric.misc import cartesian

    nn = [len(f.weights) for f in iids]

    cart = cartesian([range(e) for e in nn])

    nodes = np.concatenate(
        [f.points[cart[:, i], :] for i, f in enumerate(iids)], axis=1
    )
    weights = iids[0].weights
    for f in iids[1:]:
        weights = np.kron(weights, f.weights)

    return FiniteDistribution(nodes, weights)


###
### Truncation and Mixtures
###

C = TypeVar("C", bound=ContinuousDistribution)
C1 = TypeVar("C1", bound=ContinuousDistribution)
C2 = TypeVar("C2", bound=ContinuousDistribution)


class Truncation(UnivariateContinuousDistribution, Generic[C]):

    dist: C

    def __init__(self, dist: C, lb=-np.inf, ub=np.inf):

        self.dist = dist
        if lb == -np.inf:
            self.__min_q__ = 0.0
        else:
            self.__min_q__ = self.dist.cdf([lb])[0]
        if ub == np.inf:
            self.__max_q__ = 1.0
        else:
            self.__max_q__ = self.dist.cdf([ub])[0]

    def draw(self, N: int):

        # TODO: replace this stupid algo
        raise Exception("Not Implemented")

    def ppf(self, quantiles: Vector) -> Vector:

        q_lb = self.__min_q__
        q_ub = self.__max_q__

        q = q_lb + (q_ub - q_lb) * quantiles
        return self.dist.ppf(q)


@language_element
class Mixture(ContinuousDistribution):

    index: DiscreteDistribution  # values must be [0,1,..n]
    distributions: Tuple[UnivariateContinuousDistribution, ...]  # length musth be [n]

    signature = {"index": "DiscreteDistribution", "distributions": "List[Distribution]"}

    def __init__(self, index=None, distributions=None):
        # index is a distribution which takes discrete values
        # distributions is a map from each of these values to a distribution
        self.index = index
        self.distributions = distributions
        ds = [e.d for e in self.distributions.values()]
        assert len(set(ds)) == 1
        d0 = [*self.distributions.values()][0]
        self.d = d0.d
        # TODO: check all distributions have the same variable names
        self.names = d0.names

    def discretize(self, to="iid"):

        if to == "iid":

            inddist = self.index.discretize(to=to)
            nodes = []
            weights = []
            for i in range(inddist.n_inodes(0)):
                wind = inddist.iweight(0, i)
                xind = inddist.inode(0, i)
                dist = self.distributions[str(i)].discretize(to=to)
                for j in range(dist.n_inodes(0)):
                    w = dist.iweight(0, j)
                    x = dist.inode(0, j)
                    nodes.append(x)
                    weights.append(wind * w)
            nodes = np.concatenate([e[None, :] for e in nodes], axis=0)
            weights = np.array(weights)
            return FiniteDistribution(nodes, weights)

        elif to == "mc":
            from dolo.numeric.processes import DiscretizedIIDProcess

            return self.discretize(to="iid").discretize(to="mc")

        else:
            raise Exception("Not implemented.")

    def draw(self, N: int) -> Matrix:

        # naive and overkill algorithm
        inds = self.index.draw(N)  # should be (N x 1) array
        return sum(
            [(inds == k) * dist.draw(N) for (k, dist) in self.distributions.items()]
        )


# @language_element
# def Mixture(index=None, distributions=None):

#     for dist in distributions.values():
#         if not (isinstance(dist, IIDProcess)):
#             raise Exception("Only mixtures of iid processes are supported so far.")
#     return IIDMixture(index, distributions)
#     # not clear what we might do with non-iid

# Mixture.signature = {'index': 'intprocess', 'distributions': 'Dict[int,IIDProcesses]'}
