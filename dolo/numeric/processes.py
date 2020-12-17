import numpy as np
import numpy
import pickle

from typing import List, TypeVar, Generic, Union, Any, Callable  # type: ignore
from typing import Iterator, Tuple  # type: ignore
from typing import List, Optional

Vector = List[float]
Matrix = List[Vector]

from numba import jit, njit
from dolo.numeric.grids import EmptyGrid, CartesianGrid, UnstructuredGrid
from random import random as rand

from xarray import DataArray

from dolang.language import greek_tolerance
from dolang.language import language_element


@njit
def choice(x, n, cumul):
    i = 0
    running = True
    while i < n and running:
        if x < cumul[i]:
            running = False
        else:
            i += 1
    return i


@njit
def simulate_mc_indices(C, N, T, i0):
    k = len(C)
    out = np.zeros((T, N), dtype=np.int32)
    out[0, :] = i0
    for n in range(N):
        for t in range(0, T - 1):
            ii = out[t, n]
            options = C[ii]
            x = rand()
            ix = choice(x, k, options)
            out[t + 1, n] = ix
    return out


def simulate_markov_chain(nodes, transitions, N, T, i0=None, return_values=False):
    i0_ = np.zeros(N, dtype=np.int32)
    if i0 is not None:
        i0_[:] = i0
    C = transitions.cumsum(axis=1)
    indices = simulate_mc_indices(C, N, T, i0_)
    if return_values:
        values = np.reshape(
            np.concatenate([Q[i, :][None, :] for i in indices.ravel()], axis=0),
            indices.shape + (-1,),
        )
        return indices, values
    else:
        return indices


class Process:

    d: int  # number of dimensions
    variables: Union[None, Tuple[str, ...]]  # names of variables (optional)

    @property
    def names(self):
        try:
            return self._names_
        except:
            self._names_ = tuple(f"_x_{i}" for i in range(self.d))
            return self._names_

    @names.setter
    def names(self, v):
        self._names_ = v


# A discrete process is such that integration
# can be computed exactly.
class DiscreteProcess(Process):

    n: int


class ContinuousProcess(Process):
    pass


class IIDProcess(Process):
    def draw(self, n: int) -> Vector:
        raise Exception("Not Implemented")

    def response(self, T, impulse) -> DataArray:
        irf = numpy.zeros((T + 1, self.d))
        irf[1, :] = impulse[None, :]
        coords = {"T": range(T + 1), "V": list(self.names)}
        xar = DataArray(irf, dims=coords.keys(), coords=coords)
        return xar

    def simulate(
        self, N: int = 10, T: int = 100, i0=None, m0=None, stochastic=True
    ) -> DataArray:
        sim = self.draw((T) * N).reshape((T, N, -1))
        coords = {"T": range(T), "N": range(N), "V": list(self.names)}
        xar = DataArray(sim, dims=coords.keys(), coords=coords)
        return xar


@language_element
class ConstantProcess(IIDProcess):

    signature = {"μ": "Vector"}

    @greek_tolerance
    def __init__(self, μ=None):

        if isinstance(μ, (float, int)):
            μ = [μ]
        self.μ = np.array(μ)
        assert self.μ.ndim == 1
        self.d = len(self.μ)

    def discretize(self, to=None, **kwargs):

        if to == "iid":
            x = self.μ[None, :]
            w = np.array([1.0])
            from .distribution import FiniteDistribution

            return FiniteDistribution(x, w)

        elif to == "mc":
            N = kwargs.get("N", 1)
            nodes = self.μ[None, :].repeat(N, axis=0)
            transitions = np.eye(N)
            return MarkovChain(transitions, nodes)
        else:
            raise Exception("Not implemented")


@language_element
class AggregateProcess(ConstantProcess):

    # that is a dummy class
    pass


# class VAR1(ContinuousProcess):
#     pass


class DiscretizedProcess:
    def iweight(self, i: int, j: int) -> float:
        raise Exception("Not Implemented.")

    def inode(self, i: int, j: int) -> float:
        raise Exception("Not Implemented.")

    def iteritems(self, i, eps=1e-16):
        for j in range(self.n_inodes(i)):
            w = self.iweight(i, j)
            if w > eps:
                x = self.inode(i, j)
                yield (w, x)


class GDP(DiscretizedProcess):
    def __init__(self, nodes, inodes, iweights, grid=None):
        self.__nodes__ = nodes
        self.__inodes__ = inodes
        self.__iweights__ = iweights
        self.__grid__ = grid

    # def discretize_gdp(self):
    #    return self
    @property
    def grid(self):
        return self.__grid__

    @property
    def n_nodes(self) -> int:
        return self.__nodes__.shape[0]

    def node(self, i: int):  # ->List:
        return self.__nodes__[i, :]

    @property
    def nodes(self):
        return self.__nodes__

    def n_inodes(self, i: int):  # ->int:
        return self.__inodes__.shape[1]

    def inode(self, i, j):  # ->List:
        return self.__inodes__[i, j]

    def iweight(self, i, j):  # ->float:
        return self.__iweights__[i, j]


class DiscretizedIIDProcess(DiscretizedProcess):
    def point(self, i: int) -> int:
        raise Exception("Not Implemented.")

    def weight(self, i: int) -> int:
        raise Exception("Not Implemented.")

    @property
    def grid(self):
        return EmptyGrid()

    @property
    def n_nodes(self) -> int:
        return 1  # purely conventional

    def node(self, i: int) -> Vector:
        return np.zeros(self.d)

    def n_inodes(self, i: int) -> int:
        return self.n

    def inode(self, i: int, j: int):  # vector
        return self.point(j)

    def iweight(self, i: int, j: int):  # scalar
        return self.weight(j)


@language_element
class MarkovChain(DiscretizedProcess, DiscreteProcess):

    signature = {"transitions": "Matrix", "values": "Matrix"}

    # transitions: 2d float array
    # values: 2d float array

    def __init__(self, transitions, values):

        self.transitions = np.array(transitions, dtype=np.float64)
        self.values = np.array(values, dtype=np.float64)
        self.d = self.values.shape[1]

    def discretize(self, to="mc"):

        if to == "gdp":
            nodes = self.values.copy()
            inodes = nodes[None, :, :].repeat(nodes.shape[0], axis=0)
            iweights = self.transitions
            return GDP(nodes, inodes, iweights)
            return
        elif to == "mc":
            return self
        else:
            raise Exception("Not implemented.")

    @property
    def grid(self):
        return UnstructuredGrid(self.values)

    @property
    def n_nodes(self):  # integer
        return self.values.shape[0]

    def node(self, i: int):  # vector
        return self.values[i, :]

    @property
    def nodes(self):
        return self.values

    def n_inodes(self, i: int):  # integer
        return self.transitions.shape[1]

    def inode(self, i: int, j: int):  # vector
        return self.values[j, :]

    def iweight(self, i: int, j: int):  # scalar
        return self.transitions[i, j]

    def simulate(self, N, T, i0=0, m0=None, stochastic=True):
        # m0 is basically ignored
        if stochastic:
            inds = simulate_markov_chain(self.values, self.transitions, N, T, i0=i0)
        else:
            inds = np.zeros((T, N), dtype=int) + i0
        return self.values[inds]


#%%


@language_element
class ProductProcess(Process):

    ### This class represents the product of processes

    def __init__(self, *l):
        self.processes = l
        self.d = sum([e.d for e in self.processes])

    def discretize(self, to=None, options={}):

        if isinstance(options, dict):
            kwargs = [options] * len(self.processes)
        else:
            assert len(options) == len(self.processes)
            kwargs = options

        if to is None:
            if any(
                [
                    isinstance(dp, MarkovChain) or isinstance(dp, ContinuousProcess)
                    for dp in self.processes
                ]
            ):
                to = "mc"
            elif all([isinstance(dp, IIDProcess) for dp in self.processes]):
                to = "iid"
            else:
                to = "gdp"

        if to == "iid":
            from dolo.numeric.distribution import product_iid

            fun = product_iid
        elif to == "mc":
            fun = product_mc
        elif to == "gdp":
            fun = product_gdp

        # discretize children
        discretized_processes = [
            e.discretize(to=to, **kwargs[i]) for i, e in enumerate(self.processes)
        ]

        return fun(discretized_processes)

    def simulate(self, N, T, m0=None, stochastic=True):

        if m0 is not None:
            raise Exception("Not implemented")
        sims = [
            p.simulate(N, T, m0=None, stochastic=stochastic) for p in self.processes
        ]
        return np.concatenate(sims, axis=2)


@language_element
def Product(*processes):
    return ProductProcess(*processes)


def product_mc(markov_chains: List[MarkovChain]) -> MarkovChain:

    M = [(m.values, m.transitions) for m in markov_chains]
    from dolo.numeric.discretization import tensor_markov

    [P, Q] = tensor_markov(*M)
    return MarkovChain(Q, P)


def product_gdp(gdps: List[GDP]) -> GDP:

    raise Exception("Not implemented")


@language_element
class AR1(ContinuousProcess):

    signature = {"ρ": "float", "σ": "float", "μ": "Optional[float]"}

    @greek_tolerance
    def __init__(self, ρ=None, σ=None, μ=None):
        self.ρ = ρ
        self.σ = σ
        self.μ = μ
        self.μ = μ if μ is not None else 0.0
        self.d = 1

    def discretize(self, N=3, to="mc", **kwargs):
        if to == "mc":
            return self.discretize_mc(N=N, **kwargs)
        elif to == "gdp":
            return self.discretize_gdp(N=N, **kwargs)

    def discretize_mc(self, N=3):

        rho = np.array([[self.ρ]])
        Sigma = np.array([[self.σ]])
        μ = np.array([self.μ])

        try:
            assert abs(np.eye(rho.shape[0]) * rho[0, 0] - rho).max() <= 1
        except:
            raise Exception(
                "When discretizing a Vector AR1 process, the autocorrelation coefficient must be as scalar. Found: {}".format(
                    rho_array
                )
            )

        from dolo.numeric.discretization import multidimensional_discretization

        [P, Q] = multidimensional_discretization(rho[0, 0], Sigma, N=N)

        P += μ[None, :]

        return MarkovChain(values=P, transitions=Q)

    def discretize_gdp(self, N=3):

        Σ = np.array([[self.σ]])
        ρ = self.ρ

        n_nodes = N
        n_std = 2.5
        n_integration_nodes = 5

        try:
            assert Σ.shape[0] == 1
        except:
            raise Exception("Not implemented.")

        try:
            assert ρ.shape[0] == ρ.shape[1] == 1
        except:
            raise Exception("Not implemented.")

        ρ = ρ[0, 0]
        σ = np.sqrt(Σ[0, 0])

        from dolo.numeric.discretization import gauss_hermite_nodes

        epsilons, weights = gauss_hermite_nodes([n_integration_nodes], Σ)

        min = -n_std * (σ / (np.sqrt(1 - ρ ** 2)))
        max = n_std * (σ / (np.sqrt(1 - ρ ** 2)))

        from .grids import CartesianGrid

        grid = CartesianGrid([min], [max], [n_nodes])

        nodes = np.linspace(min, max, n_nodes)[:, None]
        iweights = weights[None, :].repeat(n_nodes, axis=0)
        integration_nodes = np.zeros((n_nodes, n_integration_nodes))[:, :, None]
        for i in range(n_nodes):
            for j in range(n_integration_nodes):
                integration_nodes[i, j, :] = ρ * nodes[i, :] + epsilons[j]

        return GDP(nodes, integration_nodes, iweights, grid=grid)
        # return (nodes,integration_nodes,iweights)

    def simulate(self, N, T, m0=None, stochastic=True):

        Sigma = np.array([[self.σ]])
        mu = np.array([self.μ])
        rho = np.array([[self.ρ]])
        d = self.d

        if m0 is None:
            m0 = np.zeros(d)
        from numpy.random import multivariate_normal

        if stochastic:
            innov = multivariate_normal(mu, Sigma, N * T)
        else:
            innov = mu[None, len(mu)].repeat(T * N, axis=0)
        innov = innov.reshape((T, N, d))
        sim = np.zeros((T, N, d))
        sim[0, :, :] = m0[None, :]
        for t in range(1, sim.shape[0]):
            sim[t, :, :] = sim[t - 1, :, :] @ rho.T + innov[t, :, :]

        sim += μ[None, None, :]

        return sim

    def response(self, T, impulse):

        d = self.d
        rho = self.ρ
        μ = self.μ

        irf = np.zeros((T, d))
        irf[0, :] = impulse
        for t in range(1, irf.shape[0]):
            irf[t, :] = rho @ irf[t - 1, :]
        irf += μ[None, :]
        return irf


@language_element
class VAR1(ContinuousProcess):

    signature = {"ρ": "float", "Σ": "Matrix", "μ": "Optional[Vector]"}

    @greek_tolerance
    def __init__(self, ρ=None, Σ=None, μ=None):

        rho = ρ
        Sigma = Σ
        mu = μ

        self.Σ = np.atleast_2d(Sigma)
        d = self.Σ.shape[0]
        rho = np.array(rho)
        if rho.ndim == 0:
            self.ρ = np.eye(d) * rho
        elif rho.ndim == 1:
            self.ρ = np.diag(rho)
        else:
            self.ρ = rho
        if mu is None:
            self.μ = np.zeros(d)
        else:
            self.μ = np.array(mu, dtype=float)
        self.d = d

    def discretize(self, N=3, to="mc", **kwargs):
        if to == "mc":
            return self.discretize_mc(N=N, **kwargs)
        elif to == "gdp":
            return self.discretize_gdp(N=N, **kwargs)

    def discretize_mc(self, N=3):

        rho = self.ρ
        Sigma = self.Σ

        try:
            assert abs(np.eye(rho.shape[0]) * rho[0, 0] - rho).max() <= 1
        except:
            raise Exception(
                "When discretizing a Vector AR1 process, the autocorrelation coefficient must be as scalar. Found: {}".format(
                    rho_array
                )
            )

        from dolo.numeric.discretization import multidimensional_discretization

        [P, Q] = multidimensional_discretization(rho[0, 0], Sigma, N=N)

        P += self.μ[None, :]

        return MarkovChain(values=P, transitions=Q)

    def discretize_gdp(self, N=3):

        Σ = self.Σ
        ρ = self.ρ

        n_nodes = N
        n_std = 2.5
        n_integration_nodes = 5

        try:
            assert Σ.shape[0] == 1
        except:
            raise Exception("Not implemented.")

        try:
            assert ρ.shape[0] == ρ.shape[1] == 1
        except:
            raise Exception("Not implemented.")

        ρ = ρ[0, 0]
        σ = np.sqrt(Σ[0, 0])

        from dolo.numeric.discretization import gauss_hermite_nodes

        epsilons, weights = gauss_hermite_nodes([n_integration_nodes], Σ)

        min = -n_std * (σ / (np.sqrt(1 - ρ ** 2)))
        max = n_std * (σ / (np.sqrt(1 - ρ ** 2)))

        from .grids import CartesianGrid

        grid = CartesianGrid([min], [max], [n_nodes])

        nodes = np.linspace(min, max, n_nodes)[:, None]
        iweights = weights[None, :].repeat(n_nodes, axis=0)
        integration_nodes = np.zeros((n_nodes, n_integration_nodes))[:, :, None]
        for i in range(n_nodes):
            for j in range(n_integration_nodes):
                integration_nodes[i, j, :] = ρ * nodes[i, :] + epsilons[j]

        return GDP(nodes, integration_nodes, iweights, grid=grid)
        # return (nodes,integration_nodes,iweights)

    def simulate(self, N, T, m0=None, stochastic=True):

        d = self.d

        if m0 is None:
            m0 = np.zeros(d)
        from numpy.random import multivariate_normal

        Sigma = self.Σ
        mu = self.μ * 0
        if stochastic:
            innov = multivariate_normal(mu, Sigma, N * T)
        else:
            innov = mu[None, len(mu)].repeat(T * N, axis=0)
        innov = innov.reshape((T, N, d))
        rho = self.ρ
        sim = np.zeros((T, N, d))
        sim[0, :, :] = m0[None, :]
        for t in range(1, sim.shape[0]):
            sim[t, :, :] = sim[t - 1, :, :] @ rho.T + innov[t, :, :]

        sim += self.μ[None, None, :]

        return sim

    def response(self, T, impulse):
        d = self.d
        irf = np.zeros((T, d))
        irf[0, :] = impulse
        rho = self.ρ
        for t in range(1, irf.shape[0]):
            irf[t, :] = rho @ irf[t - 1, :]
        irf += self.μ[None, :]
        return irf


#### dummy class


@language_element
class Conditional(Process):

    signature = {"condition": "Any", "type": "Process", "arguments": "Function"}

    def __init__(self, condition=None, type=None, arguments=None):
        self.condition = condition
        self.type = type
        self.arguments = arguments
