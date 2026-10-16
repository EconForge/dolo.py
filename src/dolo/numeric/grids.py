from functools import reduce
from operator import mul
from quantecon import cartesian
import numpy as np
from numpy import zeros

from typing import TypeVar, Generic, Dict

T = TypeVar("T")
S = TypeVar("S")


def prod(l):
    return reduce(mul, l, 1.0)


from dolo.numeric.misc import mlinspace


class Grid:
    def __mul__(self, rgrid):
        return cat_grids(self, rgrid)

    @property
    def nodes(self):
        return self.__nodes__

    @property
    def n_nodes(self):
        return self.__nodes__.shape[0]

    def node(self, i):
        return self.__nodes__[i, :]


class ProductGrid(Grid, Generic[T, S]):
    def __init__(self, g1: T, g2: S, names=None):
        self.grids = [g1, g2]
        self.names = names

    def __getitem__(self, v):
        return self.grids[self.names.index(v)]

    def __repr__(self):
        return str.join(" × ", [e.__repr__() for e in self.grids])


class EmptyGrid(Grid):

    type = "empty"

    @property
    def nodes(self):
        return None

    @property
    def n_nodes(self):
        return 0

    def node(self, i):
        return None

    def __add__(self, g):
        return g


class PointGrid(Grid):

    type = "point"

    def __init__(self, point):
        self.point = np.array(point)

    @property
    def nodes(self):
        return None

    @property
    def n_nodes(self):

        return 1

    def node(self, i):
        return None


class UnstructuredGrid(Grid):

    type = "unstructured"

    def __init__(self, nodes):
        nodes = np.array(nodes, dtype=float)
        self.min = nodes.min(axis=0)
        self.max = nodes.max(axis=0)
        self.__nodes__ = nodes
        self.d = len(self.min)


class CartesianGrid(Grid):

    pass


class UniformCartesianGrid(CartesianGrid):

    type = "UniformCartesian"

    def __init__(self, min, max, n=[]):

        self.d = len(min)

        # this should be a tuple
        self.min = np.array(min, dtype=float)
        self.max = np.array(max, dtype=float)
        if len(n) == 0:
            self.n = np.zeros(n, dtype=int) + 20
        else:
            self.n = np.array(n, dtype=int)

        # this should be done only on request.
        self.__nodes__ = mlinspace(self.min, self.max, self.n)

    # def node(i:)
    # pass

    def __add__(self, g):

        if not isinstance(g, UniformCartesianGrid):
            raise Exception("Not implemented.")

        n = np.array(tuple(self.n) + tuple(g.n))
        min = np.array(tuple(self.min) + tuple(self.min))
        max = np.array(tuple(self.max) + tuple(self.max))

        return UniformCartesianGrid(min, max, n)

    def __numba_repr__(self):
        return tuple([(self.min[i], self.max[i], self.n[i]) for i in range(self.d)])


class NonUniformCartesianGrid(CartesianGrid):

    type = "NonUniformCartesian"

    def __init__(self, list_of_nodes):
        list_of_nodes = [np.array(l) for l in list_of_nodes]
        self.min = [min(l) for l in list_of_nodes]
        self.max = [max(l) for l in list_of_nodes]
        self.n = np.array([(len(e)) for e in list_of_nodes])
        # this should be done only on request.
        self.__nodes__ = cartesian(list_of_nodes)
        self.list_of_nodes = list_of_nodes  # think of a better name

    def __add__(self, g):

        if not isinstance(g, NonUniformCartesianGrid):
            raise Exception("Not implemented.")
        return NonUniformCartesianGrid(self.list_of_nodes + g.list_of_nodes)

    def __numba_repr__(self):
        return tuple([np.array(e) for e in self.list_of_nodes])


class SmolyakGrid(Grid):

    type = "Smolyak"

    def __init__(self, min, max, mu=2):

        from interpolation.smolyak import SmolyakGrid as ISmolyakGrid

        min = np.array(min)
        max = np.array(max)
        self.min = min
        self.max = max
        self.mu = mu
        d = len(min)
        sg = ISmolyakGrid(d, mu, lb=min, ub=max)
        self.sg = sg
        self.d = d
        self.__nodes__ = sg.grid


def cat_grids(grid_1, grid_2):

    if isinstance(grid_1, EmptyGrid):
        return grid_2
    if isinstance(grid_1, CartesianGrid) and isinstance(grid_2, CartesianGrid):
        min = np.concatenate([grid_1.min, grid_2.min])
        max = np.concatenate([grid_1.max, grid_2.max])
        n = np.concatenate([grid_1.n, grid_2.n])
        return CartesianGrid(min, max, n)
    else:
        raise Exception("Not Implemented.")


# compat
def node(grid, i):
    return grid.node(i)


def nodes(grid):
    return grid.nodes


def n_nodes(grid):
    return grid.n_nodes


if __name__ == "__main__":

    print("Cartsian Grid")
    grid = CartesianGrid([0.1, 0.3], [9, 0.4], [50, 10])
    print(grid.nodes)
    print(nodes(grid))

    print("UnstructuredGrid")
    ugrid = UnstructuredGrid([[0.1, 0.3], [9, 0.4], [50, 10]])
    print(nodes(ugrid))
    print(node(ugrid, 0))
    print(n_nodes(ugrid))

    print("Non Uniform CartesianGrid")
    ugrid = NonUniformCartesianGrid([[0.1, 0.3], [9, 0.4], [50, 10]])
    print(nodes(ugrid))
    print(node(ugrid, 0))
    print(n_nodes(ugrid))

    print("Smolyak Grid")
    sg = SmolyakGrid([0.1, 0.2], [1.0, 2.0], 2)
    print(nodes(sg))
    print(node(sg, 1))
    print(n_nodes(sg))
