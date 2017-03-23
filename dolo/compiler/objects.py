import numpy as np

from dolo.numeric.processes import MvNormal, DiscreteMarkovProcess, VAR1, MarkovProduct
from dolo.numeric.processes import IIDProcess

Normal = MvNormal
MarkovChain = DiscreteMarkovProcess

AR1 = VAR1

class CartesianGrid:

    def __init__(self, a=None, b=None, orders=None, mu=None, interpolation='cspline'):

        assert(len(a) == len(b) == len(orders))
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)
        self.orders = np.array(orders, dtype=int)
        if not (interpolation in ('spline','cspline')):
            raise Exception("Interpolation method '{}' is not implemented for cartesian grids.")
        self.interpolation = interpolation
        self.__grid__ = None

    @property
    def grid(self):
        if self.__grid__ is None:
            from dolo.numeric.misc import mlinspace
            self.__grid__ = mlinspace(self.a, self.b, self.orders)
        return self.__grid__


from interpolation.smolyak import SmolyakGrid as SmolyakGridO

class SmolyakGrid(SmolyakGridO):

    def __init__(self, a=None, b=None, mu=2, orders=None, interpolation='chebychev'):
        assert(len(a) == len(b))
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        self.a = a
        self.b = b
        d = len(a)
        if interpolation not in ('chebychev','polynomial'):
            raise Exception("Interpolation method '{}' is not implemented for Smolyak grids.")
        self.interpolation = interpolation
        super().__init__(d,mu,a,b)


if __name__ == '__main__':

    normal = Normal(sigma=[[0.3]])
    print(normal.discretize())
