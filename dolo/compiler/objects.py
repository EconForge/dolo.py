import numpy as np

from dolo.numeric.processes import MvNormal, DiscreteMarkovProcess, VAR1, MarkovProduct
from dolo.numeric.processes import IIDProcess
from dolo.numeric.grids import CartesianGrid, SmolyakGrid

Normal = MvNormal
MarkovChain = DiscreteMarkovProcess

AR1 = VAR1


class Domain(dict):
    def __init__(self, **kwargs):
        super().__init__()
        for k, w in kwargs.items():
            v = kwargs[k]
            self[k] = np.array(v, dtype=float)

    @property
    def min(self):
        return np.array([self[e][0] for e in self.states])

    @property
    def max(self):
        return np.array([self[e][1] for e in self.states])


#
# class CartesianGrid:
#
#     def __init__(self, min=None, max=None, n=None):
#
#         self.orders = np.array(orders, dtype=int)
#         if min is None:
#             min = np.zeros(len(n)) + 0.0
#         if max is None:
#             max = np.zeros(len(n)) + 1.0
#     if not (interpolation in ('spline','cspline')):
#         raise Exception("Interpolation method '{}' is not implemented for cartesian grids.")
#     self.interpolation = interpolation
#     self.__grid__ = None
#
# @property
# def grid(self):
#     if self.__grid__ is None:
#         from dolo.numeric.misc import mlinspace
#         self.__grid__ = mlinspace(self.a, self.b, self.orders)
#     return self.__grid__

#
# from interpolation.smolyak import SmolyakGrid # as SmolyakGridO

# class SmolyakGrid(SmolyakGridO):
#
#     def __init__(self, mu=2):
#
#         d = max([len(e) for e in [a,b,orders]])
#
#         # if interpolation not in ('chebychev','polynomial'):
#         #     raise Exception("Interpolation method '{}' is not implemented for Smolyak grids.")
#         # self.interpolation = interpolation
#         # super().__init__(d,mu)

if __name__ == '__main__':

    normal = Normal(sigma=[[0.3]])
    print(normal.discretize())
