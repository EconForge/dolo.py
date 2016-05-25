import numpy as np



class Normal:

    def __init__(self, sigma=None, orders=None, mu=None):

        self.sigma = np.array(sigma, dtype=float)
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
        return [x, w]


class MarkovChain(list):

    def __init__(self, P=None, Q=None):

        self.P = P
        self.Q = Q
        self.extend([P, Q]) # compatibility fix


class AR1(MarkovChain):

    def __init__(self, rho=None, sigma=None, N=2):

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
        self.P = P
        self.Q = Q
        self.extend([P, Q])

class MarkovProduct(MarkovChain):

    def __init__(self, M1=None, M2=None):

        from dolo.numeric.discretization import tensor_markov
        [P, Q] = tensor_markov( (M1.P,M1.Q), (M2.P, M2.Q) )
        self.P = P
        self.Q = Q
        self.extend([P,Q])


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
