import numpy as np
from scipy.stats import norm, uniform, lognorm, beta
from matplotlib import pyplot as plt
import numpy as np
from dataclasses import dataclass

from typing import List, TypeVar, Generic, Union, Any, Callable
from typing import Literal, Iterator, Tuple
Vector = List[float]
Matrix = List[Vector]



### 
### 1d distributions
###


class Distribution:

    def ppf(self, x: Vector)->Vector:

        raise Exception("Not Implemented.")

    def draw(self, n: int)->float:

        raise Exception("Not Implemented.")
        
    def integrate(self, fun)->float:
        
        raise Exception("Not Implemented.")


class ContinuousDistribution(Distribution):

    def discretize(self,  N=5, method='equiprobable', mass_point = "median" ,to='iid'):
        '''
        Returns a discretized version of this process.

        Parameters
        ----------
        N : int
            Number of point masses in the discretized distribution.

        method : str
            'equiprobable' or 'gauss-hermite'
        
        mass_point : str
            'median', 'left', 'middle', or 'right'

        to: str
            e.g. 'iid'

        Returns:
        ------------
        process : DiscretizedIIDProcess
            A discretized IID process derived from this continuous
            process.
        '''
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

D = TypeVar('D', bound=Distribution )
Cd = TypeVar('Cd', bound=ContinuousDistribution )

class DiscreteDistribution(Distribution):
    
    n: int # number of points

    def point(self, i)->float:

        raise Exception("Not Implemented.")

    def weight(self, i)->float:

        raise Exception("Not Implemented.")

    def items(self)->Iterator[Tuple[float,float]]:

        return ( (self.weight(i), self.point(i)) for i in range(self.n) )

    def integrate(self, fun:Callable[[float],float])->float:

        return sum(w*fun(x) for (w,x) in self.items())


# Here it is probably overkill to define FiniteDistribution separately

class FiniteDistribution(DiscreteDistribution):
    
    points: Vector
    weights: Vector

    def __init__(self, points: Vector, weights: Vector):

        self.n = len(self.points)
        assert(len(weights)==self.n)
        self.points = points
        self.weights = weights

    def point(self, i)->float:

        raise self.points[i]

    def weight(self, i)->float:

        raise self.weights[i]


class Uniform(ContinuousDistribution):

    # uniform distribution over an interval [a,b]
    a: float=0.0
    b: float=1.0

    def __init__(self, a:float=0.0, b:float=1.0):
        self.a = float(a)
        self.b = float(b)

    def ppf(self, quantiles:Vector)->Vector:
        x = uniform.ppf(quantiles, loc=self.a, scale=(self.b-self.a))
        return x

#%

class Truncation(ContinuousDistribution, Generic[Cd]):

    dist: Cd

    def __init__(self, dist: Cd, lb=-np.inf, ub=np.inf):

        self.dist = dist
        self.__min_ppf__, self.__max_ppf__= self.dist.ppf([lb, ub])

    def ppf(self, quantiles:Vector)->Vector:

        q_lb = self.__min_ppf__
        q_ub = self.__max_ppf__
        
        return np.maximum( np.minimum( self.dist.ppf(quantiles), q_ub ), q_lb )


        

# dist = Uniform(0, 1)

# trunc1 = Truncation(dist)
# trunc2 = Truncation[Uniform](dist)

# import numpy as np
# from scipy.stats import norm, uniform, lognorm, beta
# from matplotlib import pyplot as plt
# import numpy as np
# from dataclasses import dataclass

# from typing import List, TypeVar, Generic, Union, Any, Callable
# from typing import Literal, Iterator, Tuple
# Vector = List[float]
# Matrix = List[Vector]



# ### 
# ### 1d distributions
# ###


# class Distribution:

#     def draw(self, n: int)->float:

#         raise Exception("Not Implemented.")
        
#     def integrate(self, fun)->float:
        
#         raise Exception("Not Implemented.")


# class ContinuousDistribution(Distribution):
#     pass

# D = TypeVar('D', bound=Distribution )
# Cd = TypeVar('Cd', bound=ContinuousDistribution )

# class DiscreteDistribution(Distribution):
    
#     n: int # number of integration nodes

#     def node(self, i)->float:

#         raise Exception("Not Implemented.")

#     def weight(self, i)->float:

#         raise Exception("Not Implemented.")

#     def items(self)->Iterator[Tuple[float,float]]:

#         return ( (self.weight(i), self.node(i)) for i in range(self.n) )

#     def integrate(self, fun:Callable[[float],float])->float:

#         return sum(w*fun(x) for (w,x) in self.items())



# class Uniform(ContinuousDistribution):

#     # uniform distribution over an interval [a,b]
#     a: float=0.0
#     b: float=1.0

#     def __init__(self, a:float=0.0, b:float=1.0):
#         self.a = float(a)
#         self.b = float(b)

#     def ppf(self, quantiles:Vector)->Vector:
#         x = uniform.ppf(quantiles, loc=self.a, scale=(self.b-self.a))
#         return x

# ##%

# class Truncation(ContinuousDistribution, Generic[Cd]):

#     dist: Cd

#     def __init__(self, dist: Cd, lb=-np.inf, ub=np.inf):

#         self.dist = dist
#         self.__min_ppf__ = self.dist.ppf(lb)
#         self.__max_ppf__ = self.dist.ppf(ub)

#     def ppf(self, quantiles:Vector)->Vector:

#         q_lb = self.__min_ppf__
#         q_ub = self.__max_ppf__
        
#         return np.maximum( np.minimum( self.dist.ppf(quantiles), q_ub ), q_lb )



# ##%
# # class ProductDistribution(Generic[T,S]):
    
# #     a: T
# #     b: S
# #     d: int

# #     def __init__(self, a: T, b: S):
        
# #         self.a = a
# #         self.b = b
# #         self.d = a.d + b.d

# #     def draw(self, n) -> Matrix:

# #         x = np.zeros((n, 2))
# #         x[:,0] = self.a.draw(n)
# #         x[:,1] = self.b.draw(n)
# #         return x
