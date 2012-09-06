"""
What if I document the module here ?
Are math formulas allowed : :math:`a_1=12` ?
"""



from __future__ import division

import numpy as np
from operator import mul
from itertools import product

def cheb_extrema(n):
    jj = np.arange(1.0,n+1.0)
    zeta =  np.cos( np.pi * (jj-1) / (n-1 ) )
    return zeta

def chebychev(x, n):
    # computes the chebychev polynomials of the first kind
    dim = x.shape
    results = np.zeros((n+1,) + dim)
    results[0,...] = np.ones(dim)
    results[1,...] = x
    for i in range(2,n+1):
        results[i,...] = 2 * x * results[i-1,...] - results[i-2,...]
    return results

def chebychev2(x, n):
    # computes the chebychev polynomials of the second kind
    dim = x.shape
    results = np.zeros((n+1,) + dim)
    results[0,...] = np.ones(dim)
    results[1,...] = 2*x
    for i in range(2,n+1):
        results[i,...] = 2 * x * results[i-1,...] - results[i-2,...]
    return results

def enum(d,l):
    r = range(l)
    b = l - 1
    #stupid :
    res = []
    for maximum in range(b+1):
        res.extend( [e for e in product(r, repeat=d ) if sum(e)==maximum ] )
    return res

def build_indices_levels(l):
    return [(0,)] + [(1,2)] + [ tuple(range(2**(i-1)+1, 2**(i)+1)) for i in range(2,l) ]

def build_basic_grids(l):
    ll = [ np.array( [0.5] ) ]
    ll.extend( [ np.linspace(0.0,1.0,2**(i)+1) for i in range(1,l) ]  )
    ll = [ - np.cos( e * np.pi ) for e in ll]
    incr = [[0.0],[-1.0,1.0]]
    for i in range(2,len(ll)):
        t = ll[i]
        n = (len(t)-1)/2
        tt =  [ t[2*n+1] for n in range( int(n) ) ]
        incr.append( tt )
    incr = [np.array(i) for i in incr]
    return [ll,incr]

def smolyak_grids(d,l):

    ret,incr = build_basic_grids(l)
    tab =  build_indices_levels(l)

    eee =  [ [ tab[i] for i in e] for e in enum( d, l) ]
    smolyak_indices = []
    for ee in eee:
        smolyak_indices.extend( [e for e in product( *ee ) ] )

    fff =  [ [ incr[i] for i in e] for e in enum( d, l) ]
    smolyak_points = []
    for ff in fff:
        smolyak_points.extend( [f for f in product( *ff ) ] )

    smolyak_points = np.c_[smolyak_points]

    return [smolyak_points, smolyak_indices]

class SmolyakBasic(object):
    '''Smolyak interpolation on [-1,1]^d'''

    def __init__(self,d,l):

        self.d = d
        self.l = l

        [self.smolyak_points, self.smolyak_indices] = smolyak_grids(d,l)

        self.u_grid = self.smolyak_points.T

        self.isup = max(max(self.smolyak_indices))
        self.n_points = len(self.smolyak_points)
        #self.grid = self.real_gri

        Ts = chebychev( self.smolyak_points.T, self.n_points - 1 )
        coeffs = []
        for comb in self.smolyak_indices:
            p = reduce( mul, [Ts[comb[i],i,:] for i in range(self.d)] )
            coeffs.append(p)
        coeffs = np.row_stack( coeffs )

        self.__coeffs__ = coeffs
        self.__Ts__ = Ts

        self.bounds = np.row_stack([(0,1)]*d)

    def __call__(self,s):
        return self.interpolate(s)

    def interpolate(self, s, with_derivative=True, with_theta_deriv=False):
        # points in x must be stacked horizontally

        theta = self.theta

        [n_v, n_t] = theta.shape  # (n_v, n_t) -> (number of variables?, ?)
        assert( n_t == self.n_points )

        [n_d, n_p] = s.shape  # (n_d, n_p) -> (number of dimensions, number of points)
        n_obs = n_p # by def
        assert( n_d == self.d )

        Ts = chebychev( s, self.n_points - 1 )

        coeffs = []
        for comb in self.smolyak_indices:
            p = reduce( mul, [Ts[comb[i],i,:] for i in range(self.d)] )
            coeffs.append(p)
        coeffs = np.row_stack( coeffs )

        val = np.dot(theta,coeffs)
#
        if with_derivative:

            # derivative w.r.t. arguments
            Us = chebychev2( s, self.n_points - 2 )
            Us = np.concatenate([np.zeros( (1,n_d,n_obs) ), Us],axis=0)
            for i in range(Us.shape[0]):
                Us[i,:,:] = Us[i,:,:] * i

            der_s = np.zeros( ( n_t, n_d, n_obs ) )
            for i in range(n_d):
                #BB = Ts.copy()
                #BB[:,i,:] = Us[:,i,:]
                el = []
                for comb in self.smolyak_indices:
                    #p = reduce( mul, [BB[comb[j],j,:] for j in range(self.d)] )
                    p = reduce( mul, [ (Ts[comb[j],j,:] if i!=j else Us[comb[j],j,:]) for j in range(self.d)] )
                    el.append(p)
                el = np.row_stack(el)
                der_s[:,i,:] =  el
            dder = np.tensordot( theta, der_s, (1,0) )

            if with_theta_deriv:
                # derivative w.r.t. to theta
                l = []
                for i in range(n_v):
                    block = np.zeros( (n_v,n_t,n_obs) )
                    block[i,:,:] = coeffs
                    l.append(block)
                dval = np.concatenate( l, axis = 1 )
                return [val,dder,dval]
            else:
                return [val,dder]

        else:
            return val


    def set_values(self,res0):
        """ Updates self.theta parameter. No returns values"""

        res0 = res0.real

        coeffs = self.__coeffs__
        #######
        # derivatives w.r.t theta on the grid
        l = []
        n_v = res0.shape[0]
        n_t = res0.shape[1]

        for i in range(n_v):
            block = np.zeros( (n_v,n_t,n_t) )
            block[i,:,:] = coeffs
            l.append(block)
        self.__dval__ = np.concatenate( l, axis = 1 )

        theta0 = np.zeros(res0.shape)
        dval = self.__dval__

        idv = dval.shape[1]
        jac = dval.swapaxes(1,2)
        jac = jac.reshape((idv,idv))

        import numpy.linalg
        theta = np.linalg.solve(jac,res0.flatten())
        theta = theta.reshape(theta0.shape)

        self.theta =  theta

    def plot_grid(self):
        import matplotlib.pyplot as plt
        grid = self.smolyak_points
        if grid.shape[1] == 2:
            xs = grid[:, 0]
            ys = grid[:, 1]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(xs, ys)
            ax.grid(True, linestyle='--',color='0.75')
            plt.show()
        elif grid.shape[1] == 3:
            xs = grid[:, 0]
            ys = grid[:, 1]
            zs = grid[:, 2]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs, ys, zs)
            ax.grid(True, linestyle='--',color='0.75')
            plt.show()
        else:
            raise ValueError('Can only plot 2 or 3 dimensional problems')

class SmolyakGrid(SmolyakBasic):

    '''Smolyak interpolation'''

    def __init__(self, bounds, l, axes=None):
        """
        @param bounds: matrix of bounds
        @param l:
        @param axes:
        @return: a smolyak interpolator
        """


        d = bounds.shape[1]

        super(SmolyakGrid, self).__init__( d, l)

        self.bounds = bounds

        self.center = [b[0]+(b[1]-b[0])/2 for b in bounds.T]
        self.radius =  [(b[1]-b[0])/2 for b in bounds.T]

        import numpy.linalg

        if not axes == None:
            self.P = np.dot( axes, np.diag(self.radius))
            self.Pinv = numpy.linalg.inv(axes)
        else:
            self.P = np.diag(self.radius)
            self.Pinv = numpy.linalg.inv(self.P)

        base = np.eye(d)
        image_of_base = np.dot( self.P , base)

        self.grid = self.A( self.u_grid )

    # A goes from [0,1] to bounds
    def A(self,x):
#        '''A is the inverse of B'''
        N = x.shape[1]
        c = np.tile(self.center, (N,1) ).T
        P = self.P
        return c + np.dot(P, x)

    # B returns from bounds to [0,1]
    def B(self,y):
#        '''B is the inverse of A'''
        N = y.shape[1]
        c = np.tile(self.center, (N,1) ).T
        Pinv = self.Pinv
        return np.dot(Pinv,y-c)

    def interpolate(self, y, with_derivative=False, with_theta_deriv=False):
        x = self.B(y)  # Transform back to [0,1]
        res = super(SmolyakGrid, self).interpolate( x, with_derivative=with_derivative, with_theta_deriv=with_theta_deriv)  # Call super class' (SmolyakGrid) interpolate func
        if with_derivative:
            if with_theta_deriv:
                [val,dder,dval] = res
                dder = np.tensordot(dder, self.Pinv, axes=(1,0)).swapaxes(1,2)
                return [val,dder,dval]
            else:
                [val,dder] = res
                dder = np.tensordot(dder, self.Pinv, axes=(1,0)).swapaxes(1,2)
                return [val,dder]
        else:
            return res


    def plot_grid(self):
        import matplotlib.pyplot as plt
        grid = self.grid
        if grid.shape[0] == 2:
            xs = grid[0, :]
            ys = grid[1, :]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(xs, ys)
            ax.grid(True, linestyle='--',color='0.75')
            plt.show()
        elif grid.shape[0] == 3:
            from mpl_toolkits.mplot3d import Axes3D
            xs = grid[0, :]
            ys = grid[1, :]
            zs = grid[2, :]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs, ys, zs)
            ax.grid(True, linestyle='--',color='0.75')
            plt.show()
        else:
            raise ValueError('Can only plot 2 or 3 dimensional problems')