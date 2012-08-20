from __future__ import division

import numpy as np

from operator import mul

from itertools import product

#try:
#    import pyximport;pyximport.install()
#    from dolo.numeric.chebychev_pyx import chebychev, chebychev2, cheb_extrema
#except:
from chebychev import cheb_extrema,chebychev,chebychev2

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
        ket = []
        for comb in self.smolyak_indices:
            p = reduce( mul, [Ts[comb[i],i,:] for i in range(self.d)] )
            ket.append(p)
        ket = np.row_stack( ket )

        self.__ket__ = ket
        self.__Ts__ = Ts

        self.bounds = np.row_stack([(0,1)]*d)

    def __call__(self,x):
        return self.interpolate(x)

    def interpolate(self, x, with_derivative=True, with_theta_deriv=False):
        # points in x must be stacked horizontally

        theta = self.theta

        [n_v, n_t] = theta.shape
        assert( n_t == self.n_points )
        n = theta.shape[1] - 1

        [n_d, n_p] = x.shape
        n_obs = n_p # by def
        assert( n_d == self.d )

        s = x

        Ts = chebychev( s, self.n_points - 1 )

        ket = []
        for comb in self.smolyak_indices:
            p = reduce( mul, [Ts[comb[i],i,:] for i in range(self.d)] )
            ket.append(p)
        ket = np.row_stack( ket )

        val = np.dot(theta,ket)
#
        if with_derivative:

#            bounds = self.bounds
#            bounds_delta = bounds[1,:] - bounds[0,:]
#            # derivative w.r.t. to theta
#            l = []
#            for i in range(n_v):
#                block = np.zeros( (n_v,n_t,n_obs) )
#                block[i,:,:] = ket
#                l.append(block)
#                dval = np.concatenate( l, axis = 1 )
#
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
                    block[i,:,:] = ket
                    l.append(block)
                dval = np.concatenate( l, axis = 1 )
                return [val,dder,dval]
            else:
                return [val,dder]

        else:
            return val


    def fit_values(self,res0):

        res0 = res0.real

        ket = self.__ket__
        #######
        # derivatives w.r.t theta on the grid
        l = []
        n_v = res0.shape[0]
        n_t = res0.shape[1]

        for i in range(n_v):                         # TODO : I shouldn't recompute it every time
            block = np.zeros( (n_v,n_t,n_t) )
            block[i,:,:] = ket
            l.append(block)
        self.__dval__ = np.concatenate( l, axis = 1 )
        ######

        #res0 = f(self.real_grid)
        theta0 = np.zeros(res0.shape)
        dval = self.__dval__
        #[val,dval,dder] = self.evalfun(theta0,self.real_grid,with_derivative=True)

        idv = dval.shape[1]
        jac = dval.swapaxes(1,2)
        jac = jac.reshape((idv,idv))

        import numpy.linalg
        theta = + np.linalg.solve(jac,res0.flatten())
        theta = theta.reshape(theta0.shape)

        self.theta =  theta

    def plot_grid(self):
        import matplotlib.pyplot as plt
        grid = smolyak_grids(self.d, self.l)[0]
        if grid.shape[1] == 2:
            xs = grid[:, 0]
            ys = grid[:, 1]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(xs, ys)
            ax.grid(True, linestyle='--',color='0.75')
            plt.show()
        elif grid.shape[1] == 3:
            from mpl_toolkits.mplot3d import Axes3D
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

    def __init__(self, bounds, l, axes=None):

        d = bounds.shape[1]

        super(SmolyakGrid, self).__init__( d, l)

        self.bounds = bounds

        self.center = [b[0]+(b[1]-b[0])/2 for b in bounds.T]
        self.radius =  [(b[1]-b[0])/2 for b in bounds.T]
#        print self.center
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


    def A(self,x):
        '''A is the inverse of B'''
        N = x.shape[1]
        c = np.tile(self.center, (N,1) ).T
        P = self.P
        return c + np.dot(P, x)

    def B(self,y):
        '''B is the inverse of A'''
        N = y.shape[1]
        c = np.tile(self.center, (N,1) ).T
        Pinv = self.Pinv
        return np.dot(Pinv,y-c)

    def interpolate(self, y, with_derivative=False, with_theta_deriv=False):
        x = self.B(y)
        res = super(SmolyakGrid, self).interpolate( x, with_derivative=with_derivative, with_theta_deriv=with_theta_deriv)
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

#    def fit_values(self):
#        nothing to change


# test smolyak library
if __name__ == '__main__':


    # we define a reference funcion :
    def testfun(x):
        val = np.row_stack([
            x[0,:]*x[0,:] * (1-np.power(x[1,:],2)),
            x[0,:]*(1-np.power(x[1,:],3)) + x[1,:]/4
        ])
        return val

    bounds = np.array([[0,1],[0,1]]).T
#    bounds = np.array([[-1,1],[-1,1]]).T
    sg2 = SmolyakGrid(bounds, 2)
    sg3 = SmolyakGrid(bounds, 3)

    from serial_operations import numdiff2, numdiff1

    theta2_0 = np.zeros( (2, sg2.n_points) )
    vals = testfun(sg2.grid)

    sg2.fit_values(vals)
#
#
#
#
#    print sg2.interpolate(sg2.real_grid)
#
#    [val,dval] = sg2.interpolate(sg2.real_grid)
#    [val0,dval0] = sg2.interpolate2(sg2.real_grid)
#    ddval = numdiff1(lambda x: sg2.interpolate(x,with_derivative=False)[0],sg2.real_grid)
#    print ddval.shape
#
#    print('val - val0')
#    print(val - val0)
#    print('dval - dval0')
#    print(ddval -dval)
#    print(ddval -dval0)
#    print ddval
#    print(dval0)


    def fobj(values):
        sg2.fit_values(values.reshape((2,5)))
        return sg2.interpolate(sg2.grid, with_derivative=False)


    print('derivatives w.r.t. parameters')
    [val,dval] = sg2.interpolate(sg2.grid, with_theta_deriv=True, with_derivative=False)

    vals = vals.reshape((1,10))
#    [val0,dval0] = numdiff1(fobj,vals)
    [val1,dval1] = numdiff2(fobj,vals)

    print vals.shape
    print fobj(vals).shape
    print dval.shape
    print dval.shape
#    print dval0.shape
    print(dval1.shape)

    exit()
    ddval = numdiff1(sg2.interpolate,sg2.real_grid)
    print ddval

    print dval.shape
    print ddval.shape

    exit()

    def fobj2(theta):
        grid = sg2.real_grid
        return testfun(grid) - sg2.interpolate(theta, grid)
    theta2_0 = np.zeros( (2, sg2.n_points) )
    res_2 = fobj2(theta2_0)

    def fobj3(theta):
        grid = sg3.real_grid
        return testfun(grid) - sg3.interpolate(theta, grid)
    theta3_0 = np.zeros( (2, sg3.n_points) )
    res_3 = fobj3(theta3_0)

    theta3_0[:,:5]=res_2
    #print sg2.evalfun(res_2, sg2.real_grid )[0]
    #print sg2.evalfun(res_2, sg3.real_grid )[0]

    #print sg3.evalfun(theta3_0, sg2.real_grid )[0]
    #print sg3.evalfun(theta3_0, sg3.real_grid )[0]

    theta3_0 = np.ones((2,13))
    theta3_0[:,:5] = 0
    print sg3.evalfun(theta3_0, sg3.real_grid )

    values = testfun(sg3.real_grid)
