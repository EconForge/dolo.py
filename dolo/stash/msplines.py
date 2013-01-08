from __future__ import division
from pylab import *


expr = '(t<-2)*(t+3)**3 + (-2<=t)*(t<-1)*(-3*t**3 -15*t**2 -21*t - 5) + (-1<=t)*(t<0)*(3*t**3 + 3*t**2 - 3*t + 1) + (t>=0)*(1-t)**3'

def eval_basis( t, method='numexpr' ):
    '''Evaluate spline basis functions whose breakpoints are (-2,-1,0,1,2) at t'''
    t -= 1
    if  method == 'numexpr':
        import numexpr
        expr = '(t+3)**3*(-3<=t)*(t<-2) + (-3*t**3 -15*t**2 -21*t - 5)*(-2<=t)*(t<-1) + (3*t**3 + 3*t**2 - 3*t + 1)*(-1<=t)*(t<0) + (1-t)**3*(t>=0)*(t<1)'
        res = numexpr.evaluate(expr)
    else:
        res = (-3<=t)*(t<-2)*(t+3)**3 + ((-2<=t)*(t<-1))*(-3*t**3 -15*t**2 -21*t - 5) + ((-1<=t)*(t<0))*(3*t**3 + 3*t**2 - 3*t + 1) + (t>=0)*(t<1)*(1-t)**3
    return res/6

import time

class MultivariateSplines:

    def __init__(self,a,b,p):
        '''Construct an multidimensional spline interpolator.
        a: vector of lower bounds 
        b: vector of upper bounds
        p: number of parameters in each dimension.

        The grid used to fit the spline contains (p-2) nodes in each dimension, plus two points close to the extrema.
        '''


        a = array(a,dtype=float)
        b = array(b,dtype=float)

        t = time.time()

        d = len(p)
        h = [ (b[i]-a[i])/(p[i]-3) for i in range(d)]

        breakpoints = [ linspace(a[i]-h[i],b[i]+h[i],p[i]) for i in range(d) ]

        f = 0.5
        knots = [ concatenate( [
                        [a[i], a[i]+h[i]*f],
                        linspace(a[i]+h[i],b[i]-h[i],p[i]-4),
                        [b[i]-h[i]*f, b[i]]
                    ] )
                for i in range(d)]

        from itertools import product

        grid = column_stack( product(*knots) )

        all_Phi = []
        for i in range(d):
            Phi = zeros((p[i],p[i]))
            for n in range(p[i]):
                Phi[n,:] = eval_basis( (knots[i] - breakpoints[i][n])/h[i] )
            all_Phi.append(Phi)

        inv_big_phi = inv(all_Phi[0].T)
        for i in range(1,d):
            inv_big_phi = kron(  inv_big_phi, inv(all_Phi[i].T) )
            #inv_big_phi = kron( inv_big_phi, inv(all_Phi[i]) )

        self.breakpoints = breakpoints
        self.h = h
        self.d = d
        self.a = a
        self.b = b
        self.p = p
        self.grid = grid
        self.all_Phi = all_Phi
        self.inv_big_phi = inv_big_phi

        s = time.time()
#        print('Time to initialize : {}'.format(s-t))

    def set_values(self, values):
        '''Computes the coefficients of the splines such that values are fitted exactly on the grid.

        values: (n_x x n_s) array where n_x is the number of variables and n_s the number of elements in the grid.'''

        n_x = values.shape[0]
        coeffs = [ dot(self.inv_big_phi, values[i,:] ) for i in range(n_x) ]
        self.coeffs = [reshape(c, self.p) for c in coeffs]
        self.padded_coeffs = [pad(c, 1, mode='edge') for c in self.coeffs]

    def __call__(self, s):

        return self.interpolate(s)

    def interpolate_new(self, s):
        '''complicated way to return a slightly wrong result (see interpolate)'''


    #        s = maximum( s, self.a[:,None])
    #        s = minimum( s, self.b[:,None])

        t_start = time.time()

        N = s.shape[1]
        n_x = len(self.coeffs)
        n_s = s.shape[0]

        qq = zeros( (n_s, N), dtype=int )
        rr = zeros( (n_s, N), dtype=float )
        nzs = zeros( (n_s, 4, N), dtype=int )
        #rr = zeros( (n_s, 4, N), dtype=float )

        bases_values = zeros( (n_s, 4, N ) )

        for i_s in range(n_s):
            h = self.h[i_s]
            q,r = divmod( s[i_s,:] - self.a[i_s], self.h[i_s])
            non_zero_splines = row_stack( [q-1, q, q+1, q+2] )
            #non_zero_splines = minimum(non_zero_splines, self.p[i_s]-1)
            #non_zero_splines = maximum(non_zero_splines, 0)
            qq[i_s,:] = q
            rr[i_s,:] = r
            nzs[i_s,:,:] = non_zero_splines

            A = eval_basis(r/h+1.0)
            B = eval_basis(r/h)
            C = eval_basis(r/h-1.0)
            D = eval_basis(r/h-2.0)
            V = row_stack([A,B,C,D])

            bases_values[i_s,:,:] = V

#        from dolo.numeric.tensor import mdot
#
#        vals = zeros( (n_x, N) )
#
#        for i_x in range(n_x):
#            padded_coeffs_x = self.padded_coeffs[i_x]
#            for n in range(N):
#                main_splines = qq[:,n]
#                ranges = [ slice(e, e+4) for e in main_splines ]
#                ccp = padded_coeffs_x[ranges]
#                vv = bases_values[:,:,n]
#                vals[i_x,n] = mdot( ccp, vv )
##                print(ccp.shape)
##                print(vv.shape)

        print('Time to interpolate : {}'.format(time.time()-t_start))

        return vals

    def interpolate(self, s):

        '''Evaluates the interpolated function at s.
        s: d x N array'''
        
        
        t_start = time.time()

        N = s.shape[1]
        n_x = len(self.coeffs)

        t1 = time.time()
        phis = []
        for i in range(self.d):
            vals = zeros( (self.p[i], N) )
            for j in range(self.p[i]):  # spline index
                node = self.breakpoints[i][j]
                h = self.h[i]
                vals[j,:] = eval_basis( (s[i,:]-node)/h )
#            vals = csr_matrix(vals)   # sparsify
            phis.append( vals )
        t2 = time.time()



        from  itertools import product

        phis_combinations = []
        inds = product( *[range(pp) for pp in self.p] )
        for ind in inds:
            phi = phis[0][ind[0]:ind[0]+1,:]
            for k,i in enumerate( ind[1:] ):
                rhs =phis[k+1][i:i+1,:]
#                phi = phi.multiply(rhs)
                phi = phi*rhs
            phis_combinations.append(phi)

#        from scipy.sparse import csr_matrix,vstack
#        phis_combinations = vstack(phis_combinations)

        phis_combinations = row_stack(phis_combinations)


        hh = phis_combinations.T
        vals = zeros( (n_x, N) )
        for i_x in range(n_x):
            coeffs_x = self.coeffs[i_x].flatten()
            coeffs_x = atleast_2d(coeffs_x)
            pprod = hh.dot(coeffs_x.T)
            vals[i_x,:] = pprod.flatten()


        return vals

#        vals = zeros( (n_x, N) )
#        from dolo.numeric.tensor import mdot
#        for i_x in range(n_x):
#            coeffs_x = self.coeffs[i_x]
#            inds = product( *[range(pp) for pp in self.p] )
#            for ind in inds:
#                tt = ones(N)
#                for ti,ii in enumerate(ind):
#                    tt *= phis[ti][ii,:]
#                vals[i_x,:] += coeffs_x[ind] * tt
##            for n in range(N):
##                v = mdot( coeffs_x, [phi[:,n] for phi in phis] )
##                vals[i_x,n] = v
#
#
#        return vals

if __name__ == '__main__':

    d = 2
    smin = array( [1]*d )
    smax = array( [2]*d )
    orders = [10]*d

    gamma = 16.0

    fun = lambda x :row_stack([
#            ( sqrt( x[0,:]**2 + x[1,:]**2 + x[2,:]**2 + x[3,:]**2 + x[4,:]**2) )**(1-gamma)/1-gamma
            (sqrt( x[0,:]**2 + 2*x[1,:]**2 ) )**(1-gamma)/1-gamma,
#            x[0,:]
            #(1+ sqrt( x[0,:]**2 + 2*x[1,:]**2 ) )**(1-gamma)/1-gamma
    ])
              #( sqrt( x[0,:]**2 + x[1,:]**2 + x[2,:]**2 + x[3,:]**2 + x[4,:]**2) )**(1-gamma)/1-gamma
#            ( sqrt( reduce(sum, [x[i,:]**2 for i in range(d)], zeros(x[0,:].shape) ) ) )**(1-gamma)/1-gamma

    #fun = lambda x: x[0,:]
    from dolo.numeric.interpolation.interpolation import RectangularDomain

    dom = RectangularDomain(smin,smax,[50]*d)

    vals = fun(dom.grid)

#    exit()

#    if False:
    from dolo.numeric.interpolation.smolyak import SmolyakGrid
    sg = SmolyakGrid(smin, smax, 4)
    sg.set_values( fun(sg.grid))

    tstart = time.time()

    vals_sg = sg(dom.grid)
    tend = time.time()
#
    print('Elapsed (smolyak) : {}'.format(tend-tstart))
    t1 = time.time()
    sp = MultivariateSplines(smin,smax,orders)
    t2 = time.time()
    sp.set_values( fun(sp.grid))
    t3 = time.time()
    vals_sp = sp.interpolate(dom.grid)
    t4 = time.time()
    #vals_sp = sp(dom.grid)

    print('Elapsed (splines) : {}, {}, {}'.format(t2-t1, t3-t2, t4-t3))


    print(sp.grid.shape)
    print(sg.grid.shape)

    error_sp = abs(vals_sp - vals).max()
    error_sg = abs(vals_sg - vals).max()

    print('Errors (splines)  : {}'.format(error_sp))
    print('Errors (smolyak)  : {}'.format(error_sg))


    from scipy.ndimage.interpolation import map_coordinates

    [R, C] = meshgrid(  linspace(smin[0], smax[0], orders[0]),  linspace(smin[1], smax[1], orders[1]) )

    regular_grid = row_stack( [R.T.flatten(), C.T.flatten() ])

#    regular_grid = test_dom.grid
    values_on_grid = fun(regular_grid)

    [xg, yg] = meshgrid(  linspace(1.1,2, 50),  linspace(1.1,2, 50))
    fine_grid = row_stack([xg.T.flatten(), yg.T.flatten()])


    fine_grid = regular_grid

    values_on_fine_grid = fun(fine_grid)

    sh = array(orders)-1

    print(regular_grid.shape)
    mmin = regular_grid.min(axis=1)
    mmax = regular_grid.max(axis=1)

    coordinates = (fine_grid-mmin[:,None])/(mmax[:,None]-mmin[:,None]) * sh[:,None]
#    coordinates = (mmax[:,None] - fine_grid)/(mmax[:,None]-mmin[:,None]) * sh[:,None]

#    coordinates = row_stack([coordinates[1,:], coordinates[0,:]])
    tt = time.time()
    vv = values_on_grid[0,:].reshape(orders)
    print(vv.shape)

    prefilter=False
    mode = 'nearest'
    interp_vals_1 = map_coordinates( vv , coordinates, order=1, prefilter=prefilter, mode=mode ).flatten()
    interp_vals_2 = map_coordinates( vv , coordinates, order=3, prefilter=prefilter, mode=mode ).flatten()
    interp_vals_3 = map_coordinates( vv , coordinates, order=4, prefilter=prefilter, mode=mode ).flatten()
    interp_vals_5 = map_coordinates( vv , coordinates, order=5, prefilter=prefilter, mode=mode ).flatten()

    ss = time.time()
    print('NDimage : {}'.format(ss-tt))
    print( abs(interp_vals_1 - values_on_fine_grid).max() )
    print( abs(interp_vals_2 - values_on_fine_grid).max() )
    print( abs(interp_vals_3 - values_on_fine_grid).max() )
    print( abs(interp_vals_5 - values_on_fine_grid).max() )




    interp_vals_3 = atleast_2d(interp_vals_3)
    print(interp_vals_3.shape)

    print(xg.shape)







    fine_grid = row_stack([xg.flatten(), yg.flatten()])

    true_vals = fun(fine_grid)
    interp_vals = sp.interpolate(fine_grid)
#    interp_new_vals = sp.interpolate_new(fine_grid)
    interp_new_vals = interp_vals_3

    interp_sg_vals = sg.interpolate(fine_grid)


#    plot(fine_grid.flatten(), true_vals.flatten())
#    plot(fine_grid.flatten(), interp_vals.flatten())
#    show()


    X = xg
    Y = yg
    Z = (true_vals[0,:]).reshape(X.shape)
    Za = (interp_new_vals[0,:]).reshape(X.shape)
    Zb = (interp_sg_vals[0,:]).reshape(X.shape)
    Zc = (interp_vals[0,:]).reshape(X.shape)


    print(sg.grid.shape)
    print(sp.grid.shape)

    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()

    ax = fig.add_subplot(221, projection='3d')
    surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    title('Function')

    ax = fig.add_subplot(222, projection='3d')
    surf = ax.plot_surface(X,Y,abs(Z-Za), rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    title('Error (splines)')

    ax = fig.add_subplot(223, projection='3d')
    surf = ax.plot_surface(X,Y,abs(Z-Zb), rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    title('Error (smolyak)')

    ax = fig.add_subplot(224, projection='3d')
    surf = ax.plot_surface(X,Y,abs(Za-Zc), rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    title('Error (diff)')


    print('Errors (splines)  : {}'.format(abs(Z-Za).max()))
    print('Errors (smolyak)  : {}'.format(abs(Z-Zb).max()))

    plt.show()
