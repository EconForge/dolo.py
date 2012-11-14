from __future__ import division
from pylab import *
import numexpr


expr = '(t<-2)*(t+3)**3 + (-2<=t)*(t<-1)*(-3*t**3 -15*t**2 -21*t - 5) + (-1<=t)*(t<0)*(3*t**3 + 3*t**2 - 3*t + 1) + (t>=0)*(1-t)**3'

def eval_basis( t, method='numexpr' ):
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

        n_x = values.shape[0]
        coeffs = [ dot(self.inv_big_phi, values[i,:] ) for i in range(n_x) ]
        self.coeffs = [reshape(c, self.p) for c in coeffs]
        self.padded_coeffs = [pad(c, 1, mode='edge') for c in self.coeffs]

    def __call__(self, s):
        return self.interpolate(s)

    def interpolate_new(self, s):


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


#        s = maximum( s, self.a[:,None])
#        s = minimum( s, self.b[:,None])

        t_start = time.time()

        N = s.shape[1]
        n_x = len(self.coeffs)

        from scipy.sparse import csr_matrix, lil_matrix, kron, hstack

        t1 = time.time()
        phis = []
        for i in range(self.d):
            vals = zeros( (self.p[i], N) )
            for j in range(self.p[i]):  # spline index
                node = self.breakpoints[i][j]
                h = self.h[i]
                vals[j,:] = eval_basis( (s[i,:]-node)/h )
            vals = csr_matrix(vals)
            phis.append( vals )
        t2 = time.time()
#        print('Evaluation of basis : {}'.format(t2 -t1))



        from  itertools import product

        phis_combinations = []
        inds = product( *[range(pp) for pp in self.p] )
        for ind in inds:
            phi = phis[0][ind[0]:ind[0]+1,:]
            for k,i in enumerate( ind[1:] ):
                rhs =phis[k+1][i:i+1,:]
                phi = phi.multiply(rhs)
#                phi = multiply(phi,rhs)
            phis_combinations.append(phi)
#        phis_combinations = row_stack(phis_combinations)
        from scipy.sparse import csr_matrix,vstack

        phis_combinations = vstack(phis_combinations)


        hh = phis_combinations.T
        import numexpr
        vals = zeros( (n_x, N) )
        for i_x in range(n_x):
            coeffs_x = self.coeffs[i_x].flatten()
            coeffs_x = atleast_2d(coeffs_x)

#            pprod = dot(coeffs_x, phis_combinations)

            pprod = hh.dot(coeffs_x.T)

            vals[i_x,:] = pprod.flatten()


        #t_end = time.time()

        t_end = time.time()

        return vals

#        vals = zeros( (n_x, N) )
#        from dolo.numeric.tensor import mdot
#        for i_x in range(n_x):
#
#            coeffs_x = self.coeffs[i_x]
##            inds = product( *[range(pp) for pp in self.p] )
##            for ind in inds:
##                tt = ones(N)
##                for ti,ii in enumerate(ind):
##                    tt *= phis[ti][ii,:]
##                vals[i_x,:] += coeffs_x[ind] * tt
#            for n in range(N):
#                v = mdot( coeffs_x, [phi[:,n] for phi in phis] )
#                vals[i_x,n] = v
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
            #(1+ sqrt( x[0,:]**2 + 2*x[1,:]**2 ) )**(1-gamma)/1-gamma
    ])
              #( sqrt( x[0,:]**2 + x[1,:]**2 + x[2,:]**2 + x[3,:]**2 + x[4,:]**2) )**(1-gamma)/1-gamma
#            ( sqrt( reduce(sum, [x[i,:]**2 for i in range(d)], zeros(x[0,:].shape) ) ) )**(1-gamma)/1-gamma

    #fun = lambda x: x[0,:]
    from dolo.numeric.interpolation import RectangularDomain
    dom = RectangularDomain(smin+0.1,smax,[50]*d)

    vals = fun(dom.grid)

#    exit()

#    if False:
    from dolo.numeric.smolyak import SmolyakGrid
    sg = SmolyakGrid(smin, smax, 4)
    sg.set_values( fun(sg.grid))

    tstart = time.time()

    vals_sg = sg(dom.grid)
    tend = time.time()
    print('Elapsed : {}'.format(tend-tstart))
#
#
    sp = MultivariateSplines(smin,smax,orders)
    sp.set_values( fun(sp.grid))

    vals_sp = sp.interpolate_old(dom.grid)
    #vals_sp = sp(dom.grid)


    print(sp.grid.shape)
    print(sg.grid.shape)

    error_sp = abs(vals_sp - vals).max()
    error_sg = abs(vals_sg - vals).max()

    print('Errors (splines)  : {}'.format(error_sp))
    print('Errors (smolyak)  : {}'.format(error_sg))










    [xg, yg] = meshgrid(  linspace(1.1,2, 50),  linspace(1.1,2, 50))


    fine_grid = row_stack([xg.flatten(), yg.flatten()])

    true_vals = fun(fine_grid)
    interp_vals = sp.interpolate(fine_grid)
    interp_old_vals = sp.interpolate_old(fine_grid)

    interp_sg_vals = sg.interpolate(fine_grid)


#    plot(fine_grid.flatten(), true_vals.flatten())
#    plot(fine_grid.flatten(), interp_vals.flatten())
#    show()

    print(true_vals.shape)
    print(xg.shape)

    X = xg
    Y = yg
    Z = (true_vals[0,:]).reshape(X.shape)
    Za = (interp_old_vals[0,:]).reshape(X.shape)
    Zb = (interp_sg_vals[0,:]).reshape(X.shape)
    Zc = (interp_vals[0,:]).reshape(X.shape)


    print(sg.grid.shape)
    print(sp.grid.shape)

    from mpl_toolkits.mplot3d import Axes3D
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
