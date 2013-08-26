#
# if __name__ == '__main__':
#
#     from dolo.numeric.interpolation.splines import MultivariateSplines
#     from dolo.numeric.interpolation.multilinear import MultilinearInterpolator
#
#
#     import numpy as np
#
#     N_grid = 20
#     N_fine_grid = 50
#     d = 3
#
#     smin = np.array( [0.0]*d )
#     smax = np.array( [1.0]*d )
#     orders = [N_grid]*d
#     orders = orders[:d]
#     from itertools import product
#     grid = np.row_stack( product( *[np.linspace(smin[i],smax[i], orders[i]) for i in range(d)] )).T
#     points = np.row_stack( product( *[np.linspace(smin[i]-0.1,smax[i]+0.1, N_fine_grid) for i in range(d)] )).T
#
#
#     if d == 1:
#         f = lambda x: np.row_stack( [np.cos(x)] )
#         #        f = lambda x: np.row_stack( [np.power(0.1+x,1.0/4.0)] )
#         #f = lambda x: np.row_stack( [x*x] )
#         v = f(grid[0,:])
#     elif d == 2:
# #        f = lambda x,y: np.row_stack( [np.sin(x)*y, np.sin(y)*x] )
#         f = lambda x,y: np.row_stack( [2*x + 3*y] )
#         v = f(grid[0,:], grid[1,:])
#     elif d == 3:
#         f = lambda x,y,z: np.row_stack( [np.sin(x)*y/np.sqrt(z)] )
#         f = lambda x,y,z: np.row_stack( [2*x + 3*y + 4*z] )
#         v = f(grid[0,:], grid[1,:], grid[2,:])
#
#     elif d == 4:
#         f = lambda x,y,z,t: np.row_stack( [np.sin(x)*y/np.sqrt(z)*t] )
#         v = f(grid[0,:], grid[1,:], grid[2,:], grid[3,:])
# #
#     points = np.row_stack( product( *[np.linspace(smin[i],smax[i], N_fine_grid) for i in range(d)] )).T
#
#     points = np.ascontiguousarray(points)  # TODO : this should be checked in the interpolators
#
#     mvs = MultivariateSplines(smin, smax, orders)
#     mvs.set_values( f( *[s for i,s in enumerate(mvs.grid) if i <= d] ) )
#
#     mls = MultilinearInterpolator(smin, smax, orders)
#     mls.set_values( f( *[s for i,s in enumerate(mls.grid) if i <= d] ) )
#
# #    print(mls.grid -  mvs.grid)
# #    exit()
#
#     import time
#     t = time.time()
#     for i in range(10):
#         out = mvs(points)
#     s = time.time()
#     print('Splines : {}'.format(s-t))
#
#
#     import time
#     t = time.time()
#     for i in range(10):
#         out2 = mls(points)
#     s = time.time()
#     print('Linear : {}'.format(s-t))
#
#
#     print(out2)
#
#     print( abs(out2 - out).max() )#    print(mls.grid -  mvs.grid)
# #    exit()
#
#     mvs(points)
# #
# #    import time
# #    t = time.time()
# #    for i in range(10):
# #    #out = mvs(points)
# #        [out2, dout2] = spline_c(smin,smax,orders,mvs2.values, points, derivatives=True)
# #    s = time.time()
# #    print('Splines (diff) : {}'.format(s-t))
# #
# #    print(dout2[0,:])
# #
# #    import time
# #    t = time.time()
# #    for i in range(10):
# #        v_sg = sg(np.atleast_2d( points) )[0,:]
# #        v_sg = sg(np.atleast_2d( points) )[0,:]
# #    s = time.time()
# #    print('Chebychev : {}'.format(s-t))
# #    print( abs(out-true_vals).mean(axis=1) )
# #    print( abs(v_sg-true_vals).mean(axis=1) )
# #    print( abs(out-true_vals).max(axis=1) )
# #    print( abs(v_sg-true_vals).max(axis=1) )
# #
# #
# #
# #    out = spline_c(smin,smax,orders,v, points)
# #    out_e = mvs(points)
# #    print(out)
# #
# #    from matplotlib import pyplot
# #    points = points.flatten()
# #
# #
# #    pyplot.plot(points, out_e.flatten(), label='splines (with extrap)')
# #
# #    pyplot.plot(points, out.flatten(), label='splines')
# #
# #    pyplot.plot(points, f(points)[0,:], label='true')
# #    pyplot.plot(grid[0,:],v.flatten(),'o',label='data')
# #    pyplot.plot(points, v_sg,label='cheb')
# #    pyplot.legend()
# #
# #    pyplot.figure()
# #    pyplot.plot(points, (out-f(points))[0,:], label='splines')
# #
# #    from dolo.numeric.interpolation.smolyak import SmolyakGrid
# #
# #    print(grid.shape)
# #    sg = SmolyakGrid( np.array([0.0]), [1.0], 4 )
# #    sg.set_values( f(sg.grid) )
# #    v_sg = sg(np.atleast_2d( points) )[0,:]
# #    pyplot.plot(points, v_sg-f(points)[0,:],label='cheb')
# #    pyplot.legend()
# #    pyplot.show()
# #
# #    exit()
# ##    print(points)
# #
