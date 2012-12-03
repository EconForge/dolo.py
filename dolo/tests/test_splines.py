from multilinear_c import multilinear_c

if __name__ == '__main__':

    from multilinear_c import multilinear_c


    N_grid = 20
    N_fine_grid = 50
    d = 1

    smin = np.array( [0.0]*d )
    smax = np.array( [1.0]*d )
    orders = [N_grid]*d
    orders = orders[:d]
    from itertools import product
    grid = np.row_stack( product( *[np.linspace(smin[i],smax[i], orders[i]) for i in range(d)] )).T
    points = np.row_stack( product( *[np.linspace(smin[i]-0.1,smax[i]+0.1, N_fine_grid) for i in range(d)] )).T


    if d == 1:
        f = lambda x: np.row_stack( [np.cos(x)] )
        #        f = lambda x: np.row_stack( [np.power(0.1+x,1.0/4.0)] )
        #f = lambda x: np.row_stack( [x*x] )
        v = f(grid[0,:])
    elif d == 2:
        f = lambda x,y: np.row_stack( [np.sin(x)*y, np.sin(y)*x] )
        v = f(grid[0,:], grid[1,:])
    elif d == 3:
        f = lambda x,y,z: np.row_stack( [np.sin(x)*y/np.sqrt(z)] )
        f = lambda x,y,z: np.row_stack( [2*x + 3*y + 4*z] )
        v = f(grid[0,:], grid[1,:], grid[2,:])

    elif d == 4:
        f = lambda x,y,z,t: np.row_stack( [np.sin(x)*y/np.sqrt(z)*t] )
        v = f(grid[0,:], grid[1,:], grid[2,:], grid[3,:])

    #points = np.row_stack( product( *[np.linspace(smin[i],smax[i], N_fine_grid) for i in range(d)] )).T
    import numpy.random

    #    v = numpy.random.random((2,grid.shape[1]))
    #    points = numpy.random.random((d,N_fine_grid)) + np.array([1]*d)[:,None]
    #
    #



    #    out = spline_c(smin,smax,orders, v, points)
    true_vals = f(*[p for i,p in enumerate(points) if i <= d] )
    #    print( abs(out-true_vals).max() )
    #    exit()

    from matplotlib import pyplot
    from dolo.numeric.interpolation.smolyak import SmolyakGrid

    mvs = MultivariateSpline(smin, smax, orders)
    mvs2 = MultivariateSpline(smin, smax, orders)
    mvs.set_values( f( *[s for i,s in enumerate(mvs.grid) if i <= d] ) )
    mvs2.set_values( f( *[s for i,s in enumerate(mvs.grid) if i <= d] ) )

    sg = SmolyakGrid( smin, smax, 4)
    sg.set_values( f( *[s for i,s in enumerate(sg.grid) if i <= d] ) )


    import time
    t = time.time()
    for i in range(10):
    #out = mvs(points)
        out = spline_c(smin,smax,orders,mvs.values, points)
    s = time.time()
    print('Splines : {}'.format(s-t))

    import time
    t = time.time()
    for i in range(10):
    #out = mvs(points)
        [out2, dout2] = spline_c(smin,smax,orders,mvs2.values, points, derivatives=True)
    s = time.time()
    print('Splines (diff) : {}'.format(s-t))

    print(dout2[0,:])

    import time
    t = time.time()
    for i in range(10):
        v_sg = sg(np.atleast_2d( points) )[0,:]
        v_sg = sg(np.atleast_2d( points) )[0,:]
    s = time.time()
    print('Chebychev : {}'.format(s-t))
    print( abs(out-true_vals).mean(axis=1) )
    print( abs(v_sg-true_vals).mean(axis=1) )
    print( abs(out-true_vals).max(axis=1) )
    print( abs(v_sg-true_vals).max(axis=1) )



    out = spline_c(smin,smax,orders,v, points)
    out_e = mvs(points)
    print(out)

    from matplotlib import pyplot
    points = points.flatten()


    pyplot.plot(points, out_e.flatten(), label='splines (with extrap)')

    pyplot.plot(points, out.flatten(), label='splines')

    pyplot.plot(points, f(points)[0,:], label='true')
    pyplot.plot(grid[0,:],v.flatten(),'o',label='data')
    pyplot.plot(points, v_sg,label='cheb')
    pyplot.legend()

    pyplot.figure()
    pyplot.plot(points, (out-f(points))[0,:], label='splines')

    from dolo.numeric.interpolation.smolyak import SmolyakGrid

    print(grid.shape)
    sg = SmolyakGrid( np.array([0.0]), [1.0], 4 )
    sg.set_values( f(sg.grid) )
    v_sg = sg(np.atleast_2d( points) )[0,:]
    pyplot.plot(points, v_sg-f(points)[0,:],label='cheb')
    pyplot.legend()
    pyplot.show()

    exit()
#    print(points)

