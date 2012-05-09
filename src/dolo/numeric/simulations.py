import numpy

def simulate(gc, dr, s0, sigma, n_exp=0, horizon=40, parms=None, seed=1, discard=False):

    '''simulates series for a compiled model'''

    gc = gc.as_type('fga')

    if n_exp ==0:
        irf = True
        n_exp = 1
    else:
        irf = False

#    from dolo.symbolic.model import Model
#    if isinstance(gc,Model):
#        from dolo.compiler.compiler_global import GlobalCompiler2
#        model = gc
#        gc = GlobalCompiler2(model)
#        [y,x,parms] = model.read_calibration()

    if parms == None:
        parms = gc.model.read_calibration()[2]


    s0 = numpy.atleast_2d(s0.flatten()).T

    x0 = dr(s0)
    a0 = gc.a(s0,x0, parms)

    s_simul = numpy.zeros( (s0.shape[0],n_exp,horizon) )
    x_simul = numpy.zeros( (x0.shape[0],n_exp,horizon) )
    a_simul = numpy.zeros( (a0.shape[0],n_exp,horizon) )

    s_simul[:,:,0] = s0
    x_simul[:,:,0] = x0
    a_simul[:,:,0] = a0

    for i in range(horizon):
        mean = numpy.zeros(sigma.shape[0])
        if irf:
            epsilons = numpy.zeros( (sigma.shape[0],1) )
        else:
            seed += 1
            numpy.random.seed(seed)
            epsilons = numpy.random.multivariate_normal(mean, sigma, n_exp).T
        s = s_simul[:,:,i]
        x = dr(s)
        x_simul[:,:,i] = x

        a = gc.a(s,x,parms)

        a_simul[:,:,i] = a

        ss = gc.g(s,x,a,epsilons,parms)

        if i<(horizon-1):
            s_simul[:,:,i+1] = ss
    from numpy import any,isnan,all

    simul = numpy.row_stack( [s_simul, x_simul, a_simul] )

    if irf:
        simul = simul[:,0,:]
        return simul

    if discard:
        iA = -isnan(simul)
        valid = all( all( iA, axis=0 ), axis=1 )
        simul = simul[:,valid,:]
        n_kept = simul.shape[1]
        if n_exp > n_kept:
            print( 'Discarded {}/{}'.format(n_exp-n_kept,n_exp))

    return simul

def simulate_without_aux(gc, dr, s0, sigma, n_exp=0, horizon=40, parms=None, seed=1, discard=False):

    '''simulates series for a compiled model'''

    gc = gc.as_type('fg')

    if n_exp ==0:
        irf = True
        n_exp = 1
    else:
        irf = False

    from dolo.symbolic.model import Model
    if isinstance(gc,Model):
        from dolo.compiler.compiler_global import GlobalCompiler
        model = gc
        gc = GlobalCompiler(model)
        [y,x,parms] = model.read_calibration()

    if parms == None:
        parms = gc.model.read_calibration()[2]


    s0 = numpy.atleast_2d(s0.flatten()).T

    x0 = dr(s0)

    s_simul = numpy.zeros( (s0.shape[0],n_exp,horizon) )
    x_simul = numpy.zeros( (x0.shape[0],n_exp,horizon) )

    s_simul[:,:,0] = s0
    x_simul[:,:,0] = x0

    for i in range(horizon):
        mean = numpy.zeros(sigma.shape[0])
        if irf:
            epsilons = numpy.zeros( (sigma.shape[0],1) )
        else:
            seed += 1
            numpy.random.seed(seed)
            epsilons = numpy.random.multivariate_normal(mean, sigma, n_exp).T
        s = s_simul[:,:,i]
        x = dr(s)
        x_simul[:,:,i] = x

        ss = gc.g(s,x,epsilons,parms)

        if i<(horizon-1):
            s_simul[:,:,i+1] = ss

    from numpy import any,isnan,all

    simul = numpy.row_stack([s_simul, x_simul])

    if discard:
        iA = -isnan(x_simul)
        valid = all( all( iA, axis=0 ), axis=1 )
        simul = simul[:,valid,:]
        n_kept = s_simul.shape[1]
        if n_exp > n_kept:
            print( 'Discarded {}/{}'.format(n_exp-n_kept,n_exp))
    if irf:
        simul = simul[:,0,:]

    return simul


if __name__ == '__main__':
    from dolo import yaml_import, approximate_controls
    model = yaml_import('../../../examples/global_models/capital.yaml')
    dr = approximate_controls(model)

    [y,x,parms] = model.read_calibration()
    sigma = model.read_covariances()

    from dolo.compiler.compiler_global import GlobalCompiler2

    import numpy
    gc = GlobalCompiler2(model)
    s0 = numpy.atleast_2d( dr.S_bar ).T

    [s_simul, x_simul, a_simul] = simulate(gc, dr, s0, sigma, 10000, 50, parms)


    from matplotlib.pyplot import hist, show, figure


    timevec = numpy.array(range(s_simul.shape[2]))

    figure()
    for i in range( 50 ):
        hist( s_simul[0,:,i], bins=50 )
        show()
    #plot(timevec,s_simul[0,0,:])
