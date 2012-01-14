import numpy

def simulate(gc, dr, s0, sigma, n_exp, horizon, parms, seed=1, discard=True):

    '''simulates series for a compiled model'''

    if n_exp ==0:
        irf = True
        n_exp = 1
    else:
        irf = False

    g = gc.g

    x0 = dr(s0)
    a0 = gc.a(s0,x0, parms, derivs=False)[0]

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

        a = gc.a(s,x,parms,derivs=False)[0]
        a_simul[:,:,i] = a
        ss = g(s,x,epsilons,parms,derivs=False)[0]

        #for p in range(s0.shape[0]):
        #    ss[p,:] = numpy.maximum( numpy.minimum(ss[p,:], large_bounds[1,p]), large_bounds[0,p])

        if i<(horizon-1):
            s_simul[:,:,i+1] = ss
    from numpy import any,isnan,all

    if discard:
        iA = -isnan(s_simul)
        valid = all( all( iA, axis=0 ), axis=1 )
        [s_simul, x_simul, a_simul] = [ e[:,valid,:] for e in [s_simul, x_simul, a_simul] ]

    return [s_simul, x_simul, a_simul]


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
