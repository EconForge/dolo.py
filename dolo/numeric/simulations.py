import numpy

def simulate(cmodel, dr, s0, sigma, n_exp=0, horizon=40, parms=None, seed=1, discard=False, stack_series=True):

    '''
    :param cmodel: 
    :param dr:
    :param s0:
    :param sigma:
    :param n_exp:
    :param horizon:
    :param with_auxiliaries:
    :param parms:
    :param seed:
    :param discard:
    :return:
    '''

    cmodel = cmodel.as_type('fg')

    if n_exp ==0:
        irf = True
        n_exp = 1
    else:
        irf = False

    from dolo.symbolic.model import Model
    if isinstance(cmodel,Model):
        from dolo.compiler.compiler_global import GlobalCompiler
        model = cmodel
        cmodel = GlobalCompiler(model)
        [y,x,parms] = model.read_calibration()

    if parms == None:
        parms = cmodel.model.read_calibration()[2]


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

        ss = cmodel.g(s,x,epsilons,parms)

        if i<(horizon-1):
            s_simul[:,:,i+1] = ss

    from numpy import any,isnan,all

    if not hasattr(cmodel,'__a__'): # TODO: find a better test than this
        l = [s_simul, x_simul]
    else:
        n_s = s_simul.shape[0]
        n_x = x_simul.shape[0]
        a_simul = cmodel.a( s_simul.reshape((n_s,n_exp*horizon)), x_simul.reshape( (n_x,n_exp*horizon) ), parms)
        n_a = a_simul.shape[0]
        a_simul = a_simul.reshape(n_a,n_exp,horizon)
        l = [s_simul, x_simul, a_simul]

    if stack_series:
        simul = numpy.row_stack(l)

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

    from dolo.compiler.compiler_global import CModel

    import numpy
    cmodel = CModel(model)
    s0 = numpy.atleast_2d( dr.S_bar ).T
    horizon = 50

    simul = simulate(cmodel, dr, s0, sigma, n_exp=500, parms=parms, horizon=horizon)

    print(simul.shape)
    from matplotlib.pyplot import hist, show, figure


    timevec = numpy.array(range(simul.shape[2]))

    figure()
    for i in range( horizon ):
        hist( simul[0,:,i], bins=50 )

    show()
    #plot(timevec,s_simul[0,0,:])
