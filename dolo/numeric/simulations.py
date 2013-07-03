import numpy
import numpy as np
from dolo.numeric.global_solution import step_residual


def simulate(cmodel, dr, s0=None, sigma=None, n_exp=0, horizon=40, parms=None, seed=1, discard=False, stack_series=True,
             solve_expectations=False, nodes=None, weights=None, use_pandas=True):

    '''
    :param cmodel: compiled model
    :param dr: decision rule
    :param s0: initial state where all simulations start
    :param sigma: covariance matrix of the normal multivariate distribution describing the random shocks
    :param n_exp: number of draws to simulate. Use 0 for impulse-response functions
    :param horizon: horizon for the simulation
    :param parms: (vector) value for the parameters of the model
    :param seed: used to initialize the random number generator. Use it to replicate exact same results among simulations
    :param discard: (default: False) if True, then all simulations containing at least one non finite value are discarded
    :param stack_series: return simulated series for different types of variables separately (in a list)
    :return: a ``n_s x n_exp x horizon`` array where ``n_s`` is the number of states. The second dimension is omitted if
    n_exp = 0.
    '''

    # from dolo.compiler.compiler_global import CModel_
    from dolo.compiler.compiler_python import GModel
    from dolo.symbolic.model import SModel

    if isinstance(cmodel, GModel):
        # cmodel = CModel(cmodel.model)
        cmodel = cmodel
    elif isinstance(cmodel, SModel):
        model = cmodel
        cmodel = CModel(model)



    if n_exp ==0:
        irf = True
        n_exp = 1
    else:
        irf = False



    calib = cmodel.calibration

    if parms is None:
        parms = numpy.array( calib['parameters'] ) # TODO : remove reference to symbolic model

    if sigma is None:
        sigma = numpy.array( calib['covariances'] )

    if s0 is None:
        s0 = numpy.array( calib['states'] )

    s0 = numpy.atleast_2d(s0.flatten()).T

    x0 = dr(s0)

    s_simul = numpy.zeros( (s0.shape[0],n_exp,horizon) )
    x_simul = numpy.zeros( (x0.shape[0],n_exp,horizon) )

    s_simul[:,:,0] = s0
    x_simul[:,:,0] = x0

    fun = cmodel.functions
    f = fun['arbitrage']
    g = fun['transition']
    aux = fun['auxiliary']

    numpy.random.seed(seed)

    for i in range(horizon):
        mean = numpy.zeros(sigma.shape[0])
        if irf:
            epsilons = numpy.zeros( (sigma.shape[0],1) )
        else:
            epsilons = numpy.random.multivariate_normal(mean, sigma, n_exp).T
        s = s_simul[:,:,i]

        x = dr(s)

        if solve_expectations:
            from dolo.numeric.solver import solver
            from dolo.numeric.newton import newton_solver

            fobj = lambda t: step_residual(s, t, dr, f, g, aux, parms, nodes, weights, with_derivatives=False) #
#            x = solver(fobj, x,  serial_problem=True)
            x = newton_solver(fobj, x, numdiff=True)

        x_simul[:,:,i] = x

        a = aux(s,x,parms)

        ss = g(s,x,a,epsilons,parms)

        if i<(horizon-1):
            s_simul[:,:,i+1] = ss

    from numpy import any,isnan,all

    if not 'auxiliary' in fun: # TODO: find a better test than this
        l = [s_simul, x_simul]
        varnames = cmodel.symbols['states'] = cmodel.symbols['controls']
    else:
        n_s = s_simul.shape[0]
        n_x = x_simul.shape[0]
        a_simul = aux( s_simul.reshape((n_s,n_exp*horizon)), x_simul.reshape( (n_x,n_exp*horizon) ), parms)
        n_a = a_simul.shape[0]
        a_simul = a_simul.reshape(n_a,n_exp,horizon)
        l = [s_simul, x_simul, a_simul]
        varnames = cmodel.symbols['states'] + cmodel.symbols['controls'] + cmodel.symbols['auxiliary']
    if not stack_series:
        return l

    else:
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

        if use_pandas:
            import pandas
            ts = pandas.DataFrame(simul.T, columns=varnames)
            return ts

    return simul


def plot_decision_rule(model, dr, state, plot_controls=None, bounds=None, n_steps=10, s0=None, **kwargs):

    import numpy

    states_names = [str(s) for s in model.symbols['states']]
    controls_names = [str(s) for s in model.symbols['controls']]
    index = states_names.index(str(state))
    if bounds is None:
        bounds = [dr.smin[index], dr.smax[index]]
    values = numpy.linspace(bounds[0], bounds[1], n_steps)
    if s0 is None:
        s0 = model.calibration['states']
    svec = numpy.column_stack([s0]*n_steps)
    svec[index,:] = values
    xvec = dr(svec)

    if plot_controls is None:
        return [svec, xvec]
    else:
        from matplotlib import pyplot
        if isinstance(plot_controls, str):
            i = controls_names.index(plot_controls)
            pyplot.plot(values, xvec[i,:], **kwargs)
        else:
            for cn in  plot_controls:
                i = controls_names.index(cn)
                pyplot.plot(values, xvec[i,:], label=cn)
            pyplot.legend()
        pyplot.xlabel(state)



if __name__ == '__main__':
    from dolo import yaml_import, approximate_controls
    model = yaml_import('../../examples/global_models/capital.yaml')

    dr = approximate_controls(model)

    [y,x,parms] = model.read_calibration()
    sigma = model.read_covariances()

    from dolo.compiler.compiler_global import CModel

    import numpy
    cmodel = CModel(model)
    s0 = numpy.atleast_2d( dr.S_bar ).T
    horizon = 50

    simul = simulate(cmodel, dr, s0, sigma, n_exp=500, parms=parms, seed=1, horizon=horizon)



    from dolo.numeric.quantization import quantization_nodes
    N = 80
    [x,w] = quantization_nodes(N, sigma)
    simul_2 = simulate(cmodel, dr, s0, sigma, n_exp=500, parms=parms, horizon=horizon, seed=1, solve_expectations=True, nodes=x, weights=w)

    from matplotlib.pyplot import hist, show, figure, plot


    timevec = numpy.array(range(simul.shape[2]))


    figure()
    plot(simul[0,0,:] - simul_2[0,0,:])
    show()

#
#    figure()
#    plot(simul[0,0,:])
#    plot(simul_2[0,0,:])
#    show()

    figure()
    for i in range( horizon ):
        hist( simul[0,:,i], bins=50 )

    show()
    #plot(timevec,s_simul[0,0,:])
