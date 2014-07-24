import numpy as np

from dolo.numeric.decision_rules import *

def impulse_response_function(decision_rule, shock, variables = None, horizon=40, order=1, output='deviations', plot=False):

    model = decision_rule.model


    if order > 1:
        raise Exception('irfs, for order > 1 not implemented')

    dr = decision_rule
    A = dr['g_a']
    B = dr['g_e']
    Sigma = dr['Sigma']

    [n_v, n_s] = [ len(model.symbols['variables']), len(model.symbols['shocks']) ]


    if isinstance(shock,int):
        i_s = shock
    elif isinstance(shock,str):
        i_s = model.symbols['shocks'].index( shock )
    else:
        raise Exception("Unknown type for shock")

    E0 = np.zeros(  n_s )
    E0[i_s] = np.sqrt(dr['Sigma'][i_s,i_s])

    E = E0*0

    # RSS = dr.risky_ss()
    RSS = dr['ys']

    simul = np.zeros( (n_v, horizon+1) )

    start = dr( RSS, E0 )
    simul[:,0] = start
    for i in range(horizon):
        simul[:,i+1] = dr( simul[:,i], E )

    # TODO: change the correction so that it corresponds to the risky steady-state
    constant = np.tile(RSS, ( horizon+1, 1) ).T
    if output == 'deviations':
        simul = simul - constant
    elif output == 'logs':
        simul = np.log((simul-constant)/constant)
    elif output == 'percentages':
        simul = (simul-constant)/constant*100
    elif output == 'levels':
        pass
    else:
        raise Exception("Unknown output type")


    if variables:
        variables =  [(str(v)) for v in variables]
        ind_vars = [model.symbols['variables'].index( v ) for v in variables]
        simul = simul[ind_vars, :]
    else:
        variables = model.symbols['variables']


    x = np.linspace(0,horizon,horizon+1)

    if plot:
        from matplotlib import pylab
        pylab.clf()
        # pylab.title('Impulse-Response Function for ${var}$'.format(var=shock.__latex__()))
        pylab.title('Impulse-Response Function for ${var}$'.format(var=shock))
        for k in range(len(variables)):
            # pylab.plot(x[1:], simul[k,1:],label='$'+variables[k]._latex_()+'$' )
            pylab.plot(x[1:], simul[k,1:],label=variables[k] )
        pylab.plot(x,x*0,'--',color='black')
        pylab.xlabel('$t$')
        if output == 'percentages':
            pylab.ylabel('% deviations from the steady-state')
        elif output == 'deviations':
            pylab.ylabel('Deviations from the steady-state')
        elif output == 'levels':
            pylab.ylabel('Levels')
        pylab.legend()
        # if dolo.config.save_plots:
            # filename = 'irf_' + str(shock) + '__' + '_' + str.join('_',[str(v) for v in variables])
            # pylab.savefig(filename) # not good...
        # else:
        pylab.show()

    import pandas
    sim = pandas.DataFrame(simul.T, columns=variables, index=range(horizon+1))
    return sim

def stoch_simul(decision_rule, variables = None,  horizon=40, order=None, start=None, output='deviations', plot=False, seed=None):

#    if order > 1:
#        raise Exception('irfs, for order > 1 not implemented')

    dr = decision_rule
    model = dr.model

    [n_v, n_s] = [ len(model.symbols['variables']), len(model.symbols['shocks']) ]

    Sigma = dr['Sigma']
    if seed:
        np.random.seed(seed)

    E = np.random.multivariate_normal((0,)*n_s,Sigma,horizon)
    E = E.T

    simul = np.zeros( (n_v, horizon+1) )
    RSS = dr.risky_ss()
    if start is None:
        start = RSS

    if not order:
        order = dr.order

    simul[:,0] = start
    for i in range(horizon):
        simul[:,i+1] = dr( simul[:,i], E[:,i] )

    # TODO: change the correction so that it corresponds to the risky steady-state
    constant = np.tile(RSS, ( horizon+1, 1) ).T
    if output == 'deviations':
        simul = simul - constant
    elif output == 'logs':
        simul = np.log((simul-constant)/constant)
    elif output == 'percentages':
        simul = (simul-constant)/constant*100
    elif output == 'levels':
        pass
    else:
        raise Exception("Unknown output type")

    if variables:
        variables =  [(str(v)) for v in model.symbols['variables']]
        ind_vars = [variables.index( v ) for v in variables]
        simul = simul[ind_vars, :]
    else:
        variables = model.symbols['variables']

    x = np.linspace(0,horizon,horizon+1)

    if plot:
        from matplotlib import pylab
        pylab.clf()
        pylab.title('Stochastic simulation')
        for k in range(len(variables)):
            # pylab.plot(x[1:], simul[k,1:],label='$'+variables[k]._latex_()+'$' )
            pylab.plot(x[1:], simul[k,1:],label=variables[k] )
        pylab.plot(x,x*0,'--',color='black')
        pylab.xlabel('$t$')
        if output == 'percentages':
            pylab.ylabel('% deviations from the steady-state')
        elif output == 'deviations':
            pylab.ylabel('Deviations from the steady-state')
        elif output == 'levels':
            pylab.ylabel('Levels')
        pylab.legend()

        pylab.show()

    import pandas
    sim = pandas.DataFrame(simul.T, columns=variables, index=range(horizon+1))
    return sim
