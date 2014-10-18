import numpy as np

# from dolo.numeric.decision_rules import *

# def impulse_response_function(decision_rule, shock, variables = None, horizon=40, order=1, output='deviations', plot=False):
#
#     model = decision_rule.model
#
#
#     if order > 1:
#         raise Exception('irfs, for order > 1 not implemented')
#
#     dr = decision_rule
#     A = dr['g_a']
#     B = dr['g_e']
#     Sigma = dr['Sigma']
#
#     [n_v, n_s] = [ len(model.symbols['variables']), len(model.symbols['shocks']) ]
#
#
#     if isinstance(shock,int):
#         i_s = shock
#     elif isinstance(shock,str):
#         i_s = model.symbols['shocks'].index( shock )
#     else:
#         raise Exception("Unknown type for shock")
#
#     E0 = np.zeros(  n_s )
#     E0[i_s] = np.sqrt(dr['Sigma'][i_s,i_s])
#
#     E = E0*0
#
#     # RSS = dr.risky_ss()
#     RSS = dr['ys']
#
#     simul = np.zeros( (n_v, horizon+1) )
#
#     start = dr( RSS, E0 )
#     simul[:,0] = start
#     for i in range(horizon):
#         simul[:,i+1] = dr( simul[:,i], E )
#
#     # TODO: change the correction so that it corresponds to the risky steady-state
#     constant = np.tile(RSS, ( horizon+1, 1) ).T
#     if output == 'deviations':
#         simul = simul - constant
#     elif output == 'logs':
#         simul = np.log((simul-constant)/constant)
#     elif output == 'percentages':
#         simul = (simul-constant)/constant*100
#     elif output == 'levels':
#         pass
#     else:
#         raise Exception("Unknown output type")
#
#
#     if variables:
#         variables =  [(str(v)) for v in variables]
#         ind_vars = [model.symbols['variables'].index( v ) for v in variables]
#         simul = simul[ind_vars, :]
#     else:
#         variables = model.symbols['variables']
#
#
#     x = np.linspace(0,horizon,horizon+1)
#
#     if plot:
#         from matplotlib import pylab
#         pylab.clf()
#         # pylab.title('Impulse-Response Function for ${var}$'.format(var=shock.__latex__()))
#         pylab.title('Impulse-Response Function for ${var}$'.format(var=shock))
#         for k in range(len(variables)):
#             # pylab.plot(x[1:], simul[k,1:],label='$'+variables[k]._latex_()+'$' )
#             pylab.plot(x[1:], simul[k,1:],label=variables[k] )
#         pylab.plot(x,x*0,'--',color='black')
#         pylab.xlabel('$t$')
#         if output == 'percentages':
#             pylab.ylabel('% deviations from the steady-state')
#         elif output == 'deviations':
#             pylab.ylabel('Deviations from the steady-state')
#         elif output == 'levels':
#             pylab.ylabel('Levels')
#         pylab.legend()
#         # if dolo.config.save_plots:
#             # filename = 'irf_' + str(shock) + '__' + '_' + str.join('_',[str(v) for v in variables])
#             # pylab.savefig(filename) # not good...
#         # else:
#         pylab.show()
#
#     import pandas
#     sim = pandas.DataFrame(simul.T, columns=variables, index=range(horizon+1))
#     return sim

def simulate(decision_rule, horizon=40, start=None, shock=None, seed=None, n_exp=0):

    dr = decision_rule
    model = dr.model

    if n_exp==0:
        n_exp = 1
        irf = True
    else:
        irf = False

    [n_v, n_s] = [ len(model.symbols['variables']), len(model.symbols['shocks']) ]

    Sigma = dr['Sigma']
    if seed:
        np.random.seed(seed)

    if irf:
        E = np.zeros((n_exp,horizon+1,n_s))
    else:
        E = np.random.multivariate_normal((0,)*n_s,Sigma,(n_exp,horizon+1))
        E[:,0,:] = 0

    if shock is not None:
        E[:,1,:] = np.array(shock).ravel()[None,:]

    simul = np.zeros( (n_exp,horizon+1, n_v) )

    if start in ('risky', None):
        RSS = dr.risky_ss()
        start = RSS
    elif start == 'deterministic':
        start = dr['ys']

    # TODO make multiple simulations more efficient
    for n in range(n_exp):
        simul[n,0,:] = start
        for i in range(horizon):
            simul[n,i+1,:] = dr( simul[n,i,:], E[n,i+1,:] )

    all_vars = model.symbols['variables'] + model.symbols['shocks']
    sims = np.concatenate([simul, E], axis=2)

    import pandas
    if n_exp == 1:
        sim = pandas.DataFrame(sims[0,:,:], columns=all_vars, index=range(horizon+1))
    else:
        sim = pandas.Panel(sims, items=range(n_exp), major_axis=range(horizon+1), minor_axis=all_vars)

    return sim
