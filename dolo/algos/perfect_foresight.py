import numpy
from numpy import linspace, zeros, atleast_2d

from dolo.algos.steady_state import find_deterministic_equilibrium 

def deterministic_solve(model, shocks=None, start_states=None, start_constraints=None, T=100, ignore_constraints=False, maxit=100, use_pandas=True, initial_guess=None, verbose=False, tol=None):
    '''
    Computes a perfect foresight simulation using a stacked-time algorithm.

    :param model: an "fga" model
    :param shocks: a :math:`n_e\\times N` matrix containing :math:`N` realizations of the shocks. :math:`N` must be smaller than :math:`T`.    The exogenous process is assumed to remain constant and equal to its last value after `N` periods.
    :param T: the horizon for the perfect foresight simulation
    :param use_pandas: if True, returns a pandas dataframe, else the simulation matrix
    :param ignore_constraints: if True, complementarity constraintes are ignored.
    :return: a dataframe with T+1 observations of the model variables along the simulation (states, controls, auxiliaries). The first observation is the steady-state corresponding to the first value of the shocks. The simulation should return
    to a steady-state corresponding to the last value of the exogenous shocks.
    '''

    # TODO:

    if model.model_type == 'fga':
        from dolo.compiler.converter import GModel_fg_from_fga
        model = GModel_fg_from_fga(model)

    # definitions
    n_s = len(model.calibration['states'])
    n_x = len(model.calibration['controls'])

    if shocks == None:
        shocks = numpy.zeros( (len(model.calibration['shocks']),1))

    # until last period, exogenous shock takes its last value
    epsilons = numpy.zeros( (shocks.shape[0], T))
    epsilons[:,:(shocks.shape[1]-1)] = shocks[:,1:]
    epsilons[:,(shocks.shape[1]-1):] = shocks[:,-1:]
    
    # final initial and final steady-states consistent with exogenous shocks
    if isinstance(start_states,dict):
        # at least that part is clear
        start_equilibrium = start_states
        start_s = start_equilibrium['states']
        start_x = start_equilibrium['controls']
        final_s = start_equilibrium['states']
        final_x = start_equilibrium['controls']
    else:
        raise Exception("You must compute initial calibration yourself")
#        final_dict = {model.symbols['shocks'][i]: shocks[i,-1] for i in range(len(model.symbols['shocks']))}
#        start_dict = {model.symbols['shocks'][i]: shocks[i,0] for i in range(len(model.symbols['shocks']))}
#        start_equilibrium = find_deterministic_equilibrium( model, constraints=start_dict)

#        if start_constraints:
#        ### we ignore start_constraints
#            start_dict.update(start_constraints)
#            final_equilibrium = start_constraints.copy()
#        else:
#        final_equilibrium = find_deterministic_equilibrium( model, constraints=final_dict)
#        final_s = final_equilibrium['states']
#        final_x = final_equilibrium['controls']


#        start_s = start_states
#        start_x = final_x

    #TODO: for start_x, it should be possible to use first order guess


    

    final = numpy.concatenate( [final_s, final_x] )
    start = numpy.concatenate( [start_s, start_x] )

    if verbose==True:
        print("Initial states : {}".format(start_s))
        print("Final controls : {}".format(final_x))

    p = model.calibration['parameters']

    if initial_guess is None:

        initial_guess = numpy.concatenate( [start*(1-l) + final*l for l in linspace(0.0,1.0,T+1)] )
        initial_guess = initial_guess.reshape( (-1, n_s + n_x)).T

    else:
        from pandas import DataFrame
        if isinstance( initial_guess, DataFrame ):
            initial_guess = array( initial_guess ).T.copy()
        initial_guess = initial_guess[:n_s+n_x,:]
        initial_guess[:n_s,0] = start_s
        initial_guess[n_s:,-1] = final_x

    sh = initial_guess.shape

    if model.x_bounds and not ignore_constraints:
        initial_states = initial_guess[:n_s,:]
        [lb, ub] = [ u( initial_states, p ) for u in model.x_bounds]
        lower_bound = initial_guess*0 - numpy.inf
        lower_bound[n_s:,:] = lb
        upper_bound = initial_guess*0 + numpy.inf
        upper_bound[n_s:,:] = ub
        test1 = max( lb.max(axis=1) - lb.min(axis=1) )
        test2 = max( ub.max(axis=1) - ub.min(axis=1) )
        if test1 >0.00000001 or test2>0.00000001:
            raise Exception("Not implemented: perfect foresight solution requires that controls have constant bounds.")
    else:
        ignore_constraints=True
        lower_bound = None
        upper_bound = None


    nn = sh[0]*sh[1]

    fobj  = lambda vec: det_residual(model, vec.reshape(sh), start_s, final_x, epsilons)[0].ravel()

    from dolo.numeric.solver import solver

    if not ignore_constraints:
        dfobj = lambda vec: det_residual( model, vec.reshape(sh), start_s, final_x, epsilons)[1]
        from dolo.numeric.ncpsolve import ncpsolve
        ff  = lambda vec: det_residual(model, vec.reshape(sh), start_s, final_x, epsilons)
        sol = ncpsolve(ff, lower_bound.ravel(), upper_bound.ravel(), initial_guess.ravel(), verbose=verbose, maxit=maxit, tol=tol )
        if isinstance(sol, list):
            raise Exception("No convergence after {} iterations.".format(sol[1]))
#        sol = solver(fobj, initial_guess.ravel(), lb=lower_bound.ravel(), ub=upper_bound.ravel(), jac=dfobj, method='ncpsolve' )
        sol = sol.reshape(sh)
        #sol = solver(fobj, initial_guess, jac=dfobj, lb=lower_bound, ub=upper_bound, method='ncpsolve', serial_problem=False, verbose=verbose )
    else:
        from scipy.optimize import fsolve
        dfobj = lambda vec: det_residual( model, vec.reshape(sh), start_s, final_x, epsilons)[1].toarray()
        x0 = initial_guess.ravel()
        sol = fsolve(fobj, x0, fprime=dfobj)
        sol = sol.reshape(sh)

#        sol = solver(fobj, initial_guess, jac=dfobj, method='fsolve', serial_problem=False, verbose=verbose )

    if use_pandas:
        import pandas
        if 'auxiliary' in model.functions:
            colnames = model.symbols['states'] + model.symbols['controls'] + model.symbols['auxiliary']
            # compute auxiliaries
            y = model.functions['auxiliary'](sol[:n_s,:], sol[n_s:,:], p)
            sol = numpy.row_stack([sol,y])
        else:
            colnames = model.symbols['states'] + model.symbols['controls']

        ts = pandas.DataFrame(sol.T, columns=colnames)
        return ts
    else:
        return sol


def det_residual(model, guess, start, final, shocks, diff=True):
    '''
    Computes the residuals, the derivatives of the stacked-time system.
    :param model: an fga model
    :param guess: the guess for the simulated values. An `(n_s.n_x) x N` array, where n_s is the number of states,
    n_x the number of controls, and `N` the length of the simulation.
    :param start: initial boundary condition (initial value of the states)
    :param final: final boundary condition (last value of the controls)
    :param shocks: values for the exogenous shocks
    :param diff: if True, the derivatives are computes
    :return: a list with two elements:
        - an `(n_s.n_x) x N` array with the residuals of the system
        - a `(n_s.n_x) x N x (n_s.n_x) x N` array representing the jacobian of the system
    '''

    # TODO: compute a sparse derivative and ensure the solvers can deal with it

    n_s = len( model.symbols['states'] )
    n_x = len( model.symbols['controls'] )

    n_e = len( model.symbols['shocks'] )
    N = guess.shape[1]

    p = model.calibration['parameters']

    f = model.functions['arbitrage']
    g = model.functions['transition']

    vec = guess[:,:-1]
    vec_f = guess[:,1:]

    s = vec[:n_s,:]
    x = vec[n_s:,:]
    S = vec_f[:n_s,:]
    X = vec_f[n_s:,:]

    if diff:
        SS, SS_s, SS_x, SS_e = g(s,x,shocks,p, derivs=True)
        R, R_s, R_x, R_S, R_X = f(s,x,S,X,p,derivs=True)
    else:
        SS = g(s,x,shocks,p)
        R = f(s,x,S,X,p)

    res_s = SS - S
    res_x = R

    res = numpy.zeros( (n_s+n_x, N) )

    res[:n_s,1:] = res_s
    res[n_s:,:-1] = res_x

    res[:n_s,0] = - (guess[:n_s,0] - start)
    res[n_s:, -1] = - (guess[n_s:, -1] - guess[n_s:, -2] )

    if not diff:
        return res
    else:

        sparse_jac=False
        if not sparse_jac:

            # we compute the derivative matrix

            from dolo.numeric.serial_operations import serial_multiplication as smult
            res_s_s = SS_s
            res_s_x = SS_x

            # next block is probably very inefficient
            jac = numpy.zeros( (n_s+n_x, N, n_s+n_x, N) )
            for i in range(N-1):
                jac[n_s:,i,:n_s,i] = R_s[:,:,i]
                jac[n_s:,i,n_s:,i] = R_x[:,:,i]
                jac[n_s:,i,:n_s,i+1] = R_S[:,:,i]
                jac[n_s:,i,n_s:,i+1] = R_X[:,:,i]
                jac[:n_s,i+1,:n_s,i] = SS_s[:,:,i]
                jac[:n_s,i+1,n_s:,i] = SS_x[:,:,i]
                jac[:n_s,i+1,:n_s,i+1] = -numpy.eye(n_s)
            jac[:n_s,0,:n_s,0] = - numpy.eye(n_s)
            jac[n_s:,-1,n_s:,-1] = - numpy.eye(n_x)
            jac[n_s:,-1,n_s:,-2] = + numpy.eye(n_x)
            nn = jac.shape[0]*jac.shape[1]
            res = res.ravel()
            jac = jac.reshape((nn,nn))


        from scipy.sparse import csc_matrix, csr_matrix



        jac = csr_matrix(jac)

        return [res,jac]


if __name__ == '__main__':

    # this example computes the response of the rbc economy to a series of expected productivity shocks.
    # investment is bounded by an exogenous value 0.2, so that investment is constrained in the first periods

    # TODO: propose a meaningful economic example

    from dolo import *
    from pylab import *

    model = yaml_import('examples/global_models/rbc_pf.yaml')

    e_z = atleast_2d( linspace(0.1, 0.0, 10) )

    start_s = numpy.zeros(2) * numpy.nan
    start_s[0] = 1.5

    import time
    t1 = time.time()
    #sol1 = deterministic_solve(model, shocks=e_z, T=50, use_pandas=True, ignore_constraints=True, start_s=start_s)

    #sol1 = deterministic_solve(model, T=50, use_pandas=True, ignore_constraints=True, start_s=start_s)

    sol2 = deterministic_solve(model, shocks=e_z, T=1000, use_pandas=True, ignore_constraints=False)
    t2 = time.time()

    print("Elapsed : {}".format(t2-t1))

    exit()
    from pylab import *

    subplot(211)
    plot(sol1['k'], label='k')
    plot(sol1['z'], label='z')
    plot(sol1['i'], label='i')


    subplot(212)
    plot(sol2['k'], label='k')
    plot(sol2['z'], label='z')
    plot(sol2['i'], label='i')
    plot(sol2['i']*0 + sol2['i'].max(), linestyle='--', color='black')

    legend()
    show()

