import numpy
from numpy import linspace, zeros, atleast_2d

from dolo.algos.fg.steady_state import find_deterministic_equilibrium 

def deterministic_solve(model, shocks=None, start_states=None, T=100, ignore_constraints=False, maxit=100, initial_guess=None, verbose=False, tol=1e-6):
    '''
    Computes a perfect foresight simulation using a stacked-time algorithm.

    The initial state is specified either by providing a series of exogenous shocks and assuming the model is initially
    in equilibrium with the first value of the shock, or by specifying an initial value for the states.

    Parameters
    ----------

    model: NumericModel
        "fg" or "fga" model to be solved

    shocks: ndarray
        :math:`n_e\\times N` matrix containing :math:`N` realizations of the shocks. :math:`N` must be smaller than :math:`T`.    The exogenous process is assumed to remain constant and equal to its last value after `N` periods.

    start_states: ndarray or dict
        a vector with the value of initial states, or a calibration dictionary with the initial values of states and controls

    T: int
        horizon for the perfect foresight simulation

    maxit: int
        maximum number of iteration for the nonlinear solver

    verbose: boolean
        if True, the solver displays iterations

    tol: float
        stopping criterium for the nonlinear solver

    ignore_constraints: bool
        if True, complementarity constraints are ignored.

    Returns
    -------
    pandas dataframe:

        a dataframe with T+1 observations of the model variables along the simulation (states, controls, auxiliaries). The first observation is the steady-state corresponding to the first value of the shocks. The simulation should return
        to a steady-state corresponding to the last value of the exogenous shocks.
    '''

    # TODO:

    # if model.model_type == 'fga':
    #     from dolo.compiler.converter import GModel_fg_from_fga
    #     model = GModel_fg_from_fga(model)

    # definitions
    n_s = len(model.calibration['states'])
    n_x = len(model.calibration['controls'])

    if shocks is None:
        shocks = numpy.zeros( (len(model.calibration['shocks']),1))

    # until last period, exogenous shock takes its last value
    epsilons = numpy.zeros( (T+1, shocks.shape[1]))
    epsilons[:(shocks.shape[0]-1),:] = shocks[1:,:]
    epsilons[(shocks.shape[0]-1):,:] = shocks[-1:,:]

    # final initial and final steady-states consistent with exogenous shocks
    if isinstance(start_states,dict):
        # at least that part is clear
        start_equilibrium = start_states
        start_s = start_equilibrium['states']
        start_x = start_equilibrium['controls']
        final_s = start_equilibrium['states']
        final_x = start_equilibrium['controls']
    elif isinstance(start_states, numpy.ndarray):
        start_s = start_states
        start_x = model.calibration['controls']
        final_s = model.calibration['states']
        final_x = model.calibration['controls']
    else:
        # raise Exception("You must compute initial calibration yourself")
        final_dict = {model.symbols['shocks'][i]: shocks[i,-1] for i in range(len(model.symbols['shocks']))}
        start_dict = {model.symbols['shocks'][i]: shocks[i,0] for i in range(len(model.symbols['shocks']))}
        start_calib = find_deterministic_equilibrium( model, constraints=start_dict)
        final_calib = find_deterministic_equilibrium( model, constraints=start_dict)

        start_s = start_calib['states']
        start_x = start_calib['controls']
        final_s = final_calib['states']
        final_x = final_calib['controls']


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

        initial_guess = numpy.row_stack( [start*(1-l) + final*l for l in linspace(0.0,1.0,T+1)] )

    else:
        from pandas import DataFrame
        if isinstance( initial_guess, DataFrame ):
            initial_guess = array( initial_guess ).T.copy()
        initial_guess = initial_guess[:,:n_s+n_x]
        initial_guess[0,:n_s] = start_s
        initial_guess[-1,n_s:] = final_x

    sh = initial_guess.shape

    if model.x_bounds and not ignore_constraints:
        initial_states = initial_guess[:,:n_s]
        [lb, ub] = [ u( initial_states, p ) for u in model.x_bounds]
        lower_bound = initial_guess*0 - numpy.inf
        lower_bound[:, n_s:] = lb
        upper_bound = initial_guess*0 + numpy.inf
        upper_bound[:, n_s:] = ub
        test1 = max( lb.max(axis=0) - lb.min(axis=0) )
        test2 = max( ub.max(axis=0) - ub.min(axis=0) )
        if test1 >0.00000001 or test2>0.00000001:
            raise Exception("Not implemented: perfect foresight solution requires that controls have constant bounds.")
    else:
        ignore_constraints=True
        lower_bound = None
        upper_bound = None


    nn = sh[0]*sh[1]

    fobj  = lambda vec: det_residual(model, vec.reshape(sh), start_s, final_x, epsilons)[0].ravel()



    if not ignore_constraints:

        from dolo.numeric.optimize.ncpsolve import ncpsolve
        ff  = lambda vec: det_residual(model, vec.reshape(sh), start_s, final_x, epsilons, jactype='sparse')

        from dolo.numeric.optimize.newton import newton
        x0 = initial_guess.ravel()

        sol, nit = ncpsolve(ff, lower_bound.ravel(), upper_bound.ravel(), initial_guess.ravel(), verbose=verbose, maxit=maxit, tol=tol, jactype='sparse')
        
        sol = sol.reshape(sh)

    else:

        from scipy.optimize import root
        from numpy import array
        # ff  = lambda vec: det_residual(model, vec.reshape(sh), start_s, final_x, epsilons, jactype='full')
        ff  = lambda vec: det_residual(model, vec.reshape(sh), start_s, final_x, epsilons, diff=False).ravel()
        x0 = initial_guess.ravel()
        sol = root(ff, x0, jac=False)

        res = ff(sol.x)


        sol = sol.x.reshape(sh)

    import pandas
    if 'auxiliary' in model.functions:
        colnames = model.symbols['states'] + model.symbols['controls'] + model.symbols['auxiliaries']
        # compute auxiliaries
        y = model.functions['auxiliary'](sol[:,:n_s], sol[:,n_s:], p)
        sol = numpy.column_stack([sol,y])
    else:
        colnames = model.symbols['states'] + model.symbols['controls']

    sol = numpy.column_stack([sol,epsilons])
    colnames = colnames + model.symbols['shocks']

    ts = pandas.DataFrame(sol, columns=colnames)
    return ts



def det_residual(model, guess, start, final, shocks, diff=True, jactype='sparse'):
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
    N = guess.shape[0]

    p = model.calibration['parameters']

    from dolo.algos.fg.convert import get_fg_functions
    [f,g] = get_fg_functions(model)

    vec = guess[:-1,:]
    vec_f = guess[1:,:]

    s = vec[:,:n_s]
    x = vec[:,n_s:]
    S = vec_f[:,:n_s]
    X = vec_f[:,n_s:]

    e = shocks[:-1,:]
    E = shocks[1:,:]

    if diff:
        SS, SS_s, SS_x, SS_e = g(s,x,e,p, diff=True)
        R, R_s, R_x, R_e, R_S, R_X = f(s,x,E,S,X,p,diff=True)
    else:
        SS = g(s,x,e,p)
        R = f(s,x,E,S,X,p)

    res_s = SS - S
    res_x = R

    res = numpy.zeros( (N, n_s+n_x) )

    res[1:,:n_s] = res_s
    res[:-1,n_s:] = res_x

    res[0,:n_s] = - (guess[0,:n_s] - start)
    res[-1,n_s:] = - (guess[-1,n_s:] - guess[-2,n_s:] )

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
            jac = numpy.zeros( (N, n_s+n_x, N, n_s+n_x) )
            for i in range(N-1):
                jac[i,n_s:,i,:n_s] = R_s[i,:,:]
                jac[i,n_s:,i,n_s:] = R_x[i,:,:]
                jac[i,n_s:,i+1,:n_s] = R_S[i,:,:]
                jac[i,n_s:,i+1,n_s:] = R_X[i,:,:]
                jac[i+1,:n_s,i,:n_s] = SS_s[i,:,:]
                jac[i+1,:n_s,i,n_s:] = SS_x[i,:,:]
                jac[i+1,:n_s,i+1,:n_s] = -numpy.eye(n_s)
                # jac[i,n_s:,i,:n_s] = R_s[i,:,:]
                # jac[i,n_s:,i,n_s:] = R_x[i,:,:]
                # jac[i+1,n_s:,i,:n_s] = R_S[i,:,:]
                # jac[i+1,n_s:,i,n_s:] = R_X[i,:,:]
                # jac[i,:n_s,i+1,:n_s] = SS_s[i,:,:]
                # jac[i,:n_s,i+1,n_s:] = SS_x[i,:,:]
                # jac[i+1,:n_s,i+1,:n_s] = -numpy.eye(n_s)
            jac[ 0,:n_s,0,:n_s] = - numpy.eye(n_s)
            jac[-1,n_s:,-1,n_s:] = - numpy.eye(n_x)
            jac[-1,n_s:,-2,n_s:] = + numpy.eye(n_x)
            nn = jac.shape[0]*jac.shape[1]
            res = res.ravel()
            jac = jac.reshape((nn,nn))


        if jactype == 'sparse':
            from scipy.sparse import csc_matrix, csr_matrix
            jac = csc_matrix(jac)
            # scipy bug ? I don't get the same with csr

        return [res,jac]


if __name__ == '__main__':

    # this example computes the response of the rbc economy to a series of expected productivity shocks.
    # investment is bounded by an exogenous value 0.2, so that investment is constrained in the first periods

    # TODO: propose a meaningful economic example

    from dolo import *

    from pylab import *

    model = yaml_import('../../examples/models/rbc_taxes.yaml')

    s = model.calibration['states']
    p = model.calibration['parameters']

    e = model.calibration['shocks']
    x = model.calibration['controls']
    
    f = model.functions['arbitrage']
    g = model.functions['transition']



    e_z = atleast_2d( linspace(0.0, 0.0, 10) ).T

    start_s = model.calibration['states'].copy()
    start_s[0] = 1.5

    import time

    #sol1 = deterministic_solve(model, shocks=e_z, T=50, use_pandas=True, ignore_constraints=True, start_s=start_s)

    #sol1 = deterministic_solve(model, T=50, use_pandas=True, ignore_constraints=True, start_s=start_s)

    from dolo.algos.steady_state import find_deterministic_equilibrium

    calib = find_deterministic_equilibrium(model)

    t2 = time.time()

    sol1 = deterministic_solve(model, start_states=start_s,  T=50, use_pandas=True, ignore_constraints=False, verbose=True)
    
    t3 = time.time()

    t1 = time.time()

    sol2 = deterministic_solve(model, start_states=start_s,  T=50, use_pandas=True, ignore_constraints=True, verbose=True)

    t2 = time.time()
    



    print("Elapsed : {}, {}".format(t2-t1, t3 - t2))

    
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

