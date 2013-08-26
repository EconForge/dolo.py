import numpy
from numpy import linspace, zeros

def find_steady_state(model, e, force_values=None):
    '''
    Finds the steady state corresponding to exogenous shocks :math:`e`.

    :param model: an "fga" model.
    :param e: a vector with the value for the exogenous shocks.
    :param force_values: (optional) a vector where finite values override the equilibrium conditions. For instance a vector :math:`[0,nan,nan]` would impose that the first state must be equal to 0, while the two next ones, will be determined by the model equations. This is useful, when the deterministic model has a unit root.
    :return: a list containing a vector for the steady-states and the corresponding steady controls.
    '''

    s0 = model.calibration['states']
    x0 = model.calibration['controls']
    p = model.calibration['parameters']
    z = numpy.concatenate([s0, x0])

    e = numpy.atleast_2d(e.ravel()).T

    if force_values is not None:
        inds =  numpy.where( numpy.isfinite( force_values ) )[0]
        vals = force_values[inds]

    def fobj(z):
        s = numpy.atleast_2d( z[:len(s0)] ).T
        x = numpy.atleast_2d( z[len(s0):] ).T
        a = model.functions['auxiliary'](s,x,p)
        S = model.functions['transition'](s,x,a,e,p)
        if force_values is not None:
            S[inds,0] = vals
        r = model.functions['arbitrage'](s,x,a,s,x,a,p)
        return numpy.concatenate([S-s, r])

    from dolo.numeric.solver import solver
    steady_state = solver(fobj, z)

    return [steady_state[:len(s0)], steady_state[len(s0):]]

def deterministic_solve(model, shocks=None, T=100, use_pandas=True, ignore_constraints=False, start_s=None, verbose=False):
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

    # definitions
    n_s = len(model.calibration['states'])
    n_x = len(model.calibration['controls'])

    if shocks == None:
        shocks = numpy.zeros( (len(model.calibration['shocks']),1))

    # until last period, exogenous shock takes its last value
    epsilons = numpy.zeros( (shocks.shape[0], T))
    epsilons[:,:shocks.shape[1]] = shocks
    epsilons[:,shocks.shape[1:]] = shocks[:,-1:]

    # final initial and final steady-states consistent with exogenous shocks
    start = find_steady_state( model, numpy.atleast_2d(epsilons[:,0:1]), force_values=start_s)
    final = find_steady_state( model, numpy.atleast_2d(epsilons[:,-1:]))

    start_s = start[0]
    final_x = final[1]

    final = numpy.concatenate( final )
    start = numpy.concatenate( start )

    p = model.calibration['parameters']

    initial_guess = numpy.concatenate( [start*(1-l) + final*l for l in linspace(0.0,1.0,T+1)] )
    initial_guess = initial_guess.reshape( (-1, n_s + n_x)).T

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
    dfobj = lambda vec: det_residual(model, vec.reshape(sh), start_s, final_x, epsilons)[1].reshape((nn,nn))

    from dolo.numeric.solver import solver

    if not ignore_constraints:
        sol = solver(fobj, initial_guess, jac=dfobj, lb=lower_bound, ub=upper_bound, method='ncpsolve', serial_problem=False, verbose=verbose )
    else:
        sol = solver(fobj, initial_guess, jac=dfobj, method='fsolve', serial_problem=False, verbose=verbose )

    if use_pandas:
        import pandas
        colnames = model.symbols['states'] + model.symbols['controls'] + model.symbols['auxiliary']
        # compute auxiliaries
        y = model.functions['auxiliary'](sol[:n_s,:], sol[n_s:,:], p)
        sol = numpy.row_stack([sol,y])
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
    a = model.functions['auxiliary']

    vec = guess[:,:-1]
    vec_f = guess[:,1:]

    s = vec[:n_s,:]
    x = vec[n_s:,:]
    S = vec_f[:n_s,:]
    X = vec_f[n_s:,:]

    if diff:
        y, y_s, y_x = a(s,x,p, derivs=True)
        Y, Y_S, Y_X = a(S,X,p, derivs=True)
        SS, SS_s, SS_x, SS_y, SS_e = g(s,x,y,shocks,p, derivs=True)
        R, R_s, R_x, R_y, R_S, R_X, R_Y = f(s,x,y,S,X,Y,p,derivs=True)
    else:
        y = a(s,x,p)
        Y = a(S,X,p)
        SS = g(s,x,y,shocks,p)
        R = f(s,x,y,S,X,Y,p)

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

        # we compute the derivative matrix

        from dolo.numeric.serial_operations import serial_multiplication as smult
        res_s_s = SS_s + smult(SS_y, y_s)
        res_s_x = SS_x + smult(SS_y, y_x)

        F_s = R_s + smult(R_y, y_s)
        F_x = R_x + smult(R_y, y_x)
        F_S = R_S + smult(R_Y, Y_S)
        F_X = R_X + smult(R_Y, Y_X)

        # next block is probably very inefficient
        jac = numpy.zeros( (n_s+n_x, N, n_s+n_x, N) )
        for i in range(N-1):
            jac[n_s:,i,:n_s,i] = F_s[:,:,i]
            jac[n_s:,i,n_s:,i] = F_x[:,:,i]
            jac[n_s:,i,:n_s,i+1] = F_S[:,:,i]
            jac[n_s:,i,n_s:,i+1] = F_X[:,:,i]
            jac[:n_s,i+1,:n_s,i] = res_s_s[:,:,i]
            jac[:n_s,i+1,n_s:,i] = res_s_x[:,:,i]
            jac[:n_s,i+1,:n_s,i+1] = -numpy.eye(n_s)
        jac[:n_s,0,:n_s,0] = - numpy.eye(n_s)
        jac[n_s:,-1,n_s:,-1] = - numpy.eye(n_x)
        jac[n_s:,-1,n_s:,-2] = + numpy.eye(n_x)

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

    sol1 = deterministic_solve(model, shocks=e_z, T=50, use_pandas=True, ignore_constraints=True, start_s=start_s)

    sol1 = deterministic_solve(model, T=50, use_pandas=True, ignore_constraints=True, start_s=start_s)

    sol2 = deterministic_solve(model, shocks=e_z, T=50, use_pandas=True, ignore_constraints=False)
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

