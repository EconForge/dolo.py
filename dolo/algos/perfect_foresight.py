from numpy import column_stack
import numpy
from numpy import linalg, linspace, zeros

def find_steady_state(model, e):
    s0 = model.calibration['states']
    x0 = model.calibration['controls']
    p = model.calibration['parameters']
    z = numpy.concatenate([s0, x0])
    def fobj(z):
        s = numpy.atleast_2d( z[:len(s0)] ).T
        x = numpy.atleast_2d( z[len(s0):] ).T

        a = model.functions['auxiliary'](s,x,p)
        S = model.functions['transition'](s,x,a,e,p)
        r = model.functions['arbitrage'](s,x,a,s,x,a,p)
        return numpy.concatenate([S-s, r])
    from dolo.numeric.solver import solver
    steady_state = solver(fobj, z)
    return [steady_state[:len(s0)], steady_state[len(s0):]]


def deterministic_solve(model, shocks, start_s=None, initial_dr=None, T=500, use_pandas=False):

    # definitions
    n_s = len(model.calibration['states'])
    n_x = len(model.calibration['controls'])


    # until last period, exogenous shock takes its last value
    epsilons = numpy.zeros( (shocks.shape[0], T))
    epsilons[:,:shocks.shape[1]] = shocks
    epsilons[:,shocks.shape[1:]] = shocks[:,-1:]

    # final initial and final steady-states consistent with exogenous shocks
    start = find_steady_state( model, numpy.atleast_2d(epsilons[:,0:1]))
    final = find_steady_state( model, numpy.atleast_2d(epsilons[:,-1:]))

    start_s = start[0]
    final_x = final[1]

    final = numpy.concatenate( final )
    start = numpy.concatenate( start )

    p = model.calibration['parameters']


    initial_guess = numpy.concatenate( [start*(1-l) + final*l for l in linspace(0.0,1.0,T+1)] )
    initial_guess = initial_guess.reshape( (-1, n_s + n_x)).T

    sh = initial_guess.shape

    initial_states = initial_guess[:n_s,:]
    [lb, ub] = [ u( initial_states, p ) for u in model.x_bounds]

    lower_bound = initial_guess*0 - numpy.inf
    lower_bound[n_s:,:] = lb

    upper_bound = initial_guess*0 + numpy.inf
    upper_bound[n_s:,:] = ub

    nn = sh[0]*sh[1]

    fobj  = lambda vec: det_residual(model, vec.reshape(sh), start_s, final_x, epsilons)[0].ravel()
    dfobj = lambda vec: det_residual(model, vec.reshape(sh), start_s, final_x, epsilons)[1].reshape((nn,nn))

    import time
    t = time.time()

    from dolo.numeric.solver import solver

    sol = solver(fobj, initial_guess, jac=dfobj, lb=lower_bound, ub=upper_bound, method='ncpsolve', serial_problem=False )
    # sol = solver(fobj, initial_guess, lb=lower_bound, ub=upper_bound, method='ncpsolve', serial_problem=False, verbose=True )
    # sol = solver(fobj, initial_guess, serial_problem=False, verbose=True )


    s = time.time()
    print('Elapsed : {}'.format(s-t))

    if use_pandas:
        import pandas
        colnames = model.symbols['states'] + model.symbols['controls']
        ts = pandas.DataFrame(sol.T, columns=colnames)
        return ts
    return sol


def det_residual(model, guess, start, final, shocks):

    n_s = len( model.symbols['states'] )
    n_x = len( model.symbols['controls'] )

    n_e = len( model.symbols['shocks'] )
    N = guess.shape[1]

    ee = zeros((n_e,1))

    p = model.calibration['parameters']

    f = model.functions['arbitrage']
    g = model.functions['transition']
    a = model.functions['auxiliary']

    # guess = numpy.concatenate( [start, guess, final] )
    # guess = guess.reshape( (-1, n_s+n_x) ).T

    vec = guess[:,:-1]
    vec_f = guess[:,1:]

    s = vec[:n_s,:]
    x = vec[n_s:,:]
    S = vec_f[:n_s,:]
    X = vec_f[n_s:,:]

    y, y_s, y_x = a(s,x,p, derivs=True)
    Y, Y_S, Y_X = a(S,X,p, derivs=True)

    SS, SS_s, SS_x, SS_y, SS_e = g(s,x,y,shocks,p, derivs=True)
    R, R_s, R_x, R_y, R_S, R_X, R_Y = f(s,x,y,S,X,Y,p,derivs=True)

    res_s = SS - S
    res_x = R


    res = numpy.zeros( (n_s+n_x, N) )

    res[:n_s,1:] = res_s
    res[n_s:,:-1] = res_x

    res[:n_s,0] = - (guess[:n_s,0] - start)
    res[n_s:,-1] = - (guess[n_s:,-1] - final)


    from dolo.numeric.serial_operations import serial_multiplication as smult

    from numpy import concatenate

    res_s_s = SS_s + smult(SS_y, y_s)
    res_s_x = SS_x + smult(SS_y, y_x)
    # res_s_S = numpy.zeros( (n_s,n_s,N-1) ) - numpy.eye(n_s)[:,:,None]
    # res_s_X = numpy.zeros( (n_s,n_x,N-1) )

    F_s = R_s + smult(R_y, y_s)
    F_x = R_x + smult(R_y, y_x)
    F_S = R_S + smult(R_Y, Y_S)
    F_X = R_X + smult(R_Y, Y_X)

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

    return [res,jac]


if __name__ == '__main__':

    from dolo import *

    from pylab import *

    model = yaml_import('examples/global_models/rbc_pf.yaml')

    e_z = atleast_2d( linspace(0.1, 0.0, 10) )

    sol = deterministic_solve(model, shocks=e_z, T=100, use_pandas=True)

    from pylab import *

    plot(sol['k'], label='k')
    plot(sol['z'], label='z')
    legend()
    show()

