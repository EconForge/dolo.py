# warning : this code is EXPERIMENTAL
# don't use it in serious projects unless you know what you are doing


from __future__ import division

from dolo.misc.yamlfile import yaml_import
from dolo.numeric.decision_rules_states import CDR

import numpy
numpy.set_printoptions(precision=3, suppress=3)

import numpy as np


def solve_model_around_risky_ss(model, verbose=False, return_dr=True, initial_sol=None):

    #model = yaml_import(filename)

    if initial_sol == None:
        if 'model_type' in model and model['model_type'] == 'portfolios':
            print('This is a portfolio model ! Converting to deterministic one.')
            from portfolio_perturbation import portfolios_to_deterministic
            model = portfolios_to_deterministic(model,['x_1','x_2'])
            model.check()
        from dolo.numeric.perturbations_to_states import approximate_controls
        perturb_sol = approximate_controls(model, order = 1, return_dr=False, substitute_auxiliary=True)
        [X_bar,X_s] =  perturb_sol
    else:
        perturb_sol = initial_sol
        [X_bar,X_s] =  perturb_sol


    # reduce X_s to the real controls  (remove auxiliary variables

    X_s = X_s[:len(model['variables_groups']['controls']),:]
    X_bar = X_bar[:len(model['variables_groups']['controls'])]
    X_bar = numpy.array(X_bar)

    if  abs(X_s.imag).max() < 1e-10:
        X_s = X_s.real
    else:
        raise( Exception('Complex decision rule') )

    print('Perturbation solution found')

    #print model.parameters
    #exit()

    X_bar_1 = X_bar
    X_s_1 = X_s

    #model = yaml_import(filename)

    #from dolo.symbolic.symbolic import Parameter

    [S_bar, X_bar, X_s, P] = solve_risky_ss(model, X_bar, X_s, verbose=verbose)

    if return_dr:
        cdr = CDR([S_bar, X_bar, X_s])
#        cdr.P = P
        return cdr
    return [S_bar, X_bar, X_s, P]



def solve_risky_ss(model, X_bar, X_s, verbose=False):

    import numpy
    from dolo.compiler.compiling import compile_function
    import time
    from dolo.compiler.compiler_global import simple_global_representation


    [y,x,parms] = model.read_calibration()
    sigma = model.read_covariances()

    sgm = simple_global_representation(model, substitute_auxiliary=True)

    states = sgm['states']
    controls = sgm['controls']
    shocks = sgm['shocks']
    parameters = sgm['parameters']
    f_eqs = sgm['f_eqs']
    g_eqs = sgm['g_eqs']


    g_args = [s(-1) for s in states] + [c(-1) for c in controls] + shocks
    f_args = states + controls + [v(1) for v in states] + [v(1) for v in controls]
    p_args = parameters



    g_fun = compile_function(g_eqs, g_args, p_args, 2, return_function=True)
    f_fun = compile_function(f_eqs, f_args, p_args, 3, return_function=True)


    epsilons_0 = np.zeros((sigma.shape[0]))

    from numpy import dot
    from dolo.numeric.tensor import sdot,mdot

    def residuals(X, sigma, parms, g_fun, f_fun):

        import numpy

        dummy_x = X[0:1,0]
        X_bar = X[1:,0]
        S_bar = X[0,1:]
        X_s = X[1:,1:]

        [n_x,n_s] = X_s.shape

        n_e = sigma.shape[0]

        xx = np.concatenate([S_bar, X_bar, epsilons_0])

        [g_0, g_1, g_2] = g_fun(xx, parms)
        [f_0,f_1,f_2,f_3] = f_fun( np.concatenate([S_bar, X_bar, S_bar, X_bar]), parms)

        res_g = g_0 - S_bar

        # g is a first order function
        g_s = g_1[:,:n_s]
        g_x = g_1[:,n_s:n_s+n_x]
        g_e = g_1[:,n_s+n_x:]
        g_se = g_2[:,:n_s,n_s+n_x:]
        g_xe = g_2[:, n_s:n_s+n_x, n_s+n_x:]


        # S(s,e) = g(s,x,e)
        S_s = g_s + dot(g_x, X_s)
        S_e = g_e
        S_se = g_se + mdot(g_xe,[X_s, numpy.eye(n_e)])


        # V(s,e) = [ g(s,x,e) ; x( g(s,x,e) ) ]
        V_s = np.row_stack([
            S_s,
            dot( X_s, S_s )
        ])    # ***

        V_e = np.row_stack([
            S_e,
            dot( X_s, S_e )
        ])

        V_se = np.row_stack([
            S_se,
            dot( X_s, S_se )
        ])

        # v(s) = [s, x(s)]
        v_s = np.row_stack([
            numpy.eye(n_s),
            X_s
        ])


        # W(s,e) = [xx(s,e); yy(s,e)]
        W_s = np.row_stack([
            v_s,
            V_s
        ])

        #return

        nn = n_s + n_x
        f_v = f_1[:,:nn]
        f_V = f_1[:,nn:]
        f_1V = f_2[:,:,nn:]
        f_VV = f_2[:,nn:,nn:]
        f_1VV = f_3[:,:,nn:,nn:]


        #        E = lambda v: np.tensordot(v, sigma,  axes=((2,3),(0,1)) ) # expectation operator

        F = f_0 + 0.5*np.tensordot(  mdot(f_VV,[V_e,V_e]), sigma, axes=((1,2),(0,1))  )

        F_s = sdot(f_1, W_s)
        f_see = mdot(f_1VV, [W_s, V_e, V_e]) + 2*mdot(f_VV, [V_se, V_e])
        F_s += 0.5 * np.tensordot(f_see, sigma,  axes=((2,3),(0,1)) ) # second order correction

        resp = np.row_stack([
            np.concatenate([dummy_x,res_g]),
            np.column_stack([F,F_s])
        ])

        return resp

    #    S_bar = s_fun_init( numpy.atleast_2d(X_bar).T ,parms).flatten()
    #    S_bar = S_bar.flatten()
    S_bar = [ y[i] for i,v in enumerate(model.variables) if v in model['variables_groups']['states'] ]
    S_bar = np.array(S_bar)

    X0 = np.row_stack([
        np.concatenate([np.zeros(1),S_bar]),
        np.column_stack([X_bar,X_s])
    ])

    from dolo.numeric.solver import solver


    fobj = lambda X: residuals(X,  sigma, parms, g_fun, f_fun)

    if verbose:
        val = fobj(X0)
        print('val')
        print(val)

    #    exit()

    t = time.time()

    sol = solver(fobj,X0, method='lmmcp', verbose=verbose, options={'preprocess':False, 'eps1':1e-15, 'eps2': 1e-15})

    if verbose:
        print('initial guess')
        print(X0)
        print('solution')
        print sol
        print('initial residuals')
        print(fobj(X0))
        print('residuals')
        print fobj(sol)
        s = time.time()


    if verbose:
        print('Elapsed : {0}'.format(s-t))
        #sol = solver(fobj,X0, method='fsolve', verbose=True, options={'preprocessor':False})

    norm = lambda x: numpy.linalg.norm(x,numpy.inf)
    if verbose:
        print( "Initial error: {0}".format( norm( fobj(X0)) ) )
        print( "Final error: {0}".format( norm( fobj(sol) ) ) )

        print("Solution")
        print(sol)

    X_bar = sol[1:,0]
    S_bar = sol[0,1:]
    X_s = sol[1:,1:]

    # compute transitions
    n_s = len(states)
    n_x = len(controls)
    [g, dg, junk] = g_fun( np.concatenate( [S_bar, X_bar, epsilons_0] ), parms)
    g_s = dg[:,:n_s]
    g_x = dg[:,n_s:n_s+n_x]

    P = g_s + dot(g_x, X_s)


    if verbose:
        eigenvalues = numpy.linalg.eigvals(P)
        print eigenvalues
        eigenvalues = [abs(e) for e in eigenvalues]
        eigenvalues.sort()
        print(eigenvalues)

    return [S_bar, X_bar, X_s, P]


#
#
#
#fname =  'capital'
#
#filename = 'models/{0}.yaml'.format(fname)
#
#[S_bar, X_bar, X_s, P] = solve_model_around_risky_ss(filename)
if __name__ == '__main__':
    #fname = '/home/pablo/Documents/Research/Thesis/chapter_4/code/models/open_economy.yaml'
    fname = '/home/pablo/Documents/Research/Thesis/chapter_4/code/models/open_economy_with_pf.yaml'
    sol = solve_model_around_risky_ss(fname, verbose=True)