from __future__ import print_function

old_print = print

from numpy import zeros_like, zeros
from numpy.linalg import solve

import numpy

def serial_solve(M, res):

    sol = zeros_like(res)

    for i in range(sol.shape[0]):
        try:
            sol[i,:] = solve( M[i,:,:], res[i,:])
        except:
            # Should be a special type of exception
            a = Exception("Error solving point {}".format(i))
            a.x = res[i,:]
            a.J = M[i,:,:]
            a.i = i
            raise a
    
    return sol

import time

def newton(f, x, verbose=False, tol=1e-6, maxit=5, jactype='serial'):

    if verbose:
        print = lambda txt: old_print(txt)
    else:
        print = lambda txt: None

    it = 0
    error = 10
    converged = False
    maxbacksteps = 30

    x0 = x

    if jactype == 'sparse':
        from scipy.sparse.linalg import spsolve as solve
    elif jactype == 'full':
        from numpy.linalg import solve
    else:
        solve = serial_solve

    while it<maxit and not converged:

        it += 1

        [v,dv] = f(x)

        # TODO: rewrite starting here

#        print("Time to evaluate {}".format(ss-tt)0)

        error_0 = abs(v).max()

        if error_0 < tol:
            
            converged = True

        else:

            dx = solve(dv, v)

            norm_dx = abs(dx).max()

            for bck in range(maxbacksteps):
                xx = x - dx*(2**(-bck))
                vm = f(xx)[0]
                err = abs(vm).max()
                if err < error_0:
                    break

            x = xx

            if verbose:
                print("\t> {} | {} | {}".format(it, err, bck))

    if converged:
        return [x, it]
    else:
        raise Exception("Did not converge")

serial_newton = newton

from numpy import sqrt, finfo, inf

from numpy import isinf, newaxis, diag, zeros

def SerialDifferentiableFunction(f, epsilon=1e-8):

    def df(x):

        v0 = f(x)

        N = v0.shape[0]
        n_v = v0.shape[1]
        assert(x.shape[0] == N)
        n_x = x.shape[1]

        dv = zeros( (N, n_v, n_x) )

        for i in range(n_x):

            xi = x.copy()
            xi[:,i] += epsilon

            vi = f(xi)

            dv[:,:,i] = (vi - v0)/epsilon

        return [v0, dv]

    return df         
