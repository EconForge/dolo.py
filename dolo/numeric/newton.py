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
            # Should be a special type of excaption
            a = Exception("Error solving point {}".format(i))
            a.x = res[i,:]
            a.J = M[i,:,:]
            a.i = i
            raise a
    
    return sol

import time
def serial_newton(f, x, verbose=False, tol=1e-6, maxit=5):

    if verbose:
        print = lambda txt: old_print(txt)
    else:
        print = lambda txt: None

    it = 0
    error = 10
    converged = False
    maxbacksteps = 30

    x0 = x

    while it<maxit and not converged:

        it += 1

        tt = time.time()

        [v,dv] = f(x)
        ss = time.time()
#        print("Time to evaluate {}".format(ss-tt)0)

        error_0 = abs(v).max()

        if error_0 < tol:
            
            converged = True

        else:

            
            print("Correction size :")
            print(abs(dv).max(axis=(0,)))


            t1 = time.time()
            dx = serial_solve(dv, v)
            t2 = time.time()
            print("Time to invert {}".format(t2-t1))
            print(abs(dx).max(axis=(0,)))
            norm_dx = abs(dx).max()
#            if norm_dx > 1.0:
#                dx /= norm_dx
            print("   >> {}".format(error_0))
            for bck in range(maxbacksteps):
                xx = x - dx*(2**(-bck))
                vm = f(xx)[0]
                err = abs(vm).max()
                print( "   >>> {}".format(err))
                if err < error_0:
                    break
                

            x = xx

            if verbose:
                print("\t> {} | {} | {}".format(it, err, bck))

#    if converged:
    return [x, it]
#    else:
#        raise Exception("Did not converge")

from numpy import sqrt, finfo, inf

from numpy import isinf, newaxis, diag, zeros

def serial_smooth(x, a, b, fx, J):


    dainf = isinf(a)
    dbinf = isinf(b)

    da = a - x
    db = b - x

    # TODO: ignore warnings when there are infinite values.
    sq1 = sqrt( fx**2 + da**2)
    pval = fx + sq1 + da
    pval[dainf] = fx[dainf]

    sq2 = sqrt(pval**2 + db**2)

    fxnew = pval - sq2 + db

    fxnew[dbinf] = pval[dbinf]

    dpdy = 1 + fx/sq1
    dpdy[dainf] = 1

    dpdz = 1 + da/sq1
    dpdz[dainf] = 0

    dmdy = 1 - pval/sq2
    dmdy[dbinf] = 1

    dmdz = 1 - db/sq2
    dmdz[dbinf] = 0

    ff = dmdy*dpdy
    xx = dmdy*dpdz + dmdz

    # TODO: rewrite starting here

    fff = ff[:,:,newaxis]

    xxx = zeros(J.shape)
    for i in range(xx.shape[1]):
        xxx[:,i,i] = xx[:,i]

    Jnew = fff*J - xxx

    return [fxnew, Jnew]

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
