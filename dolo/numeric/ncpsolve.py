from numpy import sqrt, finfo, inf

from numpy import isinf, newaxis, diag, zeros


from numpy.linalg import norm, solve

from numpy import float64

import warnings

from dolo.numeric.newton import serial_newton

def ncpsolve(f, a, b, x, tol=None, maxit=100, infos=False, verbose=False):

    def fcmp(z):

        [val, dval] = f(z)
        [val, dval] = serial_smooth(z, a, b, val, dval)

        return [val, dval]

    [sol, nit] = serial_newton(fcmp, x, tol=tol, maxit=maxit, verbose=verbose)

    return [sol, nit]


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