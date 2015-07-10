from numpy import sqrt, finfo, inf
from numpy import isinf, newaxis, diag, zeros
from numpy.linalg import norm, solve
from numpy import float64
import warnings

from dolo.numeric.optimize.newton import newton

def ncpsolve(f, a, b, x, tol=1e-8, maxit=100, infos=False, verbose=False, jactype='serial'):

    def fcmp(z):

        [val, dval] = f(z)

        # revert convention
        val = -val
        dval = -dval

        [val, dval] = smooth(z, a, b, val, dval, jactype=jactype)
        
        return [val, dval]

    [sol, nit] = newton(fcmp, x, tol=tol, maxit=maxit, verbose=verbose, jactype=jactype)

    return [sol, nit]


def smooth(x, a, b, fx, J, jactype='serial'):

    BIG = 1e20

    dainf = a<=-BIG   #  isinf(a) |
    dbinf = b>=BIG   #  isinf(b)

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

#    jac_is_sparse = scipy.sparse.issparse(J)

    if jactype == 'sparse':
        import scipy.sparse
        from scipy.sparse import csc_matrix, csr_matrix
        from scipy.sparse import diags
        fff = diags([ff], [0])
        xxx = diags([xx], [0])
        # TODO: preserve csc or csr format
        Jnew = fff*J - xxx

        Jnew = csc_matrix(Jnew)

        return [fxnew, Jnew]

        # TODf: preserve csc or csr format

    elif jactype == 'full':
        from numpy import diag
#        fff = diag(ff)
        fff = ff[:,None]
        xxx = diag( xx )
#        xxx = diag(xx)
        Jnew = fff*J - xxx
        return fx, J
        return [fxnew, Jnew]


    else:

        fff = ff[:,:,newaxis]

        xxx = zeros(J.shape)
        for i in range(xx.shape[1]):
            xxx[:,i,i] = xx[:,i]
        Jnew = fff*J - xxx
        return [fxnew, Jnew]
