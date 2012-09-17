from numpy import sqrt, finfo, inf

from numpy import isinf, newaxis, diag


from numpy.linalg import norm, solve

from numpy import float64

import warnings

def ncpsolve(f, a, b, x, tol=None, verbose=True):
    '''
    don't ask what ncpsolve can do for you...
    :param f:
    :param a:
    :param b:
    :param x:
    :return:
    '''

    maxit = 100

    if tol is None:
        tol = sqrt( finfo( float64 ).eps )

    maxsteps = 10
    showiters = True


    it = 0
    if verbose:
        headline = '|{0:^5} | {1:^12} | {2:^12} |'.format( 'k',' backsteps', '||f(x)||' )
        stars = '-'*len(headline)
        print stars
        print(headline)
        print stars

    while it < maxit:

        it += 1

        [fval, fjac] = f(x)
        [ftmp, fjac] = smooth(x, a, b, fval, fjac)
        fnorm = norm( ftmp, ord=inf)

        if fnorm < tol:
            if verbose:
                print(stars)
            return [x,fval]

        dx = - solve( fjac, ftmp)

        fnormold = inf

        for backsteps in range(maxsteps):

            xnew = x + dx
            fnew = f(xnew)[0] # TODO: don't ask for derivatives
            fnew = smooth( xnew, a, b, fnew )
            fnormnew = norm(fnew, ord=inf)

            if fnormnew < fnorm:
                break
            if fnormold < fnormnew:
                dx = 2*dx
                break

            fnormold = fnormnew
            dx = dx/2

        x = x + dx

        if verbose:
            print('|{0:5} | {2:12.3e} | {2:12.3e} |'.format( it, backsteps, fnormnew) )


    if verbose:
        print(stars)

    warnings.Warning('Failure to converge in ncpsolve')

    fval = f(x)

    return [x,fval]



def smooth(x, a, b, fx, J=None):

    '''
    smoooth
    :param x: vector of evaluation points
    :param a: lower bounds
    :param b: upper bounds
    :param fx: function values at x
    :param J: jacobian of f at x
    :return:
    '''



    dainf = isinf(a)
    dbinf = isinf(b)

    n = len(x)

    da = a - x
    db = b - x

    # TODO: ignore warnings when there are infinite values.
    sq1 = sqrt( fx**2 + da**2)
    pval = fx + sq1 + da
    pval[dainf] = fx[dainf]

    sq2 = sqrt(pval**2 + db**2)

    fxnew = pval - sq2 + db

    fxnew[dbinf] = pval[dbinf]

    if J is None:
        return fxnew

    # let compute the jacobian

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

    Jnew = ff[:,newaxis] * J - diag(xx)

    return [fxnew, Jnew]




