import numpy
import scipy.sparse
import time
from numba import jit
from dolo.numeric.serial_operations import serial_multiplication as smult
from dolo.algos.dtcscc.perturbations import approximate_controls
from dolo.algos.dtcscc.time_iteration import create_interpolator


def nonlinear_system(model, dr0=None, maxit=10, tol=1e-8,  grid={}, distribution={}, verbose=True):

    '''
    Finds a global solution for ``model`` by solving one large system of equations
    using a simple newton algorithm.

    Parameters
    ----------
    model: NumericModel
        "dtcscc" model to be solved
    verbose: boolean
        if True, display iterations
    dr0: decision rule
        initial guess for the decision rule
    maxit:  int
        maximum number of iterationsd
    tol: tolerance criterium for successive approximations
    grid: grid options
    distribution: distribution options

    Returns
    -------
    decision rule :
        approximated solution
    '''


    if verbose:
        headline = '|{0:^4} | {1:10} | {2:8} |'
        headline = headline.format('N', ' Error', 'Time')
        stars = '-'*len(headline)
        print(stars)
        print(headline)
        print(stars)
        # format string for within loop
        fmt_str = '|{0:4} | {1:10.3e} | {2:8.3f} |'

    f = model.functions['arbitrage']
    g = model.functions['transition']
    p = model.calibration['parameters']

    distrib = model.get_distribution(**distribution)
    nodes, weights = distrib.discretize()

    approx = model.get_endo_grid(**grid)
    ms = create_interpolator(approx, approx.interpolation)

    grid = ms.grid

    if dr0 is None:
        dr = approximate_controls(model)
    else:
        dr = dr0

    ms.set_values(dr(grid))

    x = dr(grid)
    x0 = x.copy()

    it = 0
    err = 10


    a0 = x0.copy().reshape((x0.shape[0]*x0.shape[1],))
    a = a0.copy()

    while err > tol and it < maxit:

        it += 1
        t1 = time.time()

        r, da = residuals(f, g, grid, a.reshape(x0.shape), ms, nodes, weights, p, diff=True)[:2]
        r = r.flatten()

        err = abs(r).max()

        t2 = time.time()

        if verbose:
            print(fmt_str.format(it, err, t2-t1))

        if err > tol:
            a -= scipy.sparse.linalg.spsolve(da, r)

    if verbose:
        print(stars)

    return ms


@jit
def serial_to_full(m):
    # m is a N * n_x * n_x array
    # it is converted into a (N*n_x)*(N*n_x) sparse array M
    # such that nonzero elements are M[n,i,n,j] = m[n,i,j]

    N, n_x, n_xx = m.shape
    assert(n_x == n_xx)
    val = numpy.zeros((N*n_x*N))
    ind_i = numpy.zeros((N*n_x*N))
    ind_j = numpy.zeros((N*n_x*N))
    t = 0
    for n in range(N):
        for i in range(n_x):
            for j in range(n_x):
                val[t] = m[n, i, j]
                ind_i[t] = n_x*n + i
                ind_j[t] = n_x*n + j
                t += 1

    mat = scipy.sparse.coo_matrix((val, (ind_i, ind_j)), shape=(N*n_x, N*n_x))
    mmat = mat.tocsr()
    mmat.eliminate_zeros()
    return mmat


@jit
def diag_to_full(m):
    # m is a N * n_x * N array
    # it is converted into a (N*n_x)*(N*n_x) sparse array M
    # such that nonzero elements are M[p,i,q,i] = m[p,i,q]
    N, n_x, NN = m.shape
    assert(N == NN)
    val = numpy.zeros((N*n_x*N))
    ind_i = numpy.zeros((N*n_x*N))
    ind_j = numpy.zeros((N*n_x*N))

    t = 0
    for n in range(N):
        for i in range(n_x):
            for nn in range(N):
                val[t] = m[n, i, nn]
                ind_i[t] = n_x*n + i
                ind_j[t] = n_x*nn + i
                t += 1

    mat = scipy.sparse.coo_matrix((val, (ind_i, ind_j)), shape=(N*n_x, N*n_x))
    mmat = mat.tocsr()
    mmat.eliminate_zeros()
    return mmat



def residuals(f, g, s, x, dr, nodes, weights, p, diff=True):

    N, n_x = x.shape

    dr.set_values(x)

    output = numpy.zeros((N, n_x))

    if not diff:
        for i in range(nodes.shape[0]):
            E = nodes[i, :][None, :].repeat(N, axis=0)
            S = g(s, x, E, p)
            X = dr.interpolate(S)
            t = f(s, x, E, S, X, p)
            output += weights[i]*t
        return output

    if diff:

        output_x = scipy.sparse.csr_matrix((N*n_x, N*n_x))

        for i in range(nodes.shape[0]):
            E = nodes[i, :][None, :].repeat(N, axis=0)
            S, S_s, S_x, S_E = g(s, x, E, p, diff=True)
            X, X_S, X_x = dr.interpolate(S, deriv=True, deriv_X=True)
            R, R_s, R_x, R_E, R_S, R_X = f(s, x, E, S, X, p, diff=True)

            output += weights[i]*R

            t1 = weights[i]*(R_x + smult(R_S, S_x) + smult(R_X, smult(X_S, S_x)))  # N.n_x.n_x
            t1 = serial_to_full(t1)
            t2 = weights[i] * serial_to_full(R_X) @ diag_to_full(X_x)
            output_x += t1 + t2

        return output, output_x
