from numba import jit
from numpy.linalg import solve
from numpy import zeros_like
from numba import guvectorize, float64, void, generated_jit
import numpy
from numba import generated_jit


def swaplines_tensor(i, j, M):
    n0, n1, n2 = M.shape
    for k in range(n1):
        for l in range(n2):
            t = M[i, k, l]
            M[i, k, l] = M[j, k, l]
            M[j, k, l] = t


def swaplines_matrix(i, j, M):
    n = M.shape[1]
    for k in range(n):
        t = M[i, k]
        M[i, k] = M[j, k]
        M[j, k] = t


def swaplines_vector(i, j, M):
    n = M.shape[0]
    t = M[i]
    M[i] = M[j]
    M[j] = t


@generated_jit(cache=True)
def swaplines(i, j, M):
    if M.ndim == 1:
        return swaplines_vector
    elif M.ndim == 2:
        return swaplines_matrix
    elif M.ndim == 3:
        return swaplines_tensor


def substract_tensor(i, j, c, M):
    # Li <- Li - c*Lj
    n0, n1, n2 = M.shape
    for k in range(n1):
        for l in range(n2):
            M[i, k, l] = M[i, k, l] - c * M[j, k, l]


def substract_matrix(i, j, c, M):
    # Li <- Li - c*Lj
    n = M.shape[0]
    for k in range(n):
        M[i, k] = M[i, k] - c * M[j, k]


def substract_vector(i, j, c, M):
    # Li <- Li - c*Lj
    # n = M.shape[0]
    M[i] = M[i] - c * M[j]


@generated_jit(cache=True)
def substract(i, j, c, M):
    if M.ndim == 1:
        return substract_vector
    elif M.ndim == 2:
        return substract_matrix
    elif M.ndim == 3:
        return substract_tensor


def divide_tensor(i, c, M):
    # Li <- Li - c*Lj
    n0, n1, n2 = M.shape
    for k in range(n1):
        for l in range(n2):
            M[i, k, l] /= c


def divide_matrix(i, c, M):
    # Li <- Li - c*Lj
    n = M.shape[0]
    for k in range(n):
        M[i, k] /= c


def divide_vector(i, c, M):
    # Li <- Li - c*Lj
    M[i] /= c


@generated_jit(cache=True)
def divide(i, c, M):
    if M.ndim == 1:
        return divide_vector
    elif M.ndim == 2:
        return divide_matrix
    elif M.ndim == 3:
        return divide_tensor


@jit(nopython=True, cache=True)
def invert(A, B):

    # inverts A and puts result in B (modifies inputs)
    n = A.shape[0]
    #
    for i in range(n):
        # print(A,B)
        # find pivot
        max_err = -1.0
        max_i = 0
        for i0 in range(i, n):
            err = abs(A[i0, i])
            if err >= max_err:
                max_err = err
                max_i = i0

        swaplines(i, max_i, A)
        swaplines(i, max_i, B)

        c = A[i, i]
        divide(i, c, A)
        divide(i, c, B)

        for i0 in range(i + 1, n):
            f = A[i0, i]
            substract(i0, i, f, A)
            substract(i0, i, f, B)

    for i in range(n - 1, -1, -1):
        for i0 in range(i):
            f = A[i0, i]
            substract(i0, i, f, A)
            substract(i0, i, f, B)


target = "parallel"


@guvectorize(
    [(float64[:, :], float64[:, :])],
    "(n,n)->(n,n)",
    nopython=True,
    target=target,
    cache=True,
)
def invert_gu(A, Ainv):
    Ainv[:, :] = 0
    n = A.shape[0]
    for i in range(n):
        Ainv[i, i] = 1.0
    invert(A, Ainv)


@guvectorize(
    [(float64[:, :], float64[:])],
    "(n,n)->(n)",
    nopython=True,
    target=target,
    cache=True,
)
def solve_gu(A, V):
    n = A.shape[0]
    invert(A, V)


@guvectorize(
    [(float64[:, :], float64[:, :], float64[:])],
    "(n,n),(n,p)->()",
    nopython=True,
    target=target,
    cache=True,
)
def solve_tensor(A, V, dum):
    n = A.shape[0]
    invert(A, V)


@guvectorize(
    [(float64[:, :], float64[:, :, :], float64[:])],
    "(n,n),(n,p,q)->()",
    nopython=True,
    target=target,
    cache=True,
)
def solve_tensor_old(A, V, dum):
    n = A.shape[0]
    invert(A, V)


def test_list_of_matrices():
    import numpy

    N = 10
    m = 4
    A0 = numpy.random.random((N, m, m))
    A = A0.copy()
    B = invert_gu(A)
    err_max = 0.0
    for i in range(N):
        mat = A0[i, :, :]
        matinv = B[i, :, :]
        err = abs(mat @ matinv - numpy.eye(m)).max()
        err_2 = abs(A - numpy.eye(m)).max()
        assert err < 1e-11
        assert err_2 < 1e-11


@jit
def serial_solve(A, B, diagnose=True):
    sol = zeros_like(B)
    for i in range(sol.shape[0]):
        sol[i, :] = solve(A[i, :, :], B[i, :])
    return sol


@jit(nopython=True)
def mult_AB(dres, jres):
    n_m, N, n_x, n_xx = dres.shape
    assert jres.shape == (n_m, N, n_x, n_m, n_x)
    out = jres.copy()
    for i in range(n_m):
        for j in range(n_m):
            for v in range(n_x):
                out[i, :, :, j, v] = serial_solve(dres[i, :, :, :], jres[i, :, :, j, v])
    return out


def test_list_of_arrays():
    import numpy

    m = 4
    N = 10
    x = 2
    A0 = numpy.random.random((m, N, x, x))
    B0 = numpy.random.random((m, N, x, m, x))
    A = A0.copy()
    B = B0.copy()
    C = solve_tensor(A, B)
    C2 = mult_AB(A0, B0)

    print(abs(C2 - B).max())


#
# test_list_of_arrays()
#
# test_list_of_matrices()


# if False:
#
#
#     A = numpy.random.random((3,3))
#     P = A.copy()
#     Q = numpy.eye(A.shape[0])
#     invert(P,Q)
#
#     A = numpy.random.random((3,3))
#     V = numpy.random.random((3))
#     P = A.copy()
#     W = V.copy()
#     invert(P,W)
#
#     N = 10000
#     m = 9
#     x = 8
#     A0 = numpy.random.random((N,m,8,8))
#     V0 = numpy.random.random((N,m,8))
#     A = A0.copy()
#     V = V0.copy()
#
#
#     from serial_operations import serial_solve
#     A = A0.copy()
#     V = V0.copy()
#
#
#     N = 10000
#     m = 27
#     x = 8
#     A0 = numpy.random.random((N,m,x,x))
#     V0 = numpy.random.random((N,m,x,m,x))
#     A = A0.copy()
#     V = V0.copy()
#
#     i = 5
#     A0[i,:,:]@V[i,:,0,2]
#
#     V0[i,:,0,2]
