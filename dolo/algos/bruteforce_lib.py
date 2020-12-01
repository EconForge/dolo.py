from .invert import *

from operator import mul
from functools import reduce
from dolo.numeric.optimize.newton import SerialDifferentiableFunction


def prod(l):
    return reduce(mul, l)


from math import sqrt
from numba import jit
import time
from numpy import array, zeros
from numpy.linalg import inv, solve
from numpy.linalg import inv
from numpy import abs
from numba import jit

import time


def B_prod(bases):

    # Compute B for a product of bases

    import scipy
    import scipy.sparse

    n = [b.B.shape[0] - 2 for b in bases]
    dims = array(n) + 2

    if len(dims) == 1:
        B = bases[0].B

    elif len(dims) == 2:

        B_x = bases[0].B
        B_y = bases[1].B
        B_prod_x = numpy.zeros(tuple(dims) * 2)  # filters last dim
        B_prod_y = numpy.zeros(tuple(dims) * 2)  # filters first dim

        sh = B_prod_x.shape

        for i in range(B_prod_x.shape[1]):
            # for j in B_prod_x.shape[3]:
            B_prod_x[:, i, :, i] = B_x

        for i in range(B_prod_y.shape[0]):
            # for j in B_prod_x.shape[3]
            B_prod_y[i, :, i, :] = B_y

        dim = numpy.prod(dims)

        B_p_x = scipy.sparse.csc_matrix(B_prod_x.reshape((dim, dim)))
        B_p_y = scipy.sparse.csc_matrix(B_prod_y.reshape((dim, dim)))

        B = B_p_x @ B_p_y

        B = scipy.sparse.csc_matrix(B)

    elif len(dims) == 3:

        B_x = bases[0].B
        B_y = bases[1].B
        B_z = bases[2].B
        B_prod_x = numpy.zeros(tuple(dims) * 2)  # filters last dim
        B_prod_y = numpy.zeros(tuple(dims) * 2)  # filters first dim
        B_prod_z = numpy.zeros(tuple(dims) * 2)  # filters first dim

        sh = B_prod_x.shape

        for i in range(B_prod_x.shape[1]):
            for j in range(B_prod_x.shape[2]):
                # for j in B_prod_x.shape[3]:
                B_prod_x[:, i, j, :, i, j] = B_x

        for i in range(B_prod_y.shape[0]):
            for j in range(B_prod_x.shape[2]):
                # for j in B_prod_x.shape[3]:
                B_prod_y[i, :, j, i, :, j] = B_y

        for i in range(B_prod_y.shape[0]):
            for j in range(B_prod_x.shape[1]):
                # for j in B_prod_x.shape[3]:
                B_prod_z[i, j, :, i, j, :] = B_z

        dim = numpy.prod(dims)

        B_p_x = scipy.sparse.csc_matrix(B_prod_x.reshape((dim, dim)))
        B_p_y = scipy.sparse.csc_matrix(B_prod_y.reshape((dim, dim)))
        B_p_z = scipy.sparse.csc_matrix(B_prod_z.reshape((dim, dim)))

        # C_p = (B_p_y@B_p_x)
        # C_p = (B_p_x@B_p_y)
        B = B_p_x @ B_p_y @ B_p_z
        # C_p = (B_p_z@B_p_y@B_p_x)

    return B


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


class SparseTensor:
    def __init__(self, indices, values, shape):
        """
        indices: N x d integers
        values:  N     floats
        """
        indices = numpy.array(indices, dtype=int)
        values = numpy.array(values, dtype=float)
        self.indices = indices
        self.values = values
        self.shape = shape

    @property
    def lin_indices(self):
        minds = tuple(self.indices.T)
        return numpy.ravel_multi_index(minds, self.shape)

    def reshape(self, shape):
        assert prod(shape) == prod(self.shape)
        new_indices = numpy.unravel_index(self.lin_indices, shape)
        new_indices = numpy.concatenate([a[:, None] for a in new_indices], axis=1)
        return SparseTensor(new_indices, self.values, shape)

    def as_spmatrix(self, dims=None):
        import scipy.sparse

        if dims is None:
            pp = prod(self.shape)
            P = int(sqrt(pp))
            Q = P
            assert pp == P ** 2
            dims = (P, Q)
        else:
            assert len(dims) == 2
        spmat = self.reshape(dims)
        data = spmat.values
        inds = spmat.indices
        return scipy.sparse.coo_matrix((data, (inds[:, 0], inds[:, 1])))

    def as_array(self, dims=None):
        return self.as_spmatrix(dims).toarray()


def dres_to_sparse(dres):
    import numpy

    n_m, N, n_x, junk = dres.shape
    assert n_x == junk
    big_N = n_m * N * n_x * n_m * N * n_x
    nnz = n_m * N * n_x * n_x
    inds = numpy.zeros((nnz, 6), dtype=int)
    vals = numpy.zeros(nnz, dtype=float)
    n = 0
    for i in range(n_m):
        for j in range(N):
            for k in range(n_x):
                for l in range(n_x):
                    inds[n, 0] = i
                    inds[n, 1] = j
                    inds[n, 2] = k
                    inds[n, 3] = i
                    inds[n, 4] = j
                    inds[n, 5] = l
                    vals[n] = dres[i, j, k, l]
                    n += 1
    return SparseTensor(inds, vals, (n_m, N, n_x, n_m, N, n_x))


@jit
def serial_solve(A, B, diagnose=True):

    sol = zeros_like(B)
    for i in range(sol.shape[0]):
        sol[i, :] = solve(A[i, :, :], B[i, :])

    return sol


def jres_to_sparse(jres):
    import numpy

    n_m, N, n_x, a, b = jres.shape
    assert a == n_m
    assert b == n_x
    big_N = n_m * N * n_x * n_m * N * n_x
    nnz = n_m * N * n_x * n_m * n_x
    inds = numpy.zeros((nnz, 6), dtype=float)
    vals = numpy.zeros(nnz, dtype=float)
    n = 0
    for i in range(n_m):
        for j in range(N):
            for k in range(n_x):
                for l in range(n_m):
                    for m in range(n_x):
                        inds[n, 0] = i
                        inds[n, 1] = j
                        inds[n, 2] = k
                        inds[n, 3] = l
                        inds[n, 4] = j
                        inds[n, 5] = m
                        vals[n] = jres[i, j, k, l, m]
                        n += 1
    return SparseTensor(inds, vals, (n_m, N, n_x, n_m, N, n_x))


def compact_ddx(res, fut_S):

    import scipy.sparse
    import scipy.sparse

    n_ms, N, n_x = res.shape

    Binv = numpy.linalg.inv(B)
    #         DDX = numpy.zeros((n_ms,N,n_x,n_ms,N,n_x))
    nnz = N * N * n_m * n_m * n_x
    indices = numpy.zeros((nnz, 6), dtype=int)
    vals = numpy.zeros(nnz, dtype=float)
    n = 0
    for i_m in range(n_ms):
        for i_M in range(n_ms):
            S = fut_S[i_m, :, i_M, :]
            Phi = tb.Phi(S).as_matrix()
            DD = (Phi @ Binv)[:, 1:-1]
            #             DD[abs(DD)<1e-6] = 0
            for i_x in range(n_x):
                # Phi(S) c = Phi(S) (B^{-1}) x
                # DDX[i_m,:,i_x,i_m,:,i_x] = DD
                for p in range(N):
                    for q in range(N):
                        indices[n, 0] = i_m
                        indices[n, 1] = p
                        indices[n, 2] = i_x
                        indices[n, 3] = i_M
                        indices[n, 4] = q
                        indices[n, 5] = i_x
                        vals[n] = DD[p, q]
                        n += 1
    indices = indices[:n, :]
    vals = vals[:n]
    return SparseTensor(indices, vals, (n_ms, N, n_x, n_ms, N, n_x))


#         return scipy.sparse.coo_matrix( DDX.reshape((bigN,bigN)) )


def construct_j2(jres):
    import numpy

    m_m, N, n_x, a, b = jres.shape
    assert a == n_m
    assert b == n_x
    big_N = n_m * N * n_x * n_m * N * n_x
    nnz = m_m * N * n_x * n_m * n_x
    inds = numpy.zeros((nnz, 6), dtype=float)
    vals = numpy.zeros(nnz, dtype=float)
    n = 0
    for i in range(n_m):
        for j in range(N):
            for k in range(n_x):
                for l in range(n_m):
                    for m in range(n_x):
                        inds[n, 0] = i
                        inds[n, 1] = j
                        inds[n, 2] = k
                        inds[n, 3] = l
                        inds[n, 4] = j
                        inds[n, 5] = m
                        vals[n] = jres[i, j, k, l, m]
                        n += 1
    return SparseTensor(inds, vals, (n_m, N, n_x, n_m, N, n_x))


from numpy import sqrt


from dolo.numeric.serial_operations import serial_multiplication as smul

#
# from interpolation.linear_bases.basis_uniform_cubic_splines import UniformSplineBasis
# from interpolation.linear_base    s.product import TensorBase
from dolo.numeric.serial_operations import serial_multiplication as smult


from numpy.linalg import solve
from numpy import zeros_like


class SmartJacobian:
    def __init__(self, res, dres, jres, fut_S, grid):
        self.res = res
        self.dres = dres
        self.jres = jres
        self.fut_S = fut_S
        self.n_m = res.shape[0]
        self.N = res.shape[1]
        self.n_x = res.shape[2]
        self.grid = grid
        self.__B__ = None

    def B(self):
        if self.__B__ is None:
            self.__B__ = self.get_filter()
        return self.__B__

    def get_filter(self):
        a = self.grid.a
        b = self.grid.b
        o = self.grid.orders

        bases = [UniformSplineBasis(a[i], b[i], o[i]) for i in range(len(o))]
        tb = TensorBase(bases)
        self.tb = tb

        B = B_prod(bases)

        return B

    @property
    def jac_1(self):

        n_ms, N, n_x = self.res.shape
        dres = self.dres
        bigN = n_ms * N * n_x
        DDRes = numpy.zeros((n_ms, N, n_x, n_ms, N, n_x))
        for n in range(N):
            for i_m in range(n_ms):
                DDRes[i_m, n, :, i_m, n, :] = dres[i_m, n, :, :]
        jac1 = DDRes.reshape((bigN, bigN))
        return jac1

    @property
    def j2_A(self):

        n_ms, N, n_x = self.res.shape
        jres = self.jres
        fut_S = self.fut_S
        bigN = n_ms * N * n_x

        JJRes = numpy.zeros((n_ms, N, n_x, n_ms, N, n_x))
        for n in range(N):
            # for i_m in range(n_ms):
            JJRes[:, n, :, :, n, :] = jres[:, n, :, :, :]

        return JJRes.reshape((bigN, bigN))

    @property
    def j2_B(self):

        n_ms, N, n_x = self.res.shape
        fut_S = self.fut_S
        bigN = n_ms * N * n_x
        fut_S = self.fut_S
        DDX = numpy.zeros((n_ms, N, n_x, n_ms, N, n_x))
        for i_x in range(n_x):
            for i_m in range(n_ms):
                for i_M in range(n_ms):
                    S = fut_S[i_m, :, i_M, :]
                    Phi = tb.Phi(S).as_matrix()
                    # Phi(S) c = Phi(S) (B^{-1}) x
                    # i_m -> i_M
                    DDX[i_m, :, i_x, i_m, :, i_x] = (Phi @ numpy.linalg.inv(B))[:, 1:-1]
        #                     DDX[i_M,:,i_x,i_m,:,i_x] = (Phi @ numpy.linalg.inv(B))[:,1:-1]

        return DDX.reshape((bigN, bigN))

    #     @property
    #     def jac_2(self):
    #         return self.j2_A @ self.j2_B

    @property
    def jac_2(self):

        B = self.get_filter()
        d = len(self.tb.bases)
        mdims = tuple([b.m for b in self.tb.bases])

        # works
        n_ms, N, n_x = self.res.shape
        #         jres = self.jres
        #         fut_S = self.fut_S
        bigN = n_ms * N * n_x
        jres = self.jres

        fut_S = self.fut_S
        try:
            B = B.todense()
        except:
            pass
        Binv = numpy.linalg.inv(B)
        mat = numpy.zeros((n_ms, N, n_x, n_ms, N, n_x))

        for i in range(n_ms):
            for j in range(n_ms):
                S = fut_S[i, :, j, :]
                Phi = self.tb.Phi(S).as_matrix()
                X = Phi @ Binv  # [:,1:-1]
                X = numpy.array(X)
                X = X.reshape((X.shape[0],) + mdims)
                if d == 1:
                    X = X[:, 1:-1]
                elif d == 2:
                    X = X[:, 1:-1, 1:-1]
                elif d == 3:
                    X = X[:, 1:-1, 1:-1, 1:-1]
                X = X.reshape((X.shape[0], -1))
                XX = numpy.zeros((N, n_x, N, n_x))
                for y in range(n_x):
                    XX[:, y, :, y] = X
                XX = XX.reshape((N * n_x, N * n_x))
                ff = jres[i, :, :, j, :]
                ff_m = numpy.zeros((N, n_x, N, n_x))
                for n in range(N):
                    ff_m[n, :, n, :] = ff[n, :, :]
                m = ff_m.reshape((N * n_x, N * n_x)) @ XX
                m = m.reshape((N, n_x, N, n_x))
                mat[i, :, :, j, :, :] = m
        mat = mat.reshape((bigN, bigN))
        mat[abs(mat) < 1e-6] = 0
        import scipy.sparse

        mat = scipy.sparse.coo_matrix(mat)
        return mat

    #     @property
    #     def jac_2(self):
    #         return self.j2
    # #         return self.j2_A @ self.j2_B

    @property
    def jac(self):
        return self.jac_1 + self.jac_2

    def solve(self, rr):
        import numpy.linalg

        return numpy.linalg.solve(self.jac, rr.ravel()).reshape(rr.shape)

    def solve_sp(self, rr):
        import scipy.sparse
        from scipy.sparse.linalg import spsolve

        jj = scipy.sparse.csr_matrix(self.jac)
        res = spsolve(jj, rr.ravel())
        return res.reshape(rr.shape)

    #         return numpy.linalg.solve(self.jac, rr.ravel()).reshape(rr.shape)

    def solve_smart(
        self, rr, tol=1e-10, maxit=1000, verbose=False, filt=None, scale=1.0
    ):
        n_m, N, n_x = self.res.shape
        fut_S = self.fut_S
        bigN = n_m * N * n_x
        dres = self.dres
        jres = self.jres
        grid = self.grid
        fut_S = self.fut_S
        sol, nn = invert_jac(
            rr * scale,
            dres,
            jres,
            fut_S,
            n_m,
            N,
            n_x,
            grid,
            tol=tol,
            maxit=maxit,
            verbose=verbose,
            filt=filt,
        )
        sol /= scale
        return sol

    def solve_ind(self, rr, tol=1e-12):
        # works only if there are no shocks
        jac1 = self.jac_1
        j2_A = self.j2_A
        j2_B = self.j2_B
        M = jac1
        N = -numpy.linalg.solve(jac1, j2_A @ j2_B)
        # I need to prove sp(N)<[0,1[

        abs(numpy.linalg.eig(N)[0]).max()
        abs(numpy.linalg.eig(N)[0]).min()

        term = numpy.linalg.solve(M, rr.ravel())
        tot = term
        for i in range(10000):
            term = N @ term
            err = abs(term).max()
            tot = tot + term
            if err < tol:
                break
        return tot.reshape(rr.shape)


#### utilities to compute perturbations
from dolo.numeric.extern.qz import qzordered
from numpy.linalg import solve
import numpy as np


def classical_perturbation(g_s, g_x, f_s, f_x, f_S, f_X):
    n_s = g_s.shape[0]  # number of controls
    n_x = g_x.shape[1]  # number of states
    n_v = n_s + n_x

    A = row_stack(
        [column_stack([eye(n_s), zeros((n_s, n_x))]), column_stack([-f_S, -f_X])]
    )

    B = row_stack([column_stack([g_s, g_x]), column_stack([f_s, f_x])])

    [S, T, Q, Z, eigval] = qzordered(A, B, 1.0 - 1e-8)

    Q = Q.real  # is it really necessary ?
    Z = Z.real

    diag_S = np.diag(S)
    diag_T = np.diag(T)

    tol_geneigvals = 1e-10

    Z11 = Z[:n_s, :n_s]
    # Z12 = Z[:n_s, n_s:]
    Z21 = Z[n_s:, :n_s]
    # Z22 = Z[n_s:, n_s:]
    # S11 = S[:n_s, :n_s]
    # T11 = T[:n_s, :n_s]

    # first order solution
    # P = (solve(S11.T, Z11.T).T @ solve(Z11.T, T11.T).T)
    C = solve(Z11.T, Z21.T).T
    return C, eigval


@jit(nopython=True)
def simple_max(T):
    mm = 0.0
    p, q = T.shape
    for i in range(p):
        for j in range(q):
            e = abs(T[i, j])
            if e > mm:
                mm = e
    return mm


@jit(nopython=True)
def newtonator(g_s, g_x, f_s, f_x, f_S, f_X):  # , tol=1e-8, inner_tol=1e-8, X=None):

    # solves linear approximation using mix of time-iteration and newton algorithm

    tol = 1e-14
    inner_tol = 1e-12
    inner_maxit = 1000

    X = numpy.zeros((n_x, n_s))
    Y = X

    Ainv = numpy.zeros((n_x, n_x))
    direction = numpy.zeros_like(X)

    maxit = 1000
    err = 1.0
    it = 0

    K1 = f_s + f_S @ g_s
    K2 = f_x + f_S @ g_x
    res = K1 + K2 @ X + f_X @ Y @ g_s + f_X @ Y @ g_x @ X

    while err > tol and it < maxit:

        it += 1

        A = K2 + f_X @ Y @ g_x
        B = f_X

        Ainv = inv(A)

        C = g_s + g_x @ X
        M = -Ainv @ B
        ddx = Ainv @ res
        direction[:, :] = ddx
        for i in range(inner_maxit):
            ddx = M @ ddx @ C
            #             inner_err = (abs(ddx).max())
            #             inner_err = (abs(ddx).max())
            inner_err = simple_max(ddx)
            direction[:, :] += ddx
            if inner_err < inner_tol:
                break

        X -= direction
        Y = X
        res = K1 + K2 @ X + f_X @ Y @ g_s + f_X @ Y @ g_x @ X
        err = abs(res).max()

    #         doesn't seem to make any change
    #          err = abs(Jac).max()
    #         for t in backsteps:
    #             Y[:,:] = X[:,:] - direction[:,:]*t
    #             res = K1 + K2@Y + f_X@Y@g_s + f_X@Y@g_x@Y
    #             nerr = abs(res).max()
    #             if nerr<err:
    #                 err = nerr
    #                 break
    #         X[:,:] = Y[:,:]

    return X


def spectral_radius(g_s, g_x, f_s, f_x, f_S, f_X, X=None):

    tol = 1e-14
    inner_tol = 1e-12
    inner_maxit = 1000

    if X is None:
        X = numpy.zeros((n_x, n_s))

    Y = X

    Ainv = numpy.zeros((n_x, n_x))
    direction = numpy.zeros_like(X)

    maxit = 1000
    err = 1.0
    it = 0

    errors = []
    #     res = K1 + K2@X + f_X@Y@g_s + f_X@Y@g_x@ X

    res = numpy.random.random((n_x, n_s))

    K1 = f_s + f_S @ g_s
    K2 = f_x + f_S @ g_x
    A = K2 + f_X @ Y @ g_x
    B = f_X

    Ainv = inv(A)

    C = g_s + g_x @ X
    M = -Ainv @ B
    ddx = Ainv @ res
    for i in range(inner_maxit):
        ddx = M @ ddx @ C
        errors.append(abs(ddx.max()))

    errors = numpy.array(errors)
    #     return errors
    rats = errors[1:] / errors[:-1]
    return rats[-1]
