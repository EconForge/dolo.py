from __future__ import division, print_function, absolute_import

import numpy
from numpy import real_if_close, where
from scipy.linalg import ordqz


def qzordered(A, B, crit=1.0):
    "Eigenvalues bigger than crit are sorted in the top-left."

    TOL = 1e-10

    def select(alpha, beta):
        return alpha ** 2 > crit * beta ** 2

    [S, T, alpha, beta, U, V] = ordqz(A, B, output="real", sort=select)

    eigval = abs(numpy.diag(S) / numpy.diag(T))

    return [S, T, U, V, eigval]


def test_qzordered():

    import numpy

    N = 202
    A = numpy.random.random((N, N))
    B = numpy.random.random((N, N))
    [S, T, U, V, eigval] = qzordered(A, B, 100)


if __name__ == "__main__":
    test_qzordered()
