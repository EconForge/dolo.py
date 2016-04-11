from __future__ import division, print_function, absolute_import

import numpy
from numpy import real_if_close,where

def qzordered(A,B,crit=1.0+1e-6):
    "Eigenvalues bigger than crit are sorted in the top-left."

    TOL = 1e-10

    def select(alpha, beta):
        return alpha**2>crit*beta**2

    [S,T,alpha,beta,U,V] = ordqz(A,B,output='real',sort=select)

    eigval = abs(numpy.diag(S)/numpy.diag(T))

    return [S,T,U,V,eigval]


def ordqz(A, B, sort='lhp', output='real', overwrite_a=False,
          overwrite_b=False, check_finite=True):
    """
    QZ decomposition for a pair of matrices with reordering.

    .. versionadded:: 0.17.0

    Parameters
    ----------
    A : (N, N) array_like
        2d array to decompose
    B : (N, N) array_like
        2d array to decompose
    sort : {callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
        Specifies whether the upper eigenvalues should be sorted.  A callable
        may be passed that, given a eigenvalue, returns a boolean denoting
        whether the eigenvalue should be sorted to the top-left (True). For
        real matrix pairs, the sort function takes three real arguments
        (alphar, alphai, beta). The eigenvalue
        ``x = (alphar + alphai*1j)/beta``.  For complex matrix pairs or
        output='complex', the sort function takes two complex arguments
        (alpha, beta). The eigenvalue ``x = (alpha/beta)``.
        Alternatively, string parameters may be used:

            - 'lhp'   Left-hand plane (x.real < 0.0)
            - 'rhp'   Right-hand plane (x.real > 0.0)
            - 'iuc'   Inside the unit circle (x*x.conjugate() < 1.0)
            - 'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

    output : str {'real','complex'}, optional
        Construct the real or complex QZ decomposition for real matrices.
        Default is 'real'.
    overwrite_a : bool, optional
        If True, the contents of A are overwritten.
    overwrite_b : bool, optional
        If True, the contents of B are overwritten.
    check_finite : bool, optional
        If true checks the elements of `A` and `B` are finite numbers. If
        false does no checking and passes matrix through to
        underlying algorithm.

    Returns
    -------
    AA : (N, N) ndarray
        Generalized Schur form of A.
    BB : (N, N) ndarray
        Generalized Schur form of B.
    alpha : (N,) ndarray
        alpha = alphar + alphai * 1j. See notes.
    beta : (N,) ndarray
        See notes.
    Q : (N, N) ndarray
        The left Schur vectors.
    Z : (N, N) ndarray
        The right Schur vectors.

    Notes
    -----
    On exit, ``(ALPHAR(j) + ALPHAI(j)*i)/BETA(j), j=1,...,N``, will be the
    generalized eigenvalues.  ``ALPHAR(j) + ALPHAI(j)*i`` and
    ``BETA(j),j=1,...,N`` are the diagonals of the complex Schur form (S,T)
    that would result if the 2-by-2 diagonal blocks of the real generalized
    Schur form of (A,B) were further reduced to triangular form using complex
    unitary transformations. If ALPHAI(j) is zero, then the j-th eigenvalue is
    real; if positive, then the ``j``-th and ``(j+1)``-st eigenvalues are a complex
    conjugate pair, with ``ALPHAI(j+1)`` negative.

    See also
    --------
    qz
    """


    import warnings

    import numpy as np
    from numpy import asarray_chkfinite

    from scipy.linalg.misc import LinAlgError, _datacopied
    from scipy.linalg.lapack import get_lapack_funcs

    from scipy._lib.six import callable

    from scipy.linalg._decomp_qz import _qz, _select_function

    #NOTE: should users be able to set these?
    lwork = None
    result, typ = _qz(A, B, output=output, lwork=lwork, sort=None,
                      overwrite_a=overwrite_a, overwrite_b=overwrite_b,
                      check_finite=check_finite)
    AA, BB, Q, Z = result[0], result[1], result[-4], result[-3]
    if typ not in 'cz':
        alpha, beta = result[3] + result[4]*1.j, result[5]
    else:
        alpha, beta = result[3], result[4]

    sfunction = _select_function(sort)
    select = sfunction(alpha, beta)

    tgsen, = get_lapack_funcs(('tgsen',), (AA, BB))

    if lwork is None or lwork == -1:
        result = tgsen(select, AA, BB, Q, Z, lwork=-1)
        lwork = result[-3][0].real.astype(np.int)
        # looks like wrong value passed to ZTGSYL if not
        lwork += 1

    liwork = None
    if liwork is None or liwork == -1:
        result = tgsen(select, AA, BB, Q, Z, liwork=-1)
        liwork = result[-2][0]

    result = tgsen(select, AA, BB, Q, Z, lwork=lwork, liwork=liwork)

    info = result[-1]
    if info < 0:
        raise ValueError("Illegal value in argument %d of tgsen" % -info)
    elif info == 1:
        raise ValueError("Reordering of (A, B) failed because the transformed"
                         " matrix pair (A, B) would be too far from "
                         "generalized Schur form; the problem is very "
                         "ill-conditioned. (A, B) may have been partially "
                         "reorded. If requested, 0 is returned in DIF(*), "
                         "PL, and PR.")

    # for real results has a, b, alphar, alphai, beta, q, z, m, pl, pr, dif,
    # work, iwork, info
    if typ in ['f', 'd']:
        alpha = result[2] + result[3] * 1.j
        return (result[0], result[1], alpha, result[4], result[5], result[6])
    # for complex results has a, b, alpha, beta, q, z, m, pl, pr, dif, work,
    # iwork, info
    else:
        return result[0], result[1], result[2], result[3], result[4], result[5]



def test_qzordered():

    import numpy
    N = 202
    A = numpy.random.random((N,N))
    B = numpy.random.random((N,N))
    [S,T,U,V,eigval] = qzordered(A,B, 100)

if __name__ == '__main__':
    test_qzordered()
