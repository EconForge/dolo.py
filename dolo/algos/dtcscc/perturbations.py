import numpy as np
from numpy import column_stack, dot, eye, row_stack, zeros
from numpy.linalg import solve

from dolo.algos.dtcscc.steady_state import find_deterministic_equilibrium
from dolo.numeric.extern.qz import qzordered
from dolo.numeric.taylor_expansion import TaylorExpansion as CDR


class GeneralizedEigenvaluesError(Exception):

    def __init__(self, A=None, B=None, eigval=None, cutoff=None, n_states=None):

        self.A = A
        self.B = B
        self.eigval = eigval
        self.eigval_sorted = sorted(eigval)
        self.cutoff = cutoff
        self.n_states = n_states

class GeneralizedEigenvaluesDefinition(GeneralizedEigenvaluesError):

    def __str__(self):

        return "Generalized Eigenvalues imply a 0/0 division. Undefined solution."


class GeneralizedEigenvaluesSelectionError(GeneralizedEigenvaluesError):

    def __str__(self):

        return "Impossible to select the {} bigger eigenvalues in a unique way.".format(self.n_states)



def approximate_controls(model, verbose=False, steady_state=None, eigmax=1.0-1e-6,
                         solve_steady_state=False, order=1):
    '''
    Compute first order approximation of optimal controls

    Parameters
    ----------

    model : NumericModel
        Model to be solved

    verbose : boolean
        If True: displays number of contracting eigenvalues

    steady_state : ndarray
        Use supplied steady-state value to compute the approximation.
        The routine doesn't check whether it is really a solution or not.

    solve_steady_state : boolean
        Use nonlinear solver to find the steady-state

    orders : {1}
        Approximation order. (Currently, only first order is supported).

    Returns
    -------

    TaylorExpansion :
        Decision Rule for the optimal controls around the steady-state.

    '''

    if order > 1:
        raise Exception("Not implemented.")

    f = model.functions['arbitrage']
    g = model.functions['transition']

    if steady_state is not None:
        calib = steady_state
    else:
        calib = model.calibration

    if solve_steady_state:
        calib = find_deterministic_equilibrium(model)

    p = calib['parameters']
    s = calib['states']
    x = calib['controls']
    e = calib['shocks']

    distrib = model.get_distribution()
    sigma = distrib.sigma

    l = g(s, x, e, p, diff=True)
    [junk, g_s, g_x, g_e] = l[:4]  # [el[0,...] for el in l[:4]]

    l = f(s, x, e, s, x, p, diff=True)
    [res, f_s, f_x, f_e, f_S, f_X] = l  # [el[0,...] for el in l[:6]]

    n_s = g_s.shape[0]           # number of controls
    n_x = g_x.shape[1]   # number of states
    n_e = g_e.shape[1]
    n_v = n_s + n_x

    A = row_stack([
        column_stack([eye(n_s), zeros((n_s, n_x))]),
        column_stack([-f_S    , -f_X             ])
    ])

    B = row_stack([
        column_stack([g_s, g_x]),
        column_stack([f_s, f_x])
    ])


    [S, T, Q, Z, eigval] = qzordered(A, B, 1.0-1e-8)

    Q = Q.real  # is it really necessary ?
    Z = Z.real

    diag_S = np.diag(S)
    diag_T = np.diag(T)

    tol_geneigvals = 1e-10

    try:
        ok = sum((abs(diag_S) < tol_geneigvals) *
                 (abs(diag_T) < tol_geneigvals)) == 0
        assert(ok)
    except Exception as e:
        raise GeneralizedEigenvaluesError(diag_S=diag_S, diag_T=diag_T)

    if max(eigval[:n_s]) >= 1 and min(eigval[n_s:]) < 1:
        # BK conditions are met
        pass
    else:
        eigval_s = sorted(eigval, reverse=True)
        ev_a = eigval_s[n_s-1]
        ev_b = eigval_s[n_s]
        cutoff = (ev_a - ev_b)/2
        if not ev_a > ev_b:
            raise GeneralizedEigenvaluesSelectionError(
                    A=A, B=B, eigval=eigval, cutoff=cutoff,
                    diag_S=diag_S, diag_T=diag_T, n_states=n_s
                )
        import warnings
        if cutoff > 1:
            warnings.warn("Solution is not convergent.")
        else:
            warnings.warn("There are multiple convergent solutions. The one with the smaller eigenvalues was selected.")
        [S, T, Q, Z, eigval] = qzordered(A, B, cutoff)

    Z11 = Z[:n_s, :n_s]
    # Z12 = Z[:n_s, n_s:]
    Z21 = Z[n_s:, :n_s]
    # Z22 = Z[n_s:, n_s:]
    # S11 = S[:n_s, :n_s]
    # T11 = T[:n_s, :n_s]

    # first order solution
    # P = (solve(S11.T, Z11.T).T @ solve(Z11.T, T11.T).T)
    C = solve(Z11.T, Z21.T).T
    Q = g_e

    s = s.ravel()
    x = x.ravel()

    A = g_s + g_x @ C
    B = g_e

    dr = CDR([s, x, C])
    dr.A = A
    dr.B = B
    dr.sigma = sigma

    return dr

if __name__ == '__main__':

    from dolo import *
    from os.path import abspath, dirname, join

    dolo_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
    model = yaml_import(join(dolo_dir, "examples", "models", "rbc.yaml"))
    model.set_calibration(dumb=1)

    from dolo.algos.dtcscc.perturbations import approximate_controls

    dr = approximate_controls(model)
    print(dr)
