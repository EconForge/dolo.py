def get_derivatives(model, steady_state=None):
    """The sum of two numbers."""

    from dolo.numeric.processes import VAR1, IIDProcess
    from dolo.numeric.distribution import Normal

    import numpy as np

    if steady_state is None:
        steady_state = model.calibration

    m = steady_state["exogenous"]
    s = steady_state["states"]
    x = steady_state["controls"]
    p = steady_state["parameters"]

    f = model.functions["arbitrage"]
    g = model.functions["transition"]

    n_m = len(m)
    n_s = len(s)
    n_x = len(x)

    f_, f_m, f_s, f_x, f_M, f_S, f_X = f(m, s, x, m, s, x, p, diff=True)
    g_, g_m, g_s, g_x, g_M = g(m, s, x, m, p, diff=True)

    process = model.exogenous

    if isinstance(process, VAR1):
        A = process.rho
        B = np.eye(A.shape[0])
        G_s = np.row_stack(
            [
                np.column_stack([A, np.zeros((n_m, n_s))]),
                np.column_stack([g_m + g_M @ A, g_s]),
            ]
        )
        G_x = np.row_stack([np.zeros((n_m, n_x)), g_x])
        G_e = g_m @ B
        F_s = np.column_stack([f_m, f_s])
        F_S = np.column_stack([f_M, f_S])
        F_x = f_x
        F_X = f_X
    elif isinstance(process, IIDProcess):
        G_s = g_s
        G_x = g_x
        G_e = g_m
        F_s = f_s
        F_S = f_S
        F_x = f_x
        F_X = f_X
    else:
        raise Exception(f"Not implemented: perturbation for shock {process.__class__}")

    return G_s, G_x, G_e, F_s, F_x, F_S, F_X


from dolo.numeric.grids import PointGrid, EmptyGrid
from dolo.numeric.decision_rule import CallableDecisionRule


class BivariateTaylor(CallableDecisionRule):
    def __init__(self, m_bar, s_bar, x_bar, C_m, C_s):

        self.endo_grid = PointGrid(s_bar)
        if C_m is None:
            self.exo_grid = EmptyGrid()
        else:
            self.exo_grid = PointGrid(m_bar)

        self.m_bar = m_bar
        self.s_bar = s_bar
        self.x_bar = x_bar
        self.C_m = C_m
        self.C_s = C_s

    def eval_s(self, s):

        s = np.array(s)
        if s.ndim == 1:
            return self.eval_s(s[None, :])[0, :]
        return self.eval_ms(None, s)

    def eval_ms(self, m, s):

        m = np.array(m)
        s = np.array(s)
        if m.ndim == 1 and s.ndim == 1:
            if self.C_m is not None:
                return self.eval_ms(m[None, :], s[None, :])[0, :]
            else:
                return self.eval_ms(None, s[None, :])[0, :]
        elif m.ndim == 1:
            m = m[None, :]
        elif s.ndim == 1:
            s = s[None, :]

        C_m = self.C_m
        C_s = self.C_s
        if C_m is not None:
            dm = m - self.m_bar[None, :]
            ds = s - self.s_bar[None, :]
            return self.x_bar[None, :] + dm @ C_m.T + ds @ C_s.T
        else:
            ds = s - self.s_bar[None, :]
            return self.x_bar[None, :] + ds @ C_s.T


import numpy as np
from numpy import column_stack, dot, eye, row_stack, zeros
from numpy.linalg import solve

from dolo.numeric.extern.qz import qzordered


def approximate_1st_order(g_s, g_x, g_e, f_s, f_x, f_S, f_X):

    n_s = g_s.shape[0]  # number of controls
    n_x = g_x.shape[1]  # number of states
    n_e = g_e.shape[1]
    n_v = n_s + n_x

    A = row_stack(
        [column_stack([eye(n_s), zeros((n_s, n_x))]), column_stack([-f_S, -f_X])]
    )

    B = row_stack([column_stack([g_s, g_x]), column_stack([f_s, f_x])])

    [S, T, Q, Z, eigval] = qzordered(A, B, 1.0 - 1e-8)

    Z = Z.real

    diag_S = np.diag(S)
    diag_T = np.diag(T)

    tol_geneigvals = 1e-10

    try:
        ok = sum((abs(diag_S) < tol_geneigvals) * (abs(diag_T) < tol_geneigvals)) == 0
        assert ok
    except Exception as e:
        raise GeneralizedEigenvaluesError(diag_S=diag_S, diag_T=diag_T)

    eigval_s = sorted(eigval, reverse=False)
    if max(eigval[:n_s]) >= 1 and min(eigval[n_s:]) < 1:
        # BK conditions are met
        pass
    else:
        ev_a = eigval_s[n_s - 1]
        ev_b = eigval_s[n_s]
        cutoff = (ev_a - ev_b) / 2
        if not ev_a > ev_b:
            raise GeneralizedEigenvaluesSelectionError(
                A=A,
                B=B,
                eigval=eigval,
                cutoff=cutoff,
                diag_S=diag_S,
                diag_T=diag_T,
                n_states=n_s,
            )
        import warnings

        if cutoff > 1:
            warnings.warn("Solution is not convergent.")
        else:
            warnings.warn(
                "There are multiple convergent solutions. The one with the smaller eigenvalues was selected."
            )
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

    A = g_s + g_x @ C
    B = g_e

    return C, eigval_s


from .results import AlgoResult, PerturbationResult


def perturb(
    model,
    verbose=False,
    steady_state=None,
    eigmax=1.0 - 1e-6,
    solve_steady_state=False,
    order=1,
    details=True,
):
    """Compute first order approximation of optimal controls

    Parameters:
    -----------

    model: NumericModel
        Model to be solved

    verbose: boolean
        If True: displays number of contracting eigenvalues

    steady_state: ndarray
        Use supplied steady-state value to compute the approximation.
        The routine doesn't check whether it is really a solution or not.

    solve_steady_state: boolean
        Use nonlinear solver to find the steady-state

    orders: {1}
        Approximation order. (Currently, only first order is supported).

    Returns:
    --------

    TaylorExpansion:
        Decision Rule for the optimal controls around the steady-state.

    """

    if order > 1:
        raise Exception("Not implemented.")

    if steady_state is None:
        steady_state = model.calibration

    G_s, G_x, G_e, F_s, F_x, F_S, F_X = get_derivatives(
        model, steady_state=steady_state
    )

    C, eigvals = approximate_1st_order(G_s, G_x, G_e, F_s, F_x, F_S, F_X)

    m = steady_state["exogenous"]
    s = steady_state["states"]
    x = steady_state["controls"]

    from dolo.numeric.processes import VAR1, IIDProcess
    from dolo.numeric.distribution import MvNormal

    process = model.exogenous

    if isinstance(process, VAR1):
        C_m = C[:, : len(m)]
        C_s = C[:, len(m) :]
    elif isinstance(process, IIDProcess):
        C_m = None
        C_s = C
    dr = BivariateTaylor(m, s, x, C_m, C_s)
    if not details:
        return dr
    else:
        return PerturbationResult(
            dr,
            eigvals,
            True,  # otherwise an Exception should have been raised already
            True,  # otherwise an Exception should have been raised already
            True,  # otherwise an Exception should have been raised already
        )
