from dolo.numeric.taylor_expansion import CDR
from dolo.numeric.extern.qz import qzordered
from dolo.compiler.derivatives import get_model_derivatives
from typing import List
from numpy import ndarray


def perturb(model, order=1, return_dr=True, steady_state=None, verbose=True):

    from dolo.numeric.processes import IIDProcess

    # here we should check that exogenous don't appear with time t-1 in transition
    # and with time t+1 in arbitrage.
    assert isinstance(model.exogenous, IIDProcess)

    import numpy

    if steady_state is None:
        calibration = model.calibration
    else:
        calibration = steady_state

    [f, g] = get_model_derivatives(model, order=order, calibration=calibration)
    sigma = model.exogenous.Σ

    problem = PerturbationProblem(f, g, sigma)

    pert_sol = state_perturb(problem, verbose=verbose)

    controls_ss = calibration["controls"]
    states_ss = calibration["states"]

    n_s = len(model.symbols["states"])
    n_c = len(model.symbols["controls"])

    if order == 1:
        if return_dr:
            S_bar = numpy.array(states_ss)
            X_bar = numpy.array(controls_ss)
            # add transitions of states to the d.r.

            X_s = pert_sol[0]
            A = g[1][:, :n_s] + numpy.dot(g[1][:, n_s : n_s + n_c], X_s)
            B = g[1][:, n_s + n_c :]
            dr = CDR(S_bar, X_bar, X_s)
            dr.A = A
            dr.B = B
            dr.sigma = sigma
            return dr

        return [controls_ss] + pert_sol

    if order == 2:
        [[X_s, X_ss], [X_tt]] = pert_sol
        X_bar = controls_ss + X_tt / 2
        if return_dr:
            S_bar = states_ss
            S_bar = numpy.array(S_bar)
            X_bar = numpy.array(X_bar)
            dr = CDR(S_bar, X_bar, X_s, X_ss)
            A = g[1][:, :n_s] + numpy.dot(g[1][:, n_s : n_s + n_c], X_s)
            B = g[1][:, n_s + n_c :]
            dr.sigma = sigma
            dr.A = A
            dr.B = B
            return dr
        return [X_bar, X_s, X_ss]

    if order == 3:
        [[X_s, X_ss, X_sss], [X_tt, X_stt]] = pert_sol
        X_bar = controls_ss + X_tt / 2
        X_s = X_s + X_stt / 2
        if return_dr:
            S_bar = states_ss
            dr = CDR(S_bar, X_bar, X_s, X_ss, X_sss)
            dr.sigma = sigma
            return dr
        return [X_bar, X_s, X_ss, X_sss]


class PerturbationProblem:
    def __init__(self, f: List[ndarray], g: List[ndarray], sigma: ndarray):
        self.f = f
        self.g = g
        self.sigma = sigma
        assert len(f) == len(g)

    @property
    def order(self):
        return len(self.f) - 1


approximate_controls = perturb


def state_perturb(problem: PerturbationProblem, verbose=True):
    """Computes a Taylor approximation of decision rules, given the supplied derivatives.

    The original system is assumed to be in the the form:

    .. math::

        E_t f(s_t,x_t,s_{t+1},x_{t+1})

        s_t = g(s_{t-1},x_{t-1}, \\lambda \\epsilon_t)

    where :math:`\\lambda` is a scalar scaling down the risk.  the solution is a function :math:`\\varphi` such that:

    .. math::

        x_t = \\varphi ( s_t, \\sigma )

    The user supplies, a list of derivatives of f and g.

    :param f_fun: list of derivatives of f [order0, order1, order2, ...]
    :param g_fun: list of derivatives of g [order0, order1, order2, ...]
    :param sigma: covariance matrix of :math:`\\epsilon_t`


    Assuming :math:`s_t` ,  :math:`x_t` and :math:`\\epsilon_t` are vectors of size
    :math:`n_s`, :math:`n_x`  and :math:`n_x`  respectively.
    In general the derivative of order :math:`i` of :math:`f`  is a multimensional array of size :math:`n_x \\times (N, ..., N)`
    with :math:`N=2(n_s+n_x)` repeated :math:`i` times (possibly 0).
    Similarly the derivative of order :math:`i` of :math:`g`  is a multidimensional array of size :math:`n_s \\times (M, ..., M)`
    with :math:`M=n_s+n_x+n_2` repeated :math:`i` times (possibly 0).
    """

    import numpy as np
    from numpy.linalg import solve

    approx_order = problem.order  # order of approximation

    [f0, f1] = problem.f[:2]
    [g0, g1] = problem.g[:2]
    sigma = problem.sigma

    n_x = f1.shape[0]  # number of controls
    n_s = f1.shape[1] // 2 - n_x  # number of states
    n_e = g1.shape[1] - n_x - n_s
    n_v = n_s + n_x

    f_s = f1[:, :n_s]
    f_x = f1[:, n_s : n_s + n_x]
    f_snext = f1[:, n_v : n_v + n_s]
    f_xnext = f1[:, n_v + n_s :]

    g_s = g1[:, :n_s]
    g_x = g1[:, n_s : n_s + n_x]
    g_e = g1[:, n_v:]

    A = np.row_stack(
        [
            np.column_stack([np.eye(n_s), np.zeros((n_s, n_x))]),
            np.column_stack([-f_snext, -f_xnext]),
        ]
    )
    B = np.row_stack([np.column_stack([g_s, g_x]), np.column_stack([f_s, f_x])])

    [S, T, Q, Z, eigval] = qzordered(A, B, 1.0 - 1e-8)

    Q = Q.real  # is it really necessary ?
    Z = Z.real

    diag_S = np.diag(S)
    diag_T = np.diag(T)

    tol_geneigvals = 1e-10

    try:
        ok = sum((abs(diag_S) < tol_geneigvals) * (abs(diag_T) < tol_geneigvals)) == 0
        assert ok
    except Exception as e:
        raise GeneralizedEigenvaluesError(diag_S=diag_S, diag_T=diag_T)

    if max(eigval[:n_s]) >= 1 and min(eigval[n_s:]) < 1:
        # BK conditions are met
        pass
    else:
        eigval_s = sorted(eigval, reverse=True)
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
    S11 = S[:n_s, :n_s]
    T11 = T[:n_s, :n_s]

    # first order solution
    C = solve(Z11.T, Z21.T).T
    P = np.dot(solve(S11.T, Z11.T).T, solve(Z11.T, T11.T).T)
    Q = g_e

    # if False:
    #     from numpy import dot
    #     test = f_s + f_x @ C + f_snext @ (g_s + g_x @ C) + f_xnext @ C @ (g_s + g_x @ C)
    #     print('Error: ' + str(abs(test).max()))

    if approx_order == 1:
        return [C]

    # second order solution
    from dolo.numeric.tensor import sdot, mdot
    from numpy import dot
    from dolo.numeric.matrix_equations import solve_sylvester

    f2 = problem.f[2]
    g2 = problem.g[2]
    g_ss = g2[:, :n_s, :n_s]
    g_sx = g2[:, :n_s, n_s:n_v]
    g_xx = g2[:, n_s:n_v, n_s:n_v]

    X_s = C

    V1_3 = g_s + dot(g_x, X_s)
    V1 = np.row_stack([np.eye(n_s), X_s, V1_3, X_s @ V1_3])

    K2 = g_ss + 2 * sdot(g_sx, X_s) + mdot(g_xx, X_s, X_s)
    A = f_x + dot(f_snext + dot(f_xnext, X_s), g_x)
    B = f_xnext
    C = V1_3
    D = mdot(f2, V1, V1) + sdot(f_snext + dot(f_xnext, X_s), K2)

    X_ss = solve_sylvester(A, B, C, D)

    #    test = sdot( A, X_ss ) + sdot( B,  mdot(X_ss,V1_3,V1_3) ) + D

    g_ee = g2[:, n_v:, n_v:]

    v = np.row_stack([g_e, dot(X_s, g_e)])

    K_tt = mdot(f2[:, n_v:, n_v:], v, v)
    K_tt += sdot(f_snext + dot(f_xnext, X_s), g_ee)
    K_tt += mdot(sdot(f_xnext, X_ss), g_e, g_e)
    K_tt = np.tensordot(K_tt, sigma, axes=((1, 2), (0, 1)))

    L_tt = f_x + dot(f_snext, g_x) + dot(f_xnext, dot(X_s, g_x) + np.eye(n_x))
    X_tt = solve(L_tt, -K_tt)

    if approx_order == 2:
        return [[X_s, X_ss], [X_tt]]

    # third order solution

    f3 = problem.f[3]
    g3 = problem.g[3]
    g_sss = g3[:, :n_s, :n_s, :n_s]
    g_ssx = g3[:, :n_s, :n_s, n_s:n_v]
    g_sxx = g3[:, :n_s, n_s:n_v, n_s:n_v]
    g_xxx = g3[:, n_s:n_v, n_s:n_v, n_s:n_v]

    V2_3 = K2 + sdot(g_x, X_ss)
    V2 = np.row_stack(
        [np.zeros((n_s, n_s, n_s)), X_ss, V2_3, dot(X_s, V2_3) + mdot(X_ss, V1_3, V1_3)]
    )

    K3 = g_sss + 3 * sdot(g_ssx, X_s) + 3 * mdot(g_sxx, X_s, X_s) + 2 * sdot(g_sx, X_ss)
    K3 += 3 * mdot(g_xx, X_ss, X_s) + mdot(g_xxx, X_s, X_s, X_s)
    L3 = 3 * mdot(X_ss, V1_3, V2_3)

    # A = f_x + dot( f_snext + dot(f_xnext,X_s), g_x) # same as before
    # B = f_xnext # same
    # C = V1_3 # same
    D = (
        mdot(f3, V1, V1, V1)
        + 3 * mdot(f2, V2, V1)
        + sdot(f_snext + dot(f_xnext, X_s), K3)
    )
    D += sdot(f_xnext, L3)

    X_sss = solve_sylvester(A, B, C, D)

    # now doing sigma correction with sigma replaced by l in the subscripts

    g_se = g2[:, :n_s, n_v:]
    g_xe = g2[:, n_s:n_v, n_v:]

    g_see = g3[:, :n_s, n_v:, n_v:]
    g_xee = g3[:, n_s:n_v, n_v:, n_v:]

    W_l = np.row_stack([g_e, dot(X_s, g_e)])

    I_e = np.eye(n_e)

    V_sl = g_se + mdot(g_xe, X_s, np.eye(n_e))

    W_sl = np.row_stack([V_sl, mdot(X_ss, V1_3, g_e) + sdot(X_s, V_sl)])

    K_ee = mdot(f3[:, :, n_v:, n_v:], V1, W_l, W_l)
    K_ee += 2 * mdot(f2[:, n_v:, n_v:], W_sl, W_l)

    # stochastic part of W_ll

    SW_ll = np.row_stack([g_ee, mdot(X_ss, g_e, g_e) + sdot(X_s, g_ee)])

    DW_ll = np.concatenate([X_tt, dot(g_x, X_tt), dot(X_s, sdot(g_x, X_tt)) + X_tt])

    K_ee += mdot(f2[:, :, n_v:], V1, SW_ll)

    K_ = np.tensordot(K_ee, sigma, axes=((2, 3), (0, 1)))

    K_ += mdot(f2[:, :, n_s:], V1, DW_ll)

    def E(vec):
        n = len(vec.shape)
        return np.tensordot(vec, sigma, axes=((n - 2, n - 1), (0, 1)))

    L = sdot(g_sx, X_tt) + mdot(g_xx, X_s, X_tt)

    L += E(g_see + mdot(g_xee, X_s, I_e, I_e))

    M = E(mdot(X_sss, V1_3, g_e, g_e) + 2 * mdot(X_ss, V_sl, g_e))
    M += mdot(X_ss, V1_3, E(g_ee) + sdot(g_x, X_tt))

    A = f_x + dot(f_snext + dot(f_xnext, X_s), g_x)  # same as before
    B = f_xnext  # same
    C = V1_3  # same
    D = K_ + dot(f_snext + dot(f_xnext, X_s), L) + dot(f_xnext, M)

    X_stt = solve_sylvester(A, B, C, D)

    if approx_order == 3:
        # if sigma is None:
        #     return [X_s,X_ss,X_sss]
        # else:
        #     return [[X_s,X_ss,X_sss],[X_tt, X_stt]]
        return [[X_s, X_ss, X_sss], [X_tt, X_stt]]


if __name__ == "__main__":
    from dolo import yaml_import

    model = yaml_import("examples/models/compat/rbc.yaml")
    # model = yaml_import('/home/pablo/Programming/papers/finint/models/integration_B_pert.yaml')

    import time

    t1 = time.time()
    dr = approximate_controls(model, order=2)
    print(dr.X_s)
    # print(dr.X_ss)
    # print(dr.X_sss)
    t2 = time.time()
    print("Elapsed {}".format(t2 - t1))

    t1 = time.time()
    dr = approximate_controls(model, order=2)
    print(dr.X_s)
    # print(dr.X_ss)
    # print(dr.X_sss)
    t2 = time.time()
    print("Elapsed {}".format(t2 - t1))
