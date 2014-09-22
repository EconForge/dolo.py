from dolo.numeric.taylor_expansion import TaylorExpansion as CDR

import numpy as np
from numpy import column_stack, row_stack, eye, zeros
from numpy import dot

class BlanchardKahnError(Exception):

    def __init__(self, n_found, n_expected):
        self.n_found = n_found
        self.n_expected = n_expected

    def __str__(self):

        msg =  "There are {} eigenvalues greater than one. There should be exactly {} to meet Blanchard-Kahn conditions.".format(self.n_found, self.n_expected)
        return msg

class GeneralizedEigenvaluesError(Exception):

    def __init__(self, diag_S, diag_T):
        self.diag_S = diag_S
        self.diag_T = diag_T

    def __str__(self):
        # TODO: explain better
        return "Eigenvalues are not uniquely defined. "

           

def approximate_controls(model, verbose=False, steady_state=None, eigmax=1.0, solve_steady_state=False, order=1):
    """Compute first order approximation of optimal controls

    Parameters:
    -----------

    model: NumericModel
        Model to be solved

    verbose: boolean
        If True: displays number of contracting eigenvalues

    steady_state: ndarray
        Use supplied steady-state value to compute the approximation. The routine doesn't check whether it is really
        a solution or not.

    solve_steady_state: boolean
        Use nonlinear solver to find the steady-state

    orders: {1}
        Approximation order. (Currently, only first order is supported).

    Returns:
    --------

    TaylorExpansion:
        Decision Rule for the optimal controls around the steady-state.

    """


    if order>1:
        raise Exception("Not implemented.")

    # get steady_state
    import numpy

    # if model.model_type == 'fga':
    #     model = GModel_fg_from_fga(model)

    # g = model.functions['transition']
    # f = model.functions['arbitrage']
    from dolo.algos.fg.convert import get_fg_functions
    [f,g] = get_fg_functions(model)

    if steady_state is not None:
        calib = steady_state
    else:
        calib = model.calibration

    if solve_steady_state:
        from dolo.algos.fg.steady_state import find_deterministic_equilibrium
        calib = find_deterministic_equilibrium(model)

    p = calib['parameters']
    s = calib['states']
    x = calib['controls']
    e = calib['shocks']

    if model.covariances is not None:
        sigma = model.covariances
    else:
        sigma = numpy.zeros((len(e), len(e)))

    from numpy.linalg import solve


    l = g(s,x,e,p, diff=True)
    [junk, g_s, g_x, g_e] = l[:4] # [el[0,...] for el in l[:4]]

    l = f(s,x,e,s,x,p, diff=True)
    [res, f_s, f_x, f_e, f_S, f_X] = l #[el[0,...] for el in l[:6]]

    n_s = g_s.shape[0]           # number of controls
    n_x = g_x.shape[1]   # number of states
    n_e = g_e.shape[1]
    n_v = n_s + n_x

    A = row_stack([
        column_stack( [ eye(n_s), zeros((n_s,n_x)) ] ),
        column_stack( [ -f_S    , -f_X             ] )
    ])

    B = row_stack([
        column_stack( [ g_s, g_x ] ),
        column_stack( [ f_s, f_x ] )
    ])

    from dolo.numeric.extern.qz import qzordered
    [S,T,Q,Z,eigval] = qzordered(A,B,n_s)
    Q = Q.real # is it really necessary ?
    Z = Z.real

    diag_S = numpy.diag(S)
    diag_T = numpy.diag(T)

    # Check Blanchard=Kahn conditions
    n_big_one = sum(eigval>eigmax)
    n_expected = n_x
    if verbose:
        print( "There are {} eigenvalues greater than {}. Expected: {}.".format( n_big_one, eigmax, n_x ) )
    if n_expected != n_big_one:
        raise BlanchardKahnError(n_big_one, n_expected)


    tol_geneigvals = 1e-10
    try:
        assert( sum(  (abs( diag_S ) < tol_geneigvals) * (abs(diag_T) < tol_geneigvals) ) == 0)
    except Exception as e:
        print e
        print(numpy.column_stack([diag_S, diag_T]))
        # raise GeneralizedEigenvaluesError(diag_S, diag_T)


    Z11 = Z[:n_s,:n_s]
    Z12 = Z[:n_s,n_s:]
    Z21 = Z[n_s:,:n_s]
    Z22 = Z[n_s:,n_s:]
    S11 = S[:n_s,:n_s]
    T11 = T[:n_s,:n_s]

    # first order solution
    C = solve(Z11.T, Z21.T).T
    P = np.dot(solve(S11.T, Z11.T).T , solve(Z11.T, T11.T).T )
    Q = g_e

    s = s.ravel()
    x = x.ravel()

    A = g_s + dot( g_x, C )
    B = g_e

    dr = CDR([s, x, C])
    dr.A = A
    dr.B = B
    dr.sigma = sigma
   
    return dr

if __name__ == '__main__':

    from dolo import *

    model = yaml_import("/home/pablo/Documents/Research/Thesis/chapter_1/code/portfolios.yaml")
    model.set_calibration(dumb=1)

    from dolo.algos.perturbations import approximate_controls

    dr = approximate_controls(model)
    print(dr)


