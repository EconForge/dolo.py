from dolo.numeric.decision_rules_states import CDR
import numpy as np
from numpy import column_stack, row_stack, eye, zeros
from numpy import dot

def approximate_controls(model, return_dr=True):

    # get steady_state
    import numpy

    p = model.calibration['parameters']
    sigma = model.calibration['covariances']
    s = model.calibration['states'][:,None]
    x = model.calibration['controls'][:,None]
    e = model.calibration['shocks'][:,None]

    from numpy.linalg import solve

    g = model.functions['transition']
    f = model.functions['arbitrage']

    l = g(s,x,e,p, derivs=True)
    [junk, g_s, g_x, g_e] = [el[...,0] for el in l]

    if model.model_type == "fg2":
      l = f(s,x,e,s,x,p, derivs=True)
      [res, f_s, f_x, f_e, f_S, f_X] = [el[...,0] for el in l]
    else:
      l = f(s,x,s,x,p, derivs=True)
      [res, f_s, f_x, f_S, f_X] = [el[...,0] for el in l]

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


