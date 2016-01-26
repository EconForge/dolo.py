import numpy as np
from dolo.numeric.discretization import gauss_hermite_nodes
from dolo.numeric.interpolation.splines import MultivariateCubicSplines
from dolo.numeric.misc import mlinspace
from dolo.algos.dtcscc.perturbations import approximate_controls

import time

def pea(model, maxit=100, tol=1e-8, initial_dr=None, verbose=False):

    t1 = time.time()

    g = model.functions['transition']
    d = model.functions['direct_response']
    h = model.functions['expectation']

    p = model.calibration['parameters']

    if initial_dr is None:
        drp = approximate_controls(model)
    else:
        drp = approximate_controls(model)

    nodes, weights = gauss_hermite_nodes([5], model.covariances)

    ap = model.options['approximation_space']
    a = ap['a']
    b = ap['b']
    orders = ap['orders']
    grid = mlinspace(a,b,orders)

    dr = MultivariateCubicSplines(a,b,orders)

    N = grid.shape[0]
    z = np.zeros((N,len(model.symbols['expectations'])))

    x_0 = drp(grid)

    it = 0
    err = 10
    err_0 = 10

    if verbose:
        headline = '|{0:^4} | {1:10} | {2:8} | {3:8} |'
        headline = headline.format('N', ' Error', 'Gain', 'Time')
        stars = '-'*len(headline)
        print(stars)
        print(headline)
        print(stars)

        # format string for within loop
        fmt_str = '|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} |'

    while err>tol and it<=maxit:

        t_start = time.time()

        dr.set_values(x_0)

        z[...] = 0
        for i in range(weights.shape[0]):
            e = nodes[i,:]
            S = g(grid, x_0, e, p)
            # evaluate future controls
            X = dr(S)
            z += weights[i]*h(S,X,p)

        # TODO: check that control is admissible
        new_x = d(grid, z, p)

        # check whether they differ from the preceding guess
        err = (abs(new_x - x_0).max())

        x_0 = new_x

        if verbose:

            # update error and print if `verbose`
            err_SA = err/err_0
            err_0 = err
            t_finish = time.time()
            elapsed = t_finish - t_start
            if verbose:
                print(fmt_str.format(it, err, err_SA, elapsed))


    if it == maxit:
        import warnings
        warnings.warn(UserWarning("Maximum number of iterations reached"))

    # compute final fime and do final printout if `verbose`
    t2 = time.time()
    if verbose:
        print(stars)
        print('Elapsed: {} seconds.'.format(t2 - t1))
        print(stars)

    return dr


if __name__ == '__main__':

    from dolo import *

    model = yaml_import("examples/models/rbc_full.yaml")
    drp = approximate_controls(model)
    sol = pea(model, initial_dr=drp, verbose=True)
    sol = pea(model, initial_dr=drp, verbose=True)

# computes expectations # based on xinit
