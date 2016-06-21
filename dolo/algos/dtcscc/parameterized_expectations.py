import time
import numpy as np
from dolo.numeric.discretization import gauss_hermite_nodes
from dolo.numeric.interpolation.splines import MultivariateCubicSplines
from dolo.numeric.misc import mlinspace
from dolo.algos.dtcscc.perturbations import approximate_controls
from dolo.numeric.interpolation import create_interpolator

def parameterized_expectations_direct(model, verbose=False, initial_dr=None, pert_order=1, grid={}, distribution={}, maxit=100, tol=1e-8):

    t1 = time.time()

    g = model.functions['transition']
    d = model.functions['direct_response']
    h = model.functions['expectation']
    parms = model.calibration['parameters']

    if initial_dr is None:
        if pert_order == 1:
            initial_dr = approximate_controls(model)

        if pert_order > 1:
            raise Exception("Perturbation order > 1 not supported (yet).")

    approx      = model.get_grid(**grid)
    grid        = approx.grid
    interp_type = approx.interpolation
    dr          = create_interpolator(approx, approx.interpolation)
    expect      = create_interpolator(approx, approx.interpolation)

    distrib = model.get_distribution(**distribution)
    nodes, weights = distrib.discretize()

    N = grid.shape[0]
    z = np.zeros((N,len(model.symbols['expectations'])))

    xinit = initial_dr(grid)
    xinit = xinit.real  # just in case ...
    x_0 = xinit
    h_0 = h(grid,x_0,parms)

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

        it += 1
        t_start = time.time()

        # dr.set_values(x_0)
        expect.set_values(h_0)

        z[...] = 0
        for i in range(weights.shape[0]):
            e = nodes[i,:]
            S = g(grid, x_0, e, parms)
            # evaluate future controls
            z += weights[i]*expect(S)

        # TODO: check that control is admissible
        new_x = d(grid, z, parms)
        new_h = h(grid, new_x, parms)

        # check whether they differ from the preceding guess
        # err = (abs(new_x - x_0).max())
        err = (abs(new_h - h_0).max())

        x_0 = new_x
        h_0 = new_h

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

    dr.set_values(x_0)   # Interpolation for the decision rule 

    return dr
