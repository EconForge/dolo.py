import time
import numpy as np
from dolo.algos.dtcscc.perturbations import approximate_controls
from dolo.numeric.interpolation import create_interpolator


def parameterized_expectations_direct(model, verbose=False, initial_dr=None,
                                      pert_order=1, grid={}, distribution={},
                                      maxit=100, tol=1e-8):
    '''
    Finds a global solution for ``model`` using parameterized expectations
    function. Requires the model to be written with controls as a direct
    function of the model objects.

    The algorithm iterates on the expectations function in the arbitrage
    equation. It follows the discussion in section 9.9 of Miranda and
    Fackler (2002).

    Parameters
    ----------
    model : NumericModel
        "dtcscc" model to be solved
    verbose : boolean
        if True, display iterations
    initial_dr : decision rule
        initial guess for the decision rule
    pert_order : {1}
        if no initial guess is supplied, the perturbation solution at order
        ``pert_order`` is used as initial guess
    grid: grid options
    distribution: distribution options
    maxit: maximum number of iterations
    tol: tolerance criterium for successive approximations

    Returns
    -------
    decision rule :
        approximated solution
    '''

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

    approx = model.get_grid(**grid)
    grid = approx.grid
    interp_type = approx.interpolation
    dr = create_interpolator(approx, interp_type)
    expect = create_interpolator(approx, interp_type)

    distrib = model.get_distribution(**distribution)
    nodes, weights = distrib.discretize()

    N = grid.shape[0]
    z = np.zeros((N, len(model.symbols['expectations'])))

    x_0 = initial_dr(grid)
    x_0 = x_0.real  # just in case ...
    h_0 = h(grid, x_0, parms)

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

    while err > tol and it <= maxit:

        it += 1
        t_start = time.time()

        # dr.set_values(x_0)
        expect.set_values(h_0)

        z[...] = 0
        for i in range(weights.shape[0]):
            e = nodes[i, :]
            S = g(grid, x_0, e, parms)
            # evaluate expectation over the future state
            z += weights[i]*expect(S)

        # TODO: check that control is admissible
        new_x = d(grid, z, parms)
        new_h = h(grid, new_x, parms)

        # Update guess for decision rule and expectations function
        x_0 = new_x
        h_0 = new_h

        # update error and print if `verbose`
        err = (abs(new_h - h_0).max())
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

    # Interpolation for the decision rule
    dr.set_values(x_0)

    return dr
