from time import time

import numpy as np

from dolo import approximate_controls, time_iteration, simulate


def grid_search_sim(m, agg, verbose=True, n_exp=1000, horizon=300, tol=1e-6):
    if agg.model_type != "aggregation":
        raise ValueError("agg must have model type `'aggregation'`")

    # we only supprt dtcscc models for now
    if m.model_type != "dtcscc":
        raise ValueError("Model must be of type `'dtcscc'`")

    # for now assume a single type of heterogeneity
    if len(agg.distributions) > 1:
        msg = "only one form of heterogeneity is currently supported"
        raise NotImplementedError(msg)

    # extract the number of states and controls
    n_s = len(m.symbols["states"])
    n_x = len(m.symbols["controls"])

    # construct initial guess for decision rule
    initial_dr = approximate_controls(m, eigmax=1.0001)

    # extract which parameter has heterogeneity as well as the values it should
    # take on
    het_param = list(agg.distributions.keys())[0]
    het_vals = agg.distributions[het_param]
    n_het_vals = len(het_vals)

    # create list to hold decision rules and empty arrays to hold simulations
    drs = [None for i in range(n_het_vals)]
    sim_s = np.empty((n_het_vals, n_exp, n_s))
    sim_x = np.empty((n_het_vals, n_exp, n_x))

    # extract free parameters and their calibrated values from individual prob
    agg_vars = {k: m.calibration_dict[k] for k in agg.free_parameters}

    # extract initial value of the heterogenous parameter so we can clean up
    # when we are done
    init_het = m.calibration_dict[het_param]

    # loop over parameterizations and solve for decision rule, taking as given
    # the free parameters
    drs[0] = initial_dr
    start_time = time()
    err = np.array([10.0])
    it = 0

    while abs(err[0]) > tol:
        for i in range(n_het_vals):
            # update model calibration
            calib = {het_param: het_vals[i]}
            calib.update(agg_vars)
            m.set_calibration(**calib)

            # solve for global solution, with starting guess the previous
            # calibration's output (on first iteration use initial_dr)
            ix = i - 1 if i >= 1 else i
            drs[i] = time_iteration(m, initial_dr=drs[ix],
                                    interp_type='spline')

            # simluate using this calibration of the model and the decision
            # rule we just computed. We set a new seed each time so that we
            # get new shocks
            sim = simulate(m, drs[i], n_exp=n_exp, horizon=horizon,
                           seed=(i+1)*(it+1), return_array=True)

            # Here we make the assumption that at period `horizon` the
            # cross-sectional distribution of agents of type `i` has converged.
            # So, we keep all observations from the last period of the
            # simulation as a draw from the stationary distribution of agents
            # of type i
            sim_s[i, :, :] = sim[-1, :, :n_s]
            sim_x[i, :, :] = sim[-1, :, n_s:n_s+n_x]

            if verbose:
                msg = "\tdone with {0}  total time {1:3.2f}"
                print(msg.format(i, time() - start_time))

        # Given a draw from the cross section of each type, we are ready to
        # evaluate the equilibrium condition
        agg.function(np.swapaxes(sim_s, 2, 0),
                     np.swapaxes(sim_x, 2, 0),
                     m.calibration['parameters'],
                     err)

        if verbose:
            print("err is {0} on iteration {1}".format(err, it))

        # TODO: figure out how to update `agg_vars`
        it += 1


if __name__ == '__main__':
    from dolo import *
    from dolo.compiler.model_aggregate import ModelAggregation
    import yaml
    m = yaml_import("../../../../examples/models/bewley_dtcscc.yaml")
    txt = open("../../../../examples/models/bewley_aggregate.yaml").read()
    data = yaml.safe_load(txt)

    agg = ModelAggregation(data, [m])
    i = 1
