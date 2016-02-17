from time import time

import numpy as np

from dolo import approximate_controls, time_iteration


def grid_search_sim(m, agg, verbose=False):
    if agg.model_type != "aggregation":
        raise ValueError("agg must have model type `'aggregation'`")

    if m.model_type != "dtcscc":
        raise ValueError("Model must be of type `'dtcscc'`")

    # for now assume a single type of heterogeneity
    if len(agg.distributions) > 1:
        msg = "only one form of heterogeneity is currently supported"
        raise NotImplementedError(msg)

    # construct initial guess for decision rule
    initial_dr = approximate_controls(m, eigmax=1.0001)

    # extract which parameter has heterogeneity as well as the values it should
    # take on
    het_param = list(agg.distributions.keys())[0]
    het_vals = agg.distributions[het_param]
    drs = [None for i in range(len(het_vals))]

    # extract free parameters and their calibrated values from individual prob
    agg_vars = {k: m.calibration_dict[k] for k in agg.free_parameters}

    # extract initial value of the heterogenous parameter so we can clean up
    # when we are done
    init_het = m.calibration_dict[het_param]

    # loop over parameterizations and solve for decision rule, taking as given
    # the free parameters
    drs[0] = initial_dr
    start = time()
    for i in range(len(het_vals)):
        # update model calibration
        calib = {het_param: het_vals[i]}
        calib.update(agg_vars)
        m.set_calibration(**calib)

        # solve for global solution, with starting guess the previous
        # calibration's output (on first pass use the initial_dr from above)
        ix = i - 1 if i >= 1 else i
        drs[i] = time_iteration(m, initial_dr=drs[ix], interp_type='spline')

        if verbose:
            msg = "done with {0}\t total time {1:3.2f}"
            print(msg.format(i, time() - start))

    # Things TODO before done:
    # 1. Simulate the decision rules for each parameter value
    # 2. aggregate decision rules and check equilibrium conditions in
    #    `agg.function`
    # 3. Wrap this entire procedure in an outer loop that searches over
    #    agg_vars


if __name__ == '__main__':
    from dolo import *
    from dolo.compiler.model_aggregate import ModelAggregation
    import yaml
    m = yaml_import("../../../../examples/models/bewley_dtcscc.yaml")
    txt = open("../../../../examples/models/bewley_aggregate.yaml").read()
    data = yaml.safe_load(txt)

    agg = ModelAggregation(data, [m])
    i = 1
