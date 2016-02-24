from time import time

import numpy as np

import time
import numpy as np
from dolo.numeric.discretization import gauss_hermite_nodes
from dolo.numeric.interpolation.splines import MultivariateCubicSplines
from dolo.numeric.misc import mlinspace
from dolo.algos.dtcscc.perturbations import approximate_controls
# from dolo import approximate_controls, time_iteration, simulate

def solve(model, distributions,  maxit=100, tol=1e-8, initial_dr=None, verbose=False):

    g = model.functions['transition']
    d = model.functions['direct_response']
    h = model.functions['expectation']
    p = model.calibration['parameters']

    nodes, weights = gauss_hermite_nodes([3], model.covariances)

    ap = model.options['approximation_space']
    a = ap['a']
    b = ap['b']
    orders = ap['orders']
    grid = mlinspace(a,b,orders)
    N = grid.shape[0]

    n_x = len(model.symbols['controls'])
    n_s = len(model.symbols['states'])

    import pandas

    k = next(iter(distributions.keys()))
    v = distributions[k]
    v = np.asarray(v)[:,None]
    columns = [k]
    dist_df = pandas.DataFrame(v, columns=columns)

    # dri = [] # initial_guesses (perturbations)

    N_a = dist_df.shape[0] # number of agents
    P = p[None,:].repeat(N_a, axis=0)

    all_drs = [MultivariateCubicSplines(a,b,orders) for i in range(N_a)]

    xx_0 = np.zeros((N, N_a, n_x)) # initial guess

    for i in range(N_a):
        cc = { k :float(dist_df[k][i]) for k in dist_df.columns}
        model.set_calibration(**cc)
        P[i,:] = model.calibration['parameters']

        dr = approximate_controls(model, eigmax=1.0001)
        xx_0[:,i,:] = dr(grid)

    P = P[None,:,:].repeat(N,axis=0)  # P does not depend on the first dimension
    # gg = grid[:,None,:].repeat(N_a,axis=1)

    z = np.zeros((N,N_a,len(model.symbols['expectations'])))

    it = 0
    err = 10
    err_0 = 10

    if verbose:
        headline = '|{0:^4} | {1:10} | {2:8} | {3:8} | {4:8} %'
        headline = headline.format('N', ' Error', 'Gain', 'Time', 'Finished')
        stars = '-'*len(headline)
        print(stars)
        print(headline)
        print(stars)

        # format string for within loop
        fmt_str = '|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} | {4:8.3f}'

    X = xx_0.copy()
    gg = grid[:,None,:].repeat(N_a, axis=1)

    while err>tol and it<=maxit:

        it += 1

        t_start = time.time()

        # NEW
        for i in range(N_a):
            all_drs[i].set_values(xx_0[:,i,:])

        z[...] = 0
        for i in range(weights.shape[0]):
            e = nodes[i,:][None,None,:]
            S = g(gg, xx_0, e, P)
            arg = S.reshape((N,N_a,n_s))
            for k in range(N_a):
                X[:,k,:] = all_drs[i](arg[:,k,:])
            z += weights[i]*h(S,X,P)

        # TODO: check that control is admissible
        new_x = d(grid[:,None,:], z, P)

        # check whether they differ from the preceding guess
        err = (abs(new_x - xx_0).max(axis=(0,1,2)))
        finished = abs(new_x - xx_0).max(axis=(0,2))<=tol
        proportion = sum(finished)/len(finished)

        xx_0 = new_x

        # update error and print if `verbose`
        err_SA = err/err_0
        err_0 = err
        t_finish = time.time()
        elapsed = t_finish - t_start
        if verbose:
            print(fmt_str.format(it, err, err_SA, elapsed, proportion*100))



if __name__ == '__main__':

    from dolo import *
    from dolo.compiler.model_aggregate import ModelAggregation
    import yaml
    m = yaml_import("examples/models/bewley_dtcscc.yaml")
    txt = open("examples/models/bewley_aggregate.yaml").read()

    data = yaml.safe_load(txt)
    agg = ModelAggregation(data, [m])

    import time
    t1 = time.time()
    solve(m, agg.distributions, verbose=True)
    t2 = time.time()
    print("Total time elapsed : {}".format(t2-t1))
