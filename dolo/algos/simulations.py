import numpy
import pandas
import xarray as xr
import numpy as np
from dolo.numeric.optimize.ncpsolve import ncpsolve
from dolo.numeric.optimize.newton import newton as newton_solver
from dolo.numeric.optimize.newton import SerialDifferentiableFunction

## TODO: extend for mc process


def response(model, dr, varname, T=40, impulse: float = None):

    i_exo = model.symbols["exogenous"].index(varname)

    if impulse is None:
        try:
            impulse = numpy.sqrt(
                model.exogenous.Σ[i_exo, i_exo]
            )  # works only for IID/AR1
        except:
            impulse = numpy.sqrt(model.exogenous.σ)  # works only for IID/AR1

    e1 = numpy.zeros(len(model.symbols["exogenous"]))
    e1[i_exo] = impulse

    exogenous = model.exogenous
    print(exogenous)
    print(T, e1)
    m_simul = model.exogenous.response(T - 1, e1)  # this is an xarray T x V
    m_simul = m_simul.expand_dims("N")
    m_simul = m_simul.transpose("T", "N", "V").data

    sim = simulate(model, dr, N=1, T=T, driving_process=m_simul, stochastic=False)

    irf = sim.sel(N=0)

    return irf


def find_index(sim, values):
    sh = sim.shape
    N = sh[0]
    T = sh[1]
    indices = np.zeros((N, T), dtype=int)
    for n in range(N):
        for t in range(T):
            v = sim[n, t, :]
            ind = np.where((values == v[None, :]).all(axis=1))[0][0]
            indices[n, t] = ind
    return indices


from dolo.numeric.grids import CartesianGrid, UnstructuredGrid
from dolo.algos.results import AlgoResult


def simulate(
    model,
    dr,
    process=None,
    N=1,
    T=40,
    s0=None,
    i0=None,
    m0=None,
    driving_process=None,
    seed=42,
    stochastic=True,
):
    """
    Simulate a model using the specified decision rule.

    Parameters
    ----------

    model: Model

    dr: decision rule

    process:

    s0: ndarray
        initial state where all simulations start

    driving_process: ndarray
        realization of exogenous driving process (drawn randomly if None)

    N: int
        number of simulations
    T: int
        horizon for the simulations
    seed: int
        used to initialize the random number generator. Use it to replicate
        exact same results among simulations
    discard: boolean (False)
        if True, then all simulations containing at least one non finite value
        are discarded

    Returns
    -------
    xarray.DataArray:
        returns a ``T x N x n_v`` array where ``n_v``
           is the number of variables.
    """

    if isinstance(dr, AlgoResult):
        dr = dr.dr

    calib = model.calibration
    parms = numpy.array(calib["parameters"])

    if s0 is None:
        s0 = calib["states"]

    n_x = len(model.symbols["controls"])
    n_s = len(model.symbols["states"])

    s_simul = numpy.zeros((T, N, n_s))
    x_simul = numpy.zeros((T, N, n_x))

    s_simul[0, :, :] = s0[None, :]

    # are we simulating a markov chain or a continuous process ?
    if driving_process is not None:
        if len(driving_process.shape) == 3:
            m_simul = driving_process
            sim_type = "continuous"
            if m0 is None:
                m0 = model.calibration["exogenous"]
            x_simul[0, :, :] = dr.eval_ms(m0[None, :], s0[None, :])[0, :]
        elif len(driving_process.shape) == 2:
            i_simul = driving_process
            nodes = dr.exo_grid.nodes
            m_simul = nodes[i_simul]
            # inds = i_simul.ravel()
            # m_simul = np.reshape( np.concatenate( [nodes[i,:][None,:] for i in inds.ravel()], axis=0 ), inds.shape + (-1,) )
            sim_type = "discrete"
            x_simul[0, :, :] = dr.eval_is(i0, s0[None, :])[0, :]
        else:
            raise Exception("Incorrect specification of driving values.")
        m0 = m_simul[0, :, :]
    else:
        from dolo.numeric.processes import DiscreteProcess

        if process is None:
            if hasattr(dr, "dprocess") and hasattr(dr.dprocess, "simulate"):
                process = dr.dprocess
            else:
                process = model.exogenous

        # detect type of simulation
        if not isinstance(process, DiscreteProcess):
            sim_type = "continuous"
        else:
            sim_type = "discrete"

        if sim_type == "discrete":
            if i0 is None:
                i0 = 0
            dp = process
            m_simul = dp.simulate(N, T, i0=i0, stochastic=stochastic)
            i_simul = find_index(m_simul, dp.values)
            m0 = dp.node(i0)
            x0 = dr.eval_is(i0, s0[None, :])[0, :]
        else:
            m_simul = process.simulate(N, T, m0=m0, stochastic=stochastic)
            if isinstance(m_simul, xr.DataArray):
                m_simul = m_simul.data
            sim_type = "continuous"
            if m0 is None:
                m0 = model.calibration["exogenous"]
            x0 = dr.eval_ms(m0[None, :], s0[None, :])[0, :]
            x_simul[0, :, :] = x0[None, :]

    f = model.functions["arbitrage"]
    g = model.functions["transition"]

    numpy.random.seed(seed)

    mp = m0
    for i in range(T):
        m = m_simul[i, :, :]
        s = s_simul[i, :, :]
        if sim_type == "discrete":
            i_m = i_simul[i, :]
            xx = [
                dr.eval_is(i_m[ii], s[ii, :][None, :])[0, :] for ii in range(s.shape[0])
            ]
            x = np.row_stack(xx)
        else:
            x = dr.eval_ms(m, s)

        x_simul[i, :, :] = x

        ss = g(mp, s, x, m, parms)
        if i < T - 1:
            s_simul[i + 1, :, :] = ss
        mp = m

    if "auxiliary" not in model.functions:  # TODO: find a better test than this
        l = [s_simul, x_simul]
        varnames = model.symbols["states"] + model.symbols["controls"]
    else:
        aux = model.functions["auxiliary"]
        a_simul = aux(
            m_simul.reshape((N * T, -1)),
            s_simul.reshape((N * T, -1)),
            x_simul.reshape((N * T, -1)),
            parms,
        )
        a_simul = a_simul.reshape(T, N, -1)
        l = [m_simul, s_simul, x_simul, a_simul]
        varnames = (
            model.symbols["exogenous"]
            + model.symbols["states"]
            + model.symbols["controls"]
            + model.symbols["auxiliaries"]
        )

    simul = numpy.concatenate(l, axis=2)

    if sim_type == "discrete":
        varnames = ["_i_m"] + varnames
        simul = np.concatenate([i_simul[:, :, None], simul], axis=2)

    data = xr.DataArray(
        simul,
        dims=["T", "N", "V"],
        coords={"T": range(T), "N": range(N), "V": varnames},
    )

    return data


def tabulate(
    model, dr, state, bounds=None, n_steps=100, s0=None, i0=None, m0=None, **kwargs
):

    import numpy

    if isinstance(dr, AlgoResult):
        dr = dr.dr

    states_names = model.symbols["states"]
    controls_names = model.symbols["controls"]
    index = states_names.index(str(state))

    if bounds is None:
        try:
            endo_grid = dr.endo_grid
            bounds = [endo_grid.min[index], endo_grid.max[index]]
        except:
            domain = model.domain
            bounds = [domain.min[index], domain.max[index]]
        if bounds is None:
            raise Exception("No bounds provided for simulation or by model.")

    values = numpy.linspace(bounds[0], bounds[1], n_steps)

    if s0 is None:
        s0 = model.calibration["states"]

    svec = numpy.row_stack([s0] * n_steps)
    svec[:, index] = values

    try:
        dp = dr.dprocess
    except:
        dp = model.exogenous.discretize()

    if (i0 is None) and (m0 is None):
        from dolo.numeric.grids import UnstructuredGrid

        if isinstance(dp.grid, UnstructuredGrid):
            n_ms = dp.n_nodes
            [q, r] = divmod(n_ms, 2)
            i0 = q - 1 + r
        else:
            m0 = model.calibration["exogenous"]

    if i0 is not None:
        m = dp.node(i0)
        xvec = dr.eval_is(i0, svec)
    elif m0 is not None:
        m = m0
        xvec = dr.eval_ms(m0, svec)

    mm = numpy.row_stack([m] * n_steps)
    l = [mm, svec, xvec]

    series = (
        model.symbols["exogenous"] + model.symbols["states"] + model.symbols["controls"]
    )

    if "auxiliary" in model.functions:
        p = model.calibration["parameters"]
        pp = numpy.row_stack([p] * n_steps)
        avec = model.functions["auxiliary"](mm, svec, xvec, pp)
        l.append(avec)
        series.extend(model.symbols["auxiliaries"])

    import pandas

    tb = numpy.concatenate(l, axis=1)
    df = pandas.DataFrame(tb, columns=series)

    return df


def tabulate_2d(model, dr, states=None, i0=0, s0=None, n=[12, 13]):

    import numpy
    import xarray as xr

    if isinstance(dr, AlgoResult):
        dr = dr.dr

    if s0 is None:
        s0 = model.calibration["states"]
    if states is None:
        states = model.symbols["states"]
    assert len(states) == 2
    domain = model.get_domain()
    lps = [numpy.linspace(*domain[s], n[i]) for i, s in enumerate(states)]
    i_x = model.symbols["states"].index(states[0])
    i_y = model.symbols["states"].index(states[1])
    vals = []
    vstates = []
    s = s0.copy()
    for xx in lps[0]:
        vv = []
        s[i_x] = xx
        for yy in lps[1]:
            s[i_y] = yy
            x = dr.eval_is(i0, s)
            vv.append(numpy.concatenate([s, x]))
        vals.append(vv)
    vv = numpy.array(vals)
    controls = model.symbols["states"] + model.symbols["controls"]
    #     tab = xr.DataArray(vv, dims=[states[0], states[1], 'V'], coords=[lps[0], lps[1], 'V'])
    tab = xr.DataArray(
        vv,
        dims=[states[0], states[1], "V"],
        coords={states[0]: lps[0], states[1]: lps[1], "V": controls},
    )
    return tab


def plot3d(tab, varname):
    X = numpy.array(tab[tab.dims[0]])
    Y = numpy.array(tab[tab.dims[1]])
    Z = numpy.array(tab.loc[:, :, varname])
    data = [go.Surface(x=X, y=Y, z=Z)]
    layout = go.Layout(
        title="Equity",
        autosize=False,
        width=500,
        height=500,
        #         xaxis=go.XAxis(title=tab.dims[0]),
        #         yaxis={'title':tab.dims[1]},
        #         zaxis={'title':varname},
        xaxis=dict(
            title="x Axis",
            nticks=7,
            titlefont=dict(family="Courier New, monospace", size=18, color="#7f7f7f"),
        ),
        margin=dict(l=65, r=50, b=65, t=90),
    )
    fig = go.Figure(data=data, layout=layout)
    return iplot(fig, filename="graph_" + varname)


def plot_decision_rule(plot_controls=None, **kwargs):

    if isinstance(dr, AlgoResult):
        dr = dr.dr

    df = tabulate(dr, state, bounds=None, n_steps=100, s0=None, i0=None, m0=None)

    from matplotlib import pyplot

    if isinstance(plot_controls, str):
        cn = plot_controls
        pyplot.plot(values, df[cn], **kwargs)
    else:
        for cn in plot_controls:
            pyplot.plot(values, df[cn], label=cn, **kwargs)
        pyplot.legend()
    pyplot.xlabel("state = {} | mstate = {}".format(state, i0))
