import numpy


# from dolo import *
import pickle

# should be moved to markov
from numba import jit, njit

@njit
def choice(x, n, cumul):
    i = 0
    running = True
    while i<n and running:
        if x < cumul[i]:
            running = False
        else:
            i += 1
    return i


@jit
def simulate_markov_chain(nodes, transitions, i_0, n_exp, horizon):

    n_states = nodes.shape[0]

#    start = numpy.array( (i_0,)*n_exp )
    simul = numpy.zeros( (horizon, n_exp), dtype=int)
    simul[0,:] = i_0
    rnd = numpy.random.rand(horizon* n_exp).reshape((horizon,n_exp))

    cumuls = transitions.cumsum(axis=1)
    cumuls = numpy.ascontiguousarray(cumuls)

    for t in range(horizon-1):
        for i in range(n_exp):
            s = simul[t,i]
            p = cumuls[s,:]
            simul[t+1,i] = choice(rnd[t,i], n_states, p)

    res = numpy.row_stack(simul)

    return res

def simulate(model, dr, i_0, s0=None, drv=None, n_exp=100, horizon=50, markov_indices=None, return_array=False):

    if n_exp<1:
        is_irf = True
        n_exp = 1

    nodes, transitions = model.markov_chain
    if s0 is None:
        s0 = model.calibration['states']
    else:
        s0 = numpy.array( numpy.atleast_1d(s0), dtype=float )

    if markov_indices is None:
        markov_indices = simulate_markov_chain(nodes, transitions, i_0, n_exp, horizon)
    else:
        if markov_indices.ndim == 1:
            markov_indices = numpy.atleast_2d(markov_indices).T
        markov_indices = numpy.ascontiguousarray(markov_indices,dtype=int)
        try:
            expected_shape = (horizon,n_exp)
            found_shape = markov_indices.shape
            assert(expected_shape==found_shape)
        except:
            raise Exception("Incorrect shape for markov indices. Expected {}. Found {}.".format(
                                        markov_indices.shape,(expected_shape,found_shape)
            ))

    # s = s0.copy()

    g = model.functions['transition']

    with_aux = ('auxiliary' in model.functions)
    if with_aux:
        aux = model.functions['auxiliary']

    p = model.calibration['parameters']

    states = numpy.zeros( (horizon, n_exp, len(s0)) )
    controls = numpy.zeros( (horizon, n_exp, len(model.symbols['controls']) ) )

    if with_aux:
        auxiliaries = numpy.zeros( ( horizon, n_exp, len(model.symbols['auxiliaries']) ) )
        a = model.functions['auxiliary']

    states[0,:,:] = s0[None,:]

    m = nodes[i_0,:]

    for t in range(horizon):
        for n in range(n_exp):
            i_m = markov_indices[t,n]
            M = nodes[i_m,:]
            s = states[t:t+1,n,:]
            x = dr(i_m, s)
            controls[t:t+1,n,:] = x
            S = g(m[None,:],s,x,M[None,:],p[None,:])
            if t < horizon-1:
                states[t+1:t+2,n,:] = S

    markov_states = nodes[markov_indices,:]

    if with_aux:
        pp = numpy.repeat(p[None,:], n_exp, axis=0)
        for t in range(horizon):
            auxiliaries[t,:,:] = aux( markov_states[t,:,:], states[t,:,:], controls[t,:,:], pp)
        l = [markov_states, states, controls, auxiliaries]
    else:
        l = [markov_states, states, controls]


    l = [markov_indices[:,:,None]] + l


    if with_aux:
        columns = model.symbols['markov_states'] + model.symbols['states'] + model.symbols['controls'] + model.symbols['auxiliaries']
    else:
        columns = model.symbols['markov_states'] + model.symbols['states'] + model.symbols['controls']

    if drv is not None:
        n_vals = len(model.symbols['values'])
        vals = numpy.zeros((horizon, n_exp, n_vals))
        for t in range(horizon):
            vals[t,:,:] = drv(markov_indices[t,:], states[t,:,:] )
        l.append(vals)
        columns.extend(model.symbols['values'])

    import pandas

    sims = numpy.concatenate(l,  axis=2)

    if return_array:
        return sims

    if n_exp > 1:
        sims = pandas.Panel(sims, minor_axis=['m_ind']+columns)
        sims = sims.swapaxes(0,1)

    else:
        sims = pandas.DataFrame(sims[:,0,:], columns=['m_ind']+columns)


    return sims


def plot_decision_rule(model, dr, state, plot_controls=None, bounds=None, n_steps=100, s0=None, i0=None, **kwargs):

    import numpy

    states_names = model.symbols['states']
    controls_names = model.symbols['controls']
    index = states_names.index(str(state))

    if bounds is None:
        bounds = [dr.smin[index], dr.smax[index]]

    values = numpy.linspace(bounds[0], bounds[1], n_steps)

    if s0 is None:
        s0 = model.calibration['states']

    if i0 == None:
        P,Q = model.markov_chain
        n_ms = P.shape[0]
        [q,r] = divmod(n_ms,2)
        i0 = q-1+r

    svec = numpy.row_stack([s0]*n_steps)
    svec[:,index] = values

    xvec = dr(i0,svec)

    m = model.markov_chain[0][i0]
    mm = numpy.row_stack([m]*n_steps)
    l = [mm, svec, xvec]

    series = model.symbols['markov_states'] + model.symbols['states'] + model.symbols['controls']

    if 'auxiliary' in model.functions:
        p = model.calibration['parameters']
        pp = numpy.row_stack([p]*n_steps)
        avec = model.functions['auxiliary'](mm, svec,xvec,pp)
        l.append(avec)
        series.extend(model.symbols['auxiliaries'])

    import pandas
    tb = numpy.concatenate(l, axis=1)
    df = pandas.DataFrame(tb, columns=series)

    if plot_controls is None:
        return df
    else:
        from matplotlib import pyplot
        if isinstance(plot_controls, str):
            cn = plot_controls
            pyplot.plot(values, df[cn], **kwargs)
        else:
            for cn in  plot_controls:
                pyplot.plot(values, df[cn], label=cn, **kwargs)
            pyplot.legend()
        pyplot.xlabel('state = {} | mstate = {}'.format(state, i0))
