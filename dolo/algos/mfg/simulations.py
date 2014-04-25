import numpy


from dolo import *
import pickle

# should be moved to markov
from numba import jit, njit

@njit
def choice(x, n, cumul):
    ind = 0
    for i in range(n):
        if x < cumul[i]:
            ind = i
            break
    return ind


@jit
def simulate_markov_chain(nodes, transitions, i_0, n_exp, horizon):

    n_states = nodes.shape[0]

    start = numpy.array( (i_0,)*n_exp )
    simul = numpy.zeros( (horizon, n_exp), dtype=int)
    rnd = numpy.random.rand(horizon* n_exp).reshape((horizon,n_exp))

    cumuls = transitions.cumsum(axis=1)

    for t in range(horizon-1):
        for i in range(n_exp):
            s = simul[t,i]
            p = cumuls[s,:]
            simul[t+1,i] = choice(  rnd[t,i], n_states, p)

    res = numpy.row_stack(simul)

    return res

def simulate(model, dr, i_0, s0=None, n_exp=100, horizon=50, use_pandas=True, markov_indices=None):

    if n_exp<1:
        is_irf = True
        use_pandas = True
        n_exp = 1

    nodes, transitions = model.markov_chain
    if s0 is None:
        s0 = model.calibration['states']

    if markov_indices is None:
        markov_indices = simulate_markov_chain(nodes, transitions, i_0, n_exp, horizon)

    # s = s0.copy()

    gg = model.functions['transition']

    with_aux = ('auxiliary' in model.functions)
    if with_aux:
        aux = model.functions['auxiliary']

    p = model.calibration['parameters']

    states = numpy.zeros( (horizon, n_exp, len(s0)) )
    controls = numpy.zeros( (horizon, n_exp, len(model.symbols['controls']) ) )

    if with_aux:
        auxiliaries = numpy.zeros( ( horizon, n_exp, len(model.symbols['auxiliaries']) ) )
        a = model.functions['auxiliary']
        g = lambda m,s,x,M,p: gg(m,s,x,a(m,s,x,p),M,p)

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
        for t in range(horizon):
            auxiliaries[t,:,:] = aux( markov_states[t,:,:], states[t,:,:], controls[t,:,:], p[None,:])
        l = [markov_states, states, controls, auxiliaries]
    else:
        l = [markov_states, states, controls]

    sims = numpy.concatenate(l,  axis=2)

    if not use_pandas or n_exp != 1:
        return sims
    else:
        import pandas
        if with_aux:
            columns = model.symbols['markov_states'] + model.symbols['states'] + model.symbols['controls'] + model.symbols['auxiliaries']
        else:
            columns = model.symbols['markov_states'] + model.symbols['states'] + model.symbols['controls']
            
        sims = pandas.DataFrame(sims[:,0,:], columns=columns)
        return sims