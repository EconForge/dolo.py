from __future__ import division

def residuals(f, g, i_ms, s, x, dr, P, Q, parms):

    N = s.shape[0]
    n_s = s.shape[1]

    n_ms = P.shape[0]   # number of markov states
    n_mv = P.shape[1] # number of markov variable

    res = numpy.zeros_like(x)
    XM = numpy.zeros_like(x)


    import time

    # solving on grid for markov index i_ms
    # m = P[i_ms,:][None,:]
    m = numpy.tile(P[i_ms,:],(N,1))
    xm = x[:,:]

    for I_ms in range(n_ms):

        # M = P[I_ms,:][None,:]
        M = numpy.tile(P[I_ms,:], (N,1))
        prob = Q[i_ms, I_ms]

        S = g(m, s, xm, M, parms)
        XM[:,:] = dr(I_ms, S)
        rr = f(m,s,xm,M,S,XM,parms)
        res[:,:] += prob*rr

    return res
