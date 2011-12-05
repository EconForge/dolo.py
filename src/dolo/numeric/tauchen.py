import scipy as sp
import scipy.stats
import numpy as np
import math

# univariate tauchen code is taken from dpsolve ( https://bitbucket.org/stevejb/dpsolve )

def tauchen(N, mu, rho, sigma, m):
    """
    SJB: This is a port of Martin Floden's 1996 Matlab code to implement Tauchen 1986 Economic Letters method The following comments are Floden's. Finds a Markov chain whose sample paths approximate those of the AR(1) process z(t+1) = (1-rho)*mu + rho * z(t) + eps(t+1) where eps are normal with stddev sigma.

    :param N: scalar, number of nodes for Z
    :param mu: scalar, unconditional mean of process
    :param rho: scalar
    :param sigma: scalar, std. dev. of epsilons
    :param m: max +- std. devs.
    :returns: Z, N*1 vector, nodes for Z. Zprob, N*N matrix, transition probabilities

    Original implementation by Martin Floden Fall 1996. This procedure is an implementation of George Tauchen's algorithm described in Ec. Letters 20 (1986) 177-181.
    """
    Z     = np.zeros((N,1))
    Zprob = np.zeros((N,N))
    a     = (1-rho)*mu

    Z[-1]  = m * math.sqrt(sigma**2 / (1 - (rho**2)))
    Z[0]  = -1 * Z[-1]
    zstep = (Z[-1] - Z[0]) / (N - 1)

    for i in range(1,N):
        Z[i] = Z[0] + zstep * (i)

    Z = Z + a / (1-rho)

    for j in range(0,N):
        for k in range(0,N):
            if k == 0:
                Zprob[j,k] = sp.stats.norm.cdf((Z[0] - a - rho * Z[j] + zstep / 2) / sigma)
            elif k == (N-1):
                Zprob[j,k] = 1 - sp.stats.norm.cdf((Z[-1] - a - rho * Z[j] - zstep / 2) / sigma)
            else:
                up = sp.stats.norm.cdf((Z[k] - a - rho * Z[j] + zstep / 2) / sigma)
                down = sp.stats.norm.cdf( (Z[k] - a - rho * Z[j] - zstep / 2) / sigma)
                Zprob[j,k] = up - down

    return( (Z, Zprob) )


if __name__ == '__main__':
    [Z,Zprob] = tauchen( 5,0,0.8,0.1,1.5 )
    print Z
    print Zprob
