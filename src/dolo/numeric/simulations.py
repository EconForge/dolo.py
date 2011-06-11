import numpy as np

def multisimul(g, start, variance, horizon, n_shocks, seed=1):
    """
    Computes multiple stochastic simulations.
    @param g:
    @param start:
    @param variance:
    @param horizon:
    @param n_shocks:
    @param seed:
    @return: [mean values, standard deviations]
    """

    n_v = len(start)
    S = variance

    epsilons = np.zeros( (n_v, horizon, n_exp) )
    for i in range(n_exp):
        epsilons[:,:,i] = numpy.random.multivariate_normal(M,S,horizon).T

    simul = np.zeros( (n_v,horizon + 1,n_exp) )
    simul[:,0,:] = np.tile(start,(n_exp,1)).T

    for t in range(horizon):
        simul[:,t+1,:] = g(simul[:,t,:], epsilons[:,t,:])

    sim = np.mean(simul,axis=2)
    sd = np.std(simul,axis=2)
    return [sim,sd]


#####

if __name__ == '__main__':

    print 'Running computations'
    
    A = np.array([[0.9,0],[0.0,0.9]])
    B = np.eye(2)
    M = np.zeros(2) # mean of innovations
    S = np.eye(2)*0.005 # covariance matrix
    def g(x,e):
        return np.dot(A,x) + np.dot(B,e)

    n_v = 2
    horizon = 50
    n_exp = 100
    seed = 1
    start = np.array([1,0])

    import numpy.random
    from scipy.stats.mstats import mquantiles

    import time


    # computations

    print "\nComputing simulations using quantiles."
    s = time.time()
    
    simul = np.zeros( (n_exp,n_v,horizon + 1) )
    for k_exp in range(n_exp):
        eps = numpy.random.multivariate_normal(M,S,horizon).T
        sim = np.zeros((n_v,horizon+1))
        sim[:,0] = start
        for i in range(horizon):
            sim[:,i+1] = g(sim[:,i],eps[:,i])
        simul[k_exp,...] = sim

    i_v = 0 # variable of interest

    # compute quantiles
    quants = [0.2,0.5,0.8]
    n_q = len(quants)
    out1 = np.zeros((n_q,horizon+1))
    for t in range(horizon+1):
        out1[:,t] = mquantiles(simul[:,i_v,t],quants)
    #print out

    ss = time.time()

    print 'Elapsed : ' + str(ss - s)

    # computations 2

    print "\nComputing simulations using standard deviations."

    import time
    s = time.time()
    [simul,sd] = multisimul(g, start, S, horizon, n_exp )


    ss = time.time()
    print 'Elapsed : ' + str(ss - s)

    from matplotlib import pylab
    x = np.linspace(0,horizon+1,num=horizon+1)
    #pylab.plot(x,simul[0,i_v,:])


    pylab.figure(1)
    pylab.plot(x,out1[0,:],'-.',color='black')
    pylab.plot(x,out1[1,:],color='black')
    pylab.plot(x,out1[2,:],'-.',color='black')
    

    pylab.figure(2)
    pylab.plot(x,simul[i_v,:],color='black')
    pylab.plot(x,simul[i_v,:]-sd[i_v,:],'-.',color='black')
    pylab.plot(x,simul[i_v,:]+sd[i_v,:],'-.',color='black')
    pylab.show()