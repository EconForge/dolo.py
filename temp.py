import numpy as np

class VAR1:

    def __init__(self, rho=None, Sigma=None, mu=None, N=3):

        self.Sigma = np.atleast_2d(Sigma)
        d = self.Sigma.shape[0]
        rho = np.array(rho)
        if rho.ndim == 0:
            self.rho = np.eye(d)*rho
        elif rho.ndim ==1:
            self.rho = np.diag(rho)
        else:
            self.rho = rho
        if mu is None:
            self.mu = np.zeros(d)
        else:
            self.mu = np.array(mu, dtype=float)
        self.d = d

Σ = np.array([[0.001]])
ρ = 0.9

var = VAR1(rho=1)
#n_nodes = 3
#n_std = 2.5
#n_integration_nodes = 5


discretize(var)


def discretize(self):

    n_nodes = 3
    n_std = 2.5
    n_integration_nodes = 5

    #Σ = self.Sigma
    #ρ = self.rho

    Σ = np.array([[0.001]])
    ρ = 0.9

    try:
        assert(Σ.shape[0]==1)
    except:
        raise Exception("Not implemented.")
    σ = Σ[0,0]

    from dolo.numeric.discretization import gauss_hermite_nodes

    epsilons, weights = gauss_hermite_nodes([n_integration_nodes], Σ)

    min = -n_std*(σ/(np.sqrt(1-ρ**2)))
    max = n_std*(σ/(np.sqrt(1-ρ**2)))

    nodes = np.linspace(min,max,n_nodes)[:,None]
    iweights = weights[None,:].repeat(n_nodes,axis=0)
    for i in range(n_nodes):
        for j in range(n_integration_nodes):
            integration_nodes[i,j] =  ρ*nodes[i] + epsilons[j]

    return(nodes,inodes,iweights)






nodes = np.ndarray(shape=(1,N), dtype=float, order='F')
inodes = np.ndarray(shape=(N,M), dtype=float, order='F')
iweights =
var1 = VAR1(rho=ρ, Sigma=Σ)

var1.rho

def discretize(var, N=3, M=3):
    Σ = np.array([[0.001]])
    ρ = 0.9
    nodes: np.zeros(2.0, 3.0, num=5)
    inodes:
    iweights:

var1.Sigma
var1.rho
