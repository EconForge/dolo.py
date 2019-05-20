from dolo import *
model = yaml_import('examples\\models\\rbc_ar1.yaml')

model.exogenous

model.exogenous.Sigma
model.exogenous.rho

dp = model.exogenous.discretize()
dp
dp_gdp = model.exogenous.discretize_gdp()
dp_gdp

dp.no
dp.nodes()
dp_gdp.nodes

from matplotlib import pyplot as plt

plt.plot(dp_gdp.nodes.ravel(), dp_gdp.nodes.ravel()*0, 'o')
for i in range(dp_gdp.inodes.shape[0]):
    plt.plot(dp_gdp.inodes[i,:], 0.1+dp_gdp.inodes[i,:].ravel()*0, 'o', color='red')


########################################


import numpy as np
from typing import List # List means 1d array

class GDP:

    def __init__(self, nodes, inodes, iweights):
        self.nodes = nodes
        self.inodes = inodes
        self.iweights= iweights

    #def discretize_gdp(self):
    #    return self

    def grid(self):
        return EmptyGrid()

    def n_nodes(self)->int:
        return self.nodes.shape[0]

    def node(self, i: int): #->List:
        return self.nodes[i,:]

    def nodes(self):
        return self.nodes

    def n_inodes(self, i: int): #->int:
        return self.inodes.shape[1]

    def inode(self, i, j): #->List:
        return self.inodes[i,j]

    def iweight(self, i, j): #->float:
        return self.iweights[i,j]


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

    def discretize_gdp(self):

        Σ = self.Sigma
        ρ = self.rho

        n_nodes = 3
        n_std = 2.5
        n_integration_nodes = 5

        try:
            assert(Σ.shape[0]==1)
        except:
            raise Exception("Not implemented.")

        try:
            assert(ρ.shape[0]==ρ.shape[1]==1)
        except:
            raise Exception("Not implemented.")

        ρ = ρ[0,0]
        σ = np.sqrt(Σ[0,0])


        from dolo.numeric.discretization import gauss_hermite_nodes

        epsilons, weights = gauss_hermite_nodes([n_integration_nodes], Σ)

        min = -n_std*(σ/(np.sqrt(1-ρ**2)))
        max = n_std*(σ/(np.sqrt(1-ρ**2)))

        nodes = np.linspace(min,max,n_nodes)[:,None]
        iweights = weights[None,:].repeat(n_nodes,axis=0)
        #integration_nodes = np.zeros((n_nodes, n_integration_nodes))[:,:,None]
        integration_nodes = np.zeros((n_nodes, n_integration_nodes))
        for i in range(n_nodes):
            for j in range(n_integration_nodes):
                integration_nodes[i,j] =  ρ*nodes[i] + epsilons[j]

        return GDP(nodes,integration_nodes,iweights)
        #return (nodes,integration_nodes,iweights)



Σ = np.array([[0.001]])
ρ = 0.9
var = VAR1(rho=0.9, Sigma=Σ)
var.Sigma
var.rho



dp = var.discretize_gdp()
dp.inode(1,1)
dp.nodes
from dolo import *
model = yaml_import('examples\\models\\rbc_ar1.yaml')

model.exogenous

model.exogenous.Sigma
model.exogenous.rho
time_iteration(model, dprocess=dp)





var2 = VAR1(rho=model.exogenous.rho, Sigma=model.exogenous.Sigma)
var2

dprocess2 = var.discretize_gdp()
 dprocess2.n_nodes
 dprocess2.node



b = dprocess.n_inodes

from dolo import *

model = yaml_import('examples\\models\\rbc_ar1.yaml')
model.exogenous.Sigma
model.exogenous.rho



time_iteration(model, dprocess=dp)


# gdp = model.exogenous.discretize(method='gdp')
gdp = discretize(model.exogenous)



gdp = GDP(nodes, inodes, iweights)
nodes

gdp.node(0)


time_iteration(model, dprocess=gdp)

#%%
from matplotlib import pyplot as plt

plt.plot(nodes.ravel(), nodes.ravel()*0, 'o')
for i in range(inodes.shape[0]):
    plt.plot(inodes[i,:], 0.1+inodes[i,:].ravel()*0, 'o', color='red')
