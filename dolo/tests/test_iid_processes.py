
import unittest
import scipy
from scipy.integrate import quad
import numpy as np
from dolo.numeric.processes_iid import *

## Polynomial
def f(x):
    return x**2

    def f(x):
        return x**2

def test_UNormal():
    σ = 0.1
    μ = 0.2
    N = 10
    distNorm = UNormal(mu=μ, sigma=σ)
    disNorm_gh = distNorm.discretize()
    disNorm_ep = distNorm.discretize(N=N, method='equiprobable')
    expval_gh = np.array([f(disNorm_gh.inode(0,j))*disNorm_gh.iweight(0,j) for j in range(disNorm_gh.n_inodes(0))]).sum()
    expval_ep = np.array([f(disNorm_ep.inode(0,j))*disNorm_ep.iweight(0,j) for j in range(disNorm_ep.n_inodes(0))]).sum()
    expval_normal = quad(lambda x: f(x)/np.sqrt(2*np.pi*σ**2)*np.exp(-(x-μ)**2/(2*σ**2)), -np.inf,np.inf)[0]
    M=1000
    s_MC = np.random.normal(μ, σ, M)
    expval_MC = np.array([f(s_MC[j]) for j in range(0,M)]).sum() / M
    assert(abs(expval_gh-expval_normal)<0.1)
    assert(abs(expval_ep-expval_normal)<0.1)

def test_Uniform():
    a = -1
    b = 1
    distUni = Uniform(a, b)
    disUni = distUni.discretize(N=10)
    expval_ep = np.array([f(disUni.inode(0,j))*disUni.iweight(0,j) for j in range(disUni.n_inodes(0))]).sum()
    M=1000
    s_MC = np.random.uniform(a, b, M)
    expval_MC = np.array([f(s_MC[j]) for j in range(0,M)]).sum() / M
    assert(abs(expval_ep-expval_MC)<0.1)

def test_Lognormal():
    σ = 0.1
    μ = 0.3
    distLog = LogNormal(μ=μ, σ=σ)
    disLog = distLog.discretize(N=10)
    expval_ep = np.array([f(disLog.inode(0,j))*disLog.iweight(0,j) for j in range(disLog.n_inodes(0))]).sum()
    M=1000
    s_MC = np.random.lognormal(μ, σ, M)
    expval_MC = np.array([f(s_MC[j]) for j in range(0,M)]).sum() / M
    assert(abs(expval_ep-expval_MC)<0.1)


def test_beta():
    α = 2
    β = 5
    distbeta = Beta(α, β)
    disbeta = distbeta.discretize(N=10)
    expval_ep = np.array([f(disbeta.inode(0,j))*disbeta.iweight(0,j) for j in range(disbeta.n_inodes(0))]).sum()
    M=1000
    s_MC = np.random.beta(α, β, M)
    expval_MC = np.array([f(s_MC[j]) for j in range(0,M)]).sum() / M
    assert(abs(expval_ep-expval_MC)<0.1)
