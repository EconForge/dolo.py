from matplotlib import pyplot as plt
import scipy
from scipy.integrate import quad
import scipy.special as special
import numpy as np
from dolo.numeric.processes_iid import *

σ = 0.1
μ = 0.2
N = 10

x=2

assert(x==2)

## Polynomial
def f(x):
    return x**2

## Discretize Normal distribution with GH and EP methods

distNorm = UNormal(mu=μ, sigma=σ)
disNorm_gh = distNorm.discretize()
disNorm_ep = distNorm.discretize(N=N, method='equiprobable')

disNorm_gh.n_inodes(0) # number of nodes
# disNorm_gh.inode(0,4) # 4th node
# disNorm_gh.iweight(0,4) # weight associated to 4th node

########################################################################
############################# Plots ####################################
########################################################################
## Plot Equiprobable
weights_ep, nodes_ep = np.array( [*zip(*[*disNorm_ep.iteritems(0)])] )
plt.plot(nodes_ep, nodes_ep*0, '.')
xl = plt.xlim()
xvec = np.linspace(xl[0], xl[1], 100)
pdf = scipy.stats.norm.pdf(xvec, loc=μ, scale=σ)
plt.plot(xvec, pdf)
plt.grid()

## Plot Gauss Hermite nodes
weights_gh, nodes_gh = np.array( [*zip(*[*disNorm_gh.iteritems(0)])] )
plt.plot(nodes_gh, nodes_gh*0, '.')
xl = plt.xlim()
xvec = np.linspace(xl[0], xl[1], 100)
pdf = scipy.stats.norm.pdf(xvec, loc=μ, scale=σ)
plt.plot(xvec, pdf)
plt.grid()

## Plot Random draws
M=1000
s_MC = np.random.normal(μ, σ, M)

count, bins, ignored = plt.hist(s_MC, 30, density=True)
plt.plot(bins, 1/(σ * np.sqrt(2 * np.pi)) * np.exp( - (bins - μ)**2 / (2 * σ **2) ), linewidth=2, color='r')
plt.show()


########################################################################

## Compute expectation of the normally distributed variable

expval_normal = quad(lambda x: f(x)/np.sqrt(2*np.pi*σ**2)*np.exp(-(x-μ)**2/(2*σ**2)), -np.inf,np.inf)[0]

## Compute the mean of random draws
expval_MC = np.array([f(s_MC[j]) for j in range(0,M)]).sum() / M

## Compute ∑(f(ϵ)*w_ϵ) for each discretization method

expval_gh = np.array([f(disNorm_gh.inode(0,j))*disNorm_gh.iweight(0,j) for j in range(disNorm_gh.n_inodes(0))]).sum()
expval_ep = np.array([f(disNorm_ep.inode(0,j))*disNorm_ep.iweight(0,j) for j in range(disNorm_ep.n_inodes(0))]).sum()

## Compare

expval_normal
expval_gh
expval_ep
expval_MC

assert(abs(expval_gh-expval_normal)<0.1)
assert(abs(expval_ep-expval_normal)<0.1)


##################################################################
############################# UNIFORM ###########################
##################################################################

a = -1
b = 1 #### CANNOT PUT 0
distUni = Uniform(a, b)
disUni = distUni.discretize(N=10)
### Here there is an issue with the uniform

## Random draws
M=1000
s_MC = np.random.uniform(a, b, M)

count, bins, ignored = plt.hist(s_MC, 10, density=True)
plt.plot(bins, np.ones_like(bins)*(1/(b-a)), linewidth=2, color='r')
plt.show()

## Plot Equiprobable

weights_uni, nodes_uni = np.array( [*zip(*[*disUni.iteritems(0)])] )
plt.plot(nodes_uni, nodes_uni*0, '.')
xl = plt.xlim(a,b)
xvec = np.linspace(xl[0], xl[1], 100)
pdf = scipy.stats.uniform.pdf(xvec,a,b-a)
plt.plot(xvec, pdf)
plt.grid()

## Compute the mean of random draws
expval_MC = np.array([f(s_MC[j]) for j in range(0,M)]).sum() / M
expval_MC


# Compute ∑(f(ϵ)*w_ϵ) for each discretization method
expval_ep = np.array([f(disUni.inode(0,j))*disUni.iweight(0,j) for j in range(disUni.n_inodes(0))]).sum()

expval_ep
expval_MC

##################################################################
############################# LOGNORMAL ###########################
##################################################################

σ = 0.1
μ = 0.3

logn = LogNormal(μ=μ, σ=σ)
logn.μ
logn.σ


distLog = LogNormal(μ, σ)
disLog = distLog.discretize(N=10)
### Here there is an issue with the uniform

## Random draws
M=1000
s_MC = np.random.lognormal(μ, σ, M)

count, bins, ignored = plt.hist(s_MC, 10, density=True)
x = np.linspace(min(bins), max(bins), 10000)
pdf = (np.exp(-(np.log(x) - μ)**2 / (2 * σ**2))  / (x * σ * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, color='r')
plt.axis('tight')
plt.show()

## Plot Equiprobable

weights_Log, nodes_Log = np.array( [*zip(*[*disLog.iteritems(0)])] )
plt.plot(nodes_Log, nodes_Log*0, '.')
xl = plt.xlim()
xvec = np.linspace(xl[0], xl[1], 100)
xvec
pdf = scipy.stats.lognorm.pdf(xvec,s=σ, loc=μ, scale=np.exp(μ) )
plt.plot(xvec, pdf)
plt.grid()


## Compute the mean of random draws
expval_MC = np.array([f(s_MC[j]) for j in range(0,M)]).sum() / M

# Compute ∑(f(ϵ)*w_ϵ) for each discretization method
expval_ep = np.array([f(disLog.inode(0,j))*disLog.iweight(0,j) for j in range(disLog.n_inodes(0))]).sum()

expval_ep
expval_MC








##################################################################
############################# BETA ###########################
##################################################################



α = 2
β = 5

distbeta = Beta(α, β)
disbeta = distbeta.discretize(N=10)
### Here there is an issue with the uniform

## Random draws
M=1000
s_MC = np.random.beta(α, β, M)

count, bins, ignored = plt.hist(s_MC, 10, density=True)
x = np.linspace(min(bins), max(bins), 10000)
pdf = scipy.stats.beta.pdf(x,α, β)
plt.plot(x, pdf, linewidth=2, color='r')
plt.axis('tight')
plt.show()

## Plot Equiprobable
weights_beta, nodes_beta = np.array( [*zip(*[*disbeta.iteritems(0)])] )
plt.plot(nodes_beta, nodes_beta*0, '.')
xl = plt.xlim()
xvec = np.linspace(xl[0], xl[1], 100)
pdf = scipy.stats.beta.pdf(xvec,α, β)
plt.plot(xvec, pdf)
plt.grid()


## Compute the mean of random draws
expval_MC = np.array([f(s_MC[j]) for j in range(0,M)]).sum() / M

# Compute ∑(f(ϵ)*w_ϵ) for each discretization method
expval_ep = np.array([f(disbeta.inode(0,j))*disbeta.iweight(0,j) for j in range(disbeta.n_inodes(0))]).sum()

expval_ep
expval_MC





#################################################################################
############################ SOME EXTRA STUFF  ##################################
#################################################################################
#################################################################################



logn = LogNormal(μ=μ, σ=σ)
logn.μ
logn.σ

dp = logn.discretize(N=10)
dp2 = logn.discretize(N=10, method='equiprobable')


nodes = np.array([x for (w,x) in dp.iteritems(0)])

plt.plot(nodes, nodes*0,'.')
plt.xlim(-1,2)
plt.grid()



res_gh = n.discretize(10)
res_ep = n.discretize(10, method='equiprobable')

for (w,x) in res_ep.iteritems(0):
    print(w,x)


# neglect integration nodes whose probability is smaller than 1e-5
for (w,x) in res_gh.iteritems(0,eps=1e-5):
    print(w,x)

def f(x):
    return x**2


val = quad(lambda u: f(u)/np.sqrt(2*np.pi*σ**2)*np.exp(-(u-μ)**2/(2*σ**2)), -5, 5)


v0 = sum([f(x)*w for (w,x) in res_gh.iteritems(0)])
v1 = sum([f(x)*w for (w,x) in res_ep.iteritems(0)])

print(v0, v1, val)

sim = norm.simulate(10000,2)

sim.mean()
sim.std()



dis = n.discretize(N=50, method='equiprobable')

weights, nodes = np.array( [*zip(*[*dis.iteritems(0)])] )


plt.plot(nodes, nodes*0, '.')
xl = plt.xlim()


xvec = np.linspace(xl[0], xl[1], 100)
pdf = scipy.stats.norm.pdf(xvec)
plt.plot(xvec, pdf)
plt.grid()


from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
a = 2
b = 2
x = np.arange (-50, 50, 0.1)
y = beta.pdf(x,a,b, scale=100, loc=-50)
plt.plot(x,y)



from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
a =  2
b = 4
x = np.linspace(-50, 50, 100)
y1 = beta.pdf(x,2, 2, scale=100, loc=-50)
y2 = beta.pdf(x,3, 3, scale=100, loc=-50)
y3 = beta.pdf(x, 4,4, scale=100, loc=-50)

plt.plot(x, y1, "-", x, y2, "r--", x, y3, "g--")



###########################
logn = LogNormal(μ=μ, σ=σ)
logn.μ
logn.σ

dp = logn.discretize(N=10)
dp2 = logn.discretize(N=10, method='equiprobable')


nodes = np.array([x for (w,x) in dp.iteritems(0)])

plt.plot(nodes, nodes*0,'.')
plt.xlim(-1,2)
plt.grid()



res_gh = n.discretize(10)
res_ep = n.discretize(10, method='equiprobable')

for (w,x) in res_ep.iteritems(0):
    print(w,x)


# neglect integration nodes whose probability is smaller than 1e-5
for (w,x) in res_gh.iteritems(0,eps=1e-5):
    print(w,x)

def f(x):
    return x**2


val = quad(lambda u: f(u)/np.sqrt(2*np.pi*σ**2)*np.exp(-(u-μ)**2/(2*σ**2)), -5, 5)


v0 = sum([f(x)*w for (w,x) in res_gh.iteritems(0)])
v1 = sum([f(x)*w for (w,x) in res_ep.iteritems(0)])

print(v0, v1, val)

sim = norm.simulate(10000,2)

sim.mean()
sim.std()



dis = n.discretize(N=50, method='equiprobable')

weights, nodes = np.array( [*zip(*[*dis.iteritems(0)])] )


plt.plot(nodes, nodes*0, '.')
xl = plt.xlim()


xvec = np.linspace(xl[0], xl[1], 100)
pdf = scipy.stats.norm.pdf(xvec)
plt.plot(xvec, pdf)
plt.grid()
