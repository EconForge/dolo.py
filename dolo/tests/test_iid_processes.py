from matplotlib import pyplot as plt
import scipy
from scipy.integrate import quad
from dolo.numeric.processes_iid import *

σ = 0.1
μ = 0.2

norm = UNormal(mu=μ, sigma=σ)
norm2 = UNormal(μ=μ, σ=σ)

norm.discretize()

unif = Uniform(-1, 2)

dp = unif.discretize(N=10)

nodes = np.array([x for (w,x) in dp.iteritems(0)])

plt.plot(nodes, nodes*0,'.')
plt.xlim(-1,2)
plt.grid()



res_gh = norm.discretize(10)
res_ep = norm.discretize(10, method='equiprobable')

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



dis = norm.discretize(N=50, method='equiprobable')

weights, nodes = np.array( [*zip(*[*dis.iteritems(0)])] )


plt.plot(nodes, nodes*0, '.')
xl = plt.xlim()


xvec = np.linspace(xl[0], xl[1], 100)
pdf = scipy.stats.norm.pdf(xvec)
plt.plot(xvec, pdf)
plt.grid()
