from dolo import *


model = yaml_import('examples/models/consumption_savings.yaml')

m,s,a,x,p = model.calibration['exogenous','states','poststates', 'controls','parameters']

grid, dp = model.discretize()
s = grid['endo'].nodes

h = model.functions['expectation']
gt = model.functions['half_transition']
τ = model.functions['direct_response_egm']

aτ = model.functions['reverse_state']

assert(len(model.symbols['states'])==1)

z = h(m,s,x,p)
x = τ(m,a,x,p)
ss = aτ(m, a, x)

print(ss)

import numpy as np

sol = time_iteration(model, maxit=1000, details=True)
# tdr = sol_iti.dr
tdr = sol.dr

dr0 = lambda i,s: np.minimum(s, 1+ 0.05*(s-1))
# class DR:
#     def eval_is(self, i, s): np.minimum(0.9*s, 1+ 0.05*(s-1))

# dr0 = DR()

# dr0 = lambda i,s: 0.8*s # 
# sol_iti = improved_time_iteration(model, method='iti', smaxit=5, dr0=sol.dr, verbose=True)
# tdr = sol_iti.dr

dr0 = lambda i,s: tdr.eval_is(i,s)


print( dr0(0, s) )


n_m = dp.n_nodes
n_x = len(model.symbols['controls'])

N_a = 10
a = np.linspace(0.0, 2, N_a)[:,None]**2

N = a.shape[0]

n_h = len(model.symbols['expectations'])


xa = np.zeros( (n_m, N_a,n_x) )
sa = np.zeros( (n_m, N_a,n_x) )

from interpolation import interp

from interpolation.splines import eval_linear

import time

xa = np.zeros( (n_m, N_a,n_x) )
sa = np.zeros( (n_m, N_a,n_x) )

for k in range(0,1000):

    if k==2:
        t1 = time.time()

    if k==0:
        drfut = dr0
    else:
        drfut = lambda i,s: np.minimum(s, eval_linear((sa0[i,:,0],), xa0[i,:,0], s)[:,None])
    
    z = np.zeros( (n_m,N,1) )
    for i_m in range(n_m):
        m = dp.node(i_m)
        for i_M in range(dp.n_inodes(i_m)):
            w = dp.iweight(i_m, i_M)
            M = dp.inode(i_m, i_M)
            S = gt(m, a, M, p)
            X = drfut(i_m, S)
            z[i_m,:,:] += w*h(M,S,X,p)

        xa[i_m,:,:] = τ(m,a,z[i_m,:,:], p)
        sa[i_m,:,:] = aτ(m,a,xa[i_m,:,:], p)
    
    sa0 = sa.copy()
    xa0 = xa.copy()

t2 = time.time()
print(t2-t1)


import matplotlib
matplotlib.use("Qt5Agg")

from matplotlib import pyplot as plt

xvec = np.linspace(0, 5, 1000)

for i in range(xa.shape[0]):
    plt.subplot(1,3,i+1)
    plt.plot(xvec, xvec, linestyle=':', color='black')
    # plt.plot(xvec, dr0(i,xvec[:,None])[:,0], label="Initial Rule")
    plt.plot(xvec, drfut(i,xvec[:,None])[:,0], label="Final Rule")
    plt.plot(sa[i,:,0], xa[i,:,0], '.', label="Final Rule (no extrap)")
    plt.plot(xvec, tdr(i,xvec[:,None])[:,0], label="True Rule", linestyle=':', color='black')
    plt.ylim(0,1.5)
    plt.xlim(0,4)
    plt.grid(True)
    plt.legend()
    plt.title(f'm={dp.node(i)[0]}')
plt.tight_layout()
plt.show(block=True)

# print("Done")