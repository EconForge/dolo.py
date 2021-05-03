from dolo import *


model = yaml_import('examples/models/consumption_savings.yaml')

m,s,a,x,p = model.calibration['exogenous','states','poststates', 'controls','parameters']

h = model.functions['expectation']
gt = model.functions['half_transition']
τ = model.functions['direct_response_egm']

aτ = model.functions['direct_response_egm']

assert(len(model.symbols['states'])==1)

z = h(m,s,x,p)
x = τ(m,a,x,p)
ss = aτ(m, a, x)

print(ss)

import numpy as np

dr0 = lambda i,s: np.minimum(s, 1+ 0.05*(s-1))

s = model.endo_grid.nodes

print( dr0(0, s) )

dp = model.exogenous.discretize()

n_m = dp.n_nodes

a = np.linspace(0.1, 10, 10)[:,None]

N = a.shape[0]

n_h = len(model.symbols['expectations'])

z = np.zeros( (n_m,N,1) )

dr = dr0

for i_m in range(n_m):
    m = dp.node(i_m)
    for i_M in range(dp.n_inodes(i_m)):
        w = dp.iweight(i_m, i_M)
        M = dp.inode(i_m, i_M)
        S = gt(m, a, M, p)
        X = dr(i_m, S)
        z[i_m,:,:] += w*h(M,S,X,p)

