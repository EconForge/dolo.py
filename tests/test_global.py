from dolo import *

import numpy
from dolo.numeric.smolyak import SmolyakGrid



#print numpy.row_stack([true_values,interpolated_values_spline,interpolated_values_smolyak]).T

#from mayavi import mlab
#mlab.figure(bgcolor=(1.0,1.0,1.0))
##mlab.surf(abs(interpolated_values_smolyak-true_values),warp_scale="auto")
##        mlab.surf(abs(interpolated_values_spline-true_values),warp_scale="auto")
#mlab.surf(abs(interpolated_values_smolyak-true_values),warp_scale="auto")
#mlab.colorbar()


model = yaml_import('../examples/global_models/optimal_growth.yaml')

dr = approximate_controls(model, return_dr=True)

from dolo.numeric.perturbations_to_states import  interim_gm


[gm, g_fun, f_fun] = interim_gm(model, True, True, 1)

def asymptotic_variance( A, sigma, T=100 ):
    '''Computes asymptotic variance of the AR(1) process defined by:
    $X_t = A X_{t-1} + \epsilon_t$
    where the $\epsilon_t$ follows a random law with covariance sigma.
    '''
    import numpy
    p = A.shape[0] # number of variables
    Q = numpy.zeros( (p,p), dtype=numpy.float )
    for i in range(T):
        Q = numpy.dot( A.T, numpy.dot( Q, A)) + sigma
    return Q

import numpy.linalg

print numpy.linalg.eigvals(dr.P)
Q = asymptotic_variance(dr.P, dr.sigma)

#print ( numpy.diag(Q)**0.5 )

eigs = numpy.linalg.eigh(Q)


P = eigs[1]

x1 = eigs[1][:,0]
x2 = eigs[1][:,1]
#
#print('v1')
#print numpy.dot( Q,x1) - eigs[0][0]*x1
#print('v2')
#print numpy.dot( Q,x2) - eigs[0][1]*x2


e1 = numpy.array([1,0])
e2 = numpy.array([0,1])


n_S = 3

dev = eigs[0]
std  = dev**0.5 * n_S


s0 = dr.S_bar
bounds = numpy.row_stack([-std,std])
bounds[:,0] += s0[0]
bounds[:,1] += s0[1]
print('bounds')
print bounds

P = eigs[1]

sg = SmolyakGrid( bounds, 7, P)

[y,x,parms] = model.read_calibration()

xinit = dr( sg.grid)
sg.fit_values(xinit)

n_e = 5
epsilons = np.zeros((1,n_e))
weights = np.ones(n_e)/n_e

#from matplotlib import pyplot
##print sg.grid
#pyplot.plot(sg.grid[0,:],sg.grid[1,:],'o')
#pyplot.show()


print abs( sg.grid ).min()
print abs(sg.grid).max()
from dolo.compiler.compiler_global import GlobalCompiler, time_iteration, deterministic_residuals
gc = GlobalCompiler(model)

res = deterministic_residuals(sg.grid, xinit, sg, gc.f, gc.g, parms)



dr_smol = time_iteration(sg.grid, sg, xinit, gc.f, gc.g, parms, epsilons, weights, verbose=True )

