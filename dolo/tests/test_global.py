#from dolo import *
#
#import numpy
#from dolo.numeric.smolyak import SmolyakGrid
#
#
#
#
#model = yaml_import('../examples/global_models/optimal_growth.yaml')
#
#dr = approximate_controls(model, return_dr=True)
#
#from dolo.numeric.perturbations_to_states import  interim_gm
#
#
#[gm, g_fun, f_fun] = interim_gm(model, True, True, 1)
#from dolo.numeric.timeseries import asymptotic_variance
#
#print numpy.linalg.eigvals(dr.A)
#Q = asymptotic_variance(dr.A, dr.B, dr.sigma)
#
##print ( numpy.diag(Q)**0.5 )
#
#eigs = numpy.linalg.eigh(Q)
#
#
#P = eigs[1]
#
#x1 = eigs[1][:,0]
#x2 = eigs[1][:,1]
##
##print('v1')
##print numpy.dot( Q,x1) - eigs[0][0]*x1
##print('v2')
##print numpy.dot( Q,x2) - eigs[0][1]*x2
#
#
#e1 = numpy.array([1,0])
#e2 = numpy.array([0,1])
#
#
#n_S = 2
#
#dev = eigs[0]
#std  = dev**0.5 * n_S
#
#
#s0 = dr.S_bar
#bounds = numpy.row_stack([-std,std])
#bounds[:,0] += s0[0]
#bounds[:,1] += s0[1]
#print('bounds')
#print bounds
#
#P = eigs[1]
#
#sg = SmolyakGrid( bounds, 5, P)
#
#[y,x,parms] = model.read_calibration()
#
#xinit = dr( sg.grid)
#sg.set_values(xinit)
#
#
#from dolo.compiler.compiler_global import CModel, time_iteration, deterministic_residuals
#
#gc = CModel(model)
#
#
#
#
#res = deterministic_residuals(sg.grid, xinit, sg, gc.f, gc.g, dr.sigma, parms)
#
#
#n_e = 5
#epsilons = np.zeros((1,n_e))
#weights = np.ones(n_e)/n_e
#
#
#dr_smol = time_iteration(sg.grid, sg, xinit, gc.f, gc.g, parms, epsilons, weights, verbose=True )
#
