#from dolo import *
##
#model = yaml_import('/home/pablo/Documents/Work/Florin/flexible.yaml')
##
#from dolo.compiler.compiler_global import CModel
#cm = CModel(model)
##
#[lb, ub] = cm.x_bounds
#
##
#
#
#dr = global_solve(model, verbose=True, smolyak_order=4, interp_type='spline', numdiff=True, polish=False)
#
#dr = global_solve(model, maxit=2, verbose=True, smolyak_order=4, interp_type='smolyak', numdiff=True, polish=True)


#from matplotlib.pylab import *
#
#N = 100
#bvec = linspace(0,0.5)
#svec = row_stack([bvec, bvec*0+0.1, bvec*0+1])
#xvec = dr(svec)
#
#plot(bvec, xvec[0,:])
#show()


