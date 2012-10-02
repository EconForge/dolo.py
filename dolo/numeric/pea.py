from numpy import tile, repeat
import numpy as np

# compiled model is assumed to be a fgh model

def compute_expectations_phi( cm, dr_x, s, x, p, nodes, weights):

    N = s.shape[1]
    n_s = s.shape[0]
    n_x = x.shape[0]
    n_h = len(cm.model['variables_groups']['expectations'])
    Q = len(weights)

    ee = nodes[numpy.newaxis,:,:]

    z = numpy.zeros( (n_h, N) )
    for i in range(Q):
        e = ee[:,:,i:i+1]
        w = weights[i]
        S = cm.g(s,x,e,p)
        X = dr_x(S)
        h = cm.h(S,X,p)
        z += h*w
    return z

def compute_expectations_psi( cm, dr_x, dr_z, s, x, p, nodes, weights):

    N = s.shape[1]
    n_s = s.shape[0]
    n_x = x.shape[0]
    n_h = len(cm.model['variables_groups']['expectations'])
    Q = len(weights)

    ss = tile(s, (1,Q))
    xx = tile(x, (1,Q))
    ee = repeat(nodes, N , axis=1)
    SS = cm.g(ss,xx,ee,p)

    xxstart = dr_x(SS)

    ZZ = dr_z(SS)


    ff = lambda xt: cm.f(SS,xt,ZZ,p)
    [XX,nit] = newton_solver(ff , xxstart, numdiff=True,infos=True)

    hh = cm.h(SS,XX,p)
    z = np.zeros( (n_x,N) )
    for i in range(Q):
        z += weights[i] * hh[:,N*i:N*(i+1)]

    return z


from dolo.numeric.newton import newton_solver


def pea_solve( cm, grid, dr_x, dr_z, p, nodes, weights ):

    tol = 1e-9
    err = 10
    maxit = 500
    it = 0

    x0 = dr_x(grid)
    z0 = dr_z(grid)
#    z0 = compute_expectations_phi( cm, dr_x, grid, x0, p, nodes, weights)

    while err > tol and it < maxit:

        it += 1

        fobj = lambda x: cm.f( grid, x, z0, p )

        x1 = newton_solver( fobj, x0, numdiff=True)
        dr_x.set_values(x1) # I don't really need it

        z1 = compute_expectations_psi(cm, dr_x, dr_h, grid, x1, params, nodes, weights)
        dr_z.set_values(z1)

        err = abs(z1 - z0).max()
        err2 = abs(x1 - x0).max()
        print(err, err2)
        x0 = x1
        z0 = z1

    print('finished in {} iterations'.format(it))

    return [x0,z0]



from dolo import yaml_import

from dolo.numeric.perturbations_to_states import approximate_controls

model = yaml_import('../../../examples/global_models/rbc_fgah.yaml')


dr_pert = approximate_controls(model, order=1, substitute_auxiliary=True)

print(dr_pert.X_bar)




from dolo.numeric.smolyak import SmolyakGrid
from dolo.numeric.multilinear import MultilinearInterpolator
from dolo.numeric.global_solve import global_solve
dr_smol = global_solve(model, smolyak_order=2, maxit=2, polish=True)


from numpy import array

smin = dr_smol.bounds[0,:]
smax = dr_smol.bounds[1,:]


dr_x = SmolyakGrid( smin, smax, 3)
dr_h = SmolyakGrid( smin, smax, 3)

grid = dr_x.grid

xh_init = dr_pert(dr_x.grid)

n_x = len(model['variables_groups']['controls'])


dr_x.set_values( xh_init[:n_x,:] )
dr_h.set_values( xh_init[n_x:,:] )

#phi = MultilinearInterpolator( smin, smax, [10,10])
#psi = MultilinearInterpolator( smin, smax, [10,10])

import numpy


from dolo.compiler.cmodel_fgh import CModel_fgah, CModel_fgh_from_fgah

cmt = CModel_fgah(model)

cm = CModel_fgh_from_fgah(cmt)


sigma = cm.model.read_covariances()
params = cm.model.read_calibration()[2]
from dolo.numeric.quadrature import gauss_hermite_nodes
[nodes, weights] = gauss_hermite_nodes( [10], sigma)

[x_sol, z_sol] = pea_solve( cm, grid, dr_x, dr_h, params, nodes, weights )

# <codecell>

dr_x.set_values(x_sol)

# <codecell>


# <codecell>

#dr_glob_3 = global_solve(model_bis, smolyak_order=3)
#dr_glob_4 = global_solve(model_bis, smolyak_order=4)