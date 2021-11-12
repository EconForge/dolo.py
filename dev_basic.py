from dolo.algos.steady_state import find_steady_state
from dolo import *


model = yaml_import("examples/models/rbc.yaml")


time_iteration(model, maxit=1)

from dolo.algos.new_time_iteration import Euler

F = Euler(model)

x0 = F.x0

res0 = F(x0, x0)

print(res0.data)


for  i in range(10):

    F.dr.set_values(x0.x0)

    r = ( F(x0) )

    print(r.norm())

    J = ( F.d_A(x0) )

    dx = J.solve(r)

    error = J@dx - r
    print("Error : {}".format(error.norm()))


    x0 = x0 - dx*0.1


# from dolo.algos.new_time_iteration import CVector

# A = CVector(x0.x0[:,:,:,None].repeat(2, axis=3))
# B = CVector(x0.x0[:,:,:,None].repeat(2, axis=3))

# print(A.x0.shape)
# print(B.x0.shape)
# print(x0.x0.shape)


# (A@B).datashape
# (A@x0).datashape

# # print(x0.shape)

# F(x0, x0)


# res = model.discretize()


# sol = time_iteration(model, maxit=5)
# print(sol.dr.endo_grid.n)


# sol = time_iteration(model, maxit=5)
# print(sol.dr.endo_grid.n)

# sol = time_iteration(model, maxit=5, grid=dict(interpolation="linear"))
# print(sol.dr.endo_grid.n)