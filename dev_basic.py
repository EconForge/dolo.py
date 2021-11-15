from dolo.numeric.serial_operations import serial_solve_numba, serial_solve

from dolo.algos.steady_state import find_steady_state
from dolo import *
import time


from dolo.algos.new_time_iteration import Euler
from dolo.algos.new_time_iteration import improved_time_iteration as new_iti
from dolo.algos.new_time_iteration import time_iteration as new_ti


model = yaml_import("examples/models/rbc.yaml")

from dolo.algos.new_time_iteration import Euler

# F0 = Euler(model, ignore_constraints=True)
# r0 = F0(F0.x0, F0.x0)


new_iti(model)
improved_time_iteration(model)


print("------------------------")

new_iti(model)
improved_time_iteration(model)

exit()



F = Euler(model, ignore_constraints=False)


r1 = F(F.x0, F.x0)

r2, J = F.d_A(F.x0)
r3, L = F.d_B(F.x0)


print( abs(r2.x0-r1.x0).max())
print( abs(r3.x0-r1.x0).max())
# print( abs(r2.x0-r1.x0).max())

res,dres,jres = improved_time_iteration(model)

print("Before")

print(abs(r1.x0-res).max())
print(abs(J.x0-dres).max())
print(abs(L.M_ij-jres).max())


from dolo.algos.improved_time_iteration import smooth

from dolo.algos.new_time_iteration import smooth2

lb = F.bounds[0]
ub = F.bounds[1]

res, dres, jres = smooth(res, F.x0.x0 - lb.x0, dres=dres, jres=jres)
res[...] *= -1
dres[...] *= -1
jres[...] *= -1
res, dres, jres = smooth(res, ub.x0 - F.x0.x0, dres=dres, jres=jres, pos=-1.0)
res[...] *= -1
dres[...] *= -1
jres[...] *= -1


print("After")

print(abs(r1.x0-res).max())
print(abs(J.x0-dres).max())
print(abs(L.M_ij-jres).max())



exit()



# R,J,L = smooth2(R, F.x0, lb, ub, J=J, L=L.M_ij)


# focus on J.
print("Focus on J")
R,J = F.d_A(F.x0)
R,L = F.d_B(F.x0)
R,J = smooth2(R, F.x0, lb, ub, J=J)
print(abs(J.x0-dres).max())
exit()

# print("After")

# print(abs(R.x0-res).max())
# print(abs(L.M_ij-jres).max())

# exit()


# # print("Variant 1")


# R,J = F.d_A(F.x0)
# # R,L = F.d_B(F.x0)

# R0 = R.x0.copy()
# J0 = J.x0.copy()


# R,J = smooth2(R, F.x0, lb, ub, dres=J)
# print(abs(R.x0-res).max())
# print(abs(J.x0-dres).max())

# print(abs(R.x0-R0).max())
# print(abs(J.x0-J0).max())

# print("Variant 2")


# # R,J = F.d_A(F.x0)
# R,L = F.d_B(F.x0)

# R,L = smooth2(R, F.x0, lb, ub, jres=L.M_ij)

# print(abs(R.x0-res).max())
# print(abs(L-jres).max())


# exit()


# print("Ready ?")
# print("Set.")
# print("Go!")


improved_time_iteration(model, ignore_constraints=True, dr0=D())
new_iti(model, dr0=dr0)


time_iteration(model, maxit=5)

sol = new_ti(model, inner_maxit=6)

new_iti(model, interp_method="linear")

new_iti(model, dr0=sol.dr, grid={"endo": {"n": [15,15]}})


# TODO: not very sure whether interp_method should be an argument of grid, or a separate one

exit()



# x_ref =improved_time_iteration(model, complementarities=False, maxbsteps=1)

import time  

new_iti(model)

new_iti(model)


# t1 = time.time()
# solve(model)
# t2 = time.time()


# t1 = time.time()
# solve(model, verbose=False, maxit=300)
# t2 = time.time()

# print(f"Elapsed: {t2-t1}")
