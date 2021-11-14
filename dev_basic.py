from dolo.numeric.serial_operations import serial_solve_numba, serial_solve

from dolo.algos.steady_state import find_steady_state
from dolo import *
import time


from dolo.algos.new_time_iteration import Euler
from dolo.algos.new_time_iteration import improved_time_iteration as new_iti
from dolo.algos.new_time_iteration import time_iteration as new_ti


model = yaml_import("examples/models/rbc.yaml")

from dolo.algos.new_time_iteration import Euler

F = Euler(model)

res = F(F.x0, F.x0)
print(res.norm())

solp = perturb(model)

dr0 = solp.dr
class D:
    # def eval_is(self,i,s):
    #     return dr0(i,s)
    def eval_ms(self,m,s):
            return dr0(m,s)

improved_time_iteration(model, ignore_constraints=True)
new_iti(model)

print("Ready ?")
print("Set.")
print("Go!")


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