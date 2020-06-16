from typing import Tuple
from math import exp

### Define symbols of models

symbols = dict(
    states= ["k", "z"],
    controls= ["i", "n"],
    exogenous= ["e_z"],
    parameters= ["α", "σ", "β", "δ", "χ", "η"]
)

# def transition(m: Tuple[float], s: Tuple[float, float], x: Tuple[float, float], M: Tuple[float], p: Tuple[float, float, float, ...])->Float64:

def transition_(m, s, x, M, p):

    # we use _ to denote predetermined states

    k_, z_ = s
    i_, n_ = x
    e_z, = M
    α, σ, β, δ, χ, η = p

    k = (1-δ) *k_ + i_

    S = (k_,)

    return S


def arbitrage_(m, s, x, M, S, X, p):

    # we use _ to denote predetermined states

    # e_z_, = m_
    k_, z_ = s
    i_, n_ = x

    e_z, = M
    k, z = X
    i, n = M

    α, σ, β, δ, χ, η = p


    # definitions today
    y_= exp(z_)*k_^α*n_^(1-α)
    c_= y_ - i_
    w_= (1-α)*y_/n_

    # definitions tomorrow
    y = exp(z)*k^α*n^(1-α)
    c = y - i
    rk = α*y/k


    res_1 = χ*n_^η*c_^σ - w_
    res_2 = 1 - β*(c_/c)^(σ)*(1-δ+rk)

    return (res_1, res_2)


# here is more or less the compilation phase arises within dolo:

# guvectorize creates functions vectorized, with respect to every argument,
# with numpy-style broadcasting rules

from numba import guvectorize
@guvectorize('right signature')
def transition_gu(m, s, x, M, p, out):  
    tmp = transition_(m, s, x, M, p)
    out[0] = tmp[0]
    out[1] = tmp[1]

@guvectorize('right signature')
def arbitrage_gu(m, s, x, M, p, out):  
    tmp = transition_(m, s, x, M, p)
    out[0] = tmp[0]
    out[1] = tmp[1]

##
# standard_function creates function which follows numpy conventions, and optionnally computes
# numerical derivatives

from dolo.... import standard_function

transition = standard_function(transition, ...)
arbitrage = standard_function(arbitrage, ...)

functions = dict(
    'transition',
    'arbitrage',
)


#####
##### parameters definition
#####

# we need to define a flat dictionary Dict[str, float]
# we'll use a dolang routine, to use symbolic definitions instead, as strings

parameters_definitions = {
    'beta': '0.99',
    'phi': '1',
    'delta': '0.025',
    'alpha': ' 0.33',
    'rho': '0.8',
    'sigma': '5',
    'eta': '1',
    'sig_z': '0.016',
    'zbar': '0',
    'chi': 'w/c^sigma/n^eta',
    'c_i': '1.5',
    'c_y': '0.5',
    'e_z': '0.0',
    'n': '0.33',
    'z': 'zbar',
    'rk': '1/beta-1+delta',
    'w': '(1-alpha)*exp(z)*(k/n)^(alpha)',
    'k': 'n/(rk/alpha)^(1/(1-alpha))',
    'y': 'exp(z)*k^alpha*n^(1-alpha)',
    'i': 'delta*k',
    'c': 'y - i',
    'V': 'log(c)/(1-beta)',
    'u': 'c^(1-sigma)/(1-sigma) - chi*n^(1+eta)/(1+eta)',
    'm': 'beta/c^sigma*(1-delta+rk)',
    'kss': '10',
}

from dolang import triangular_solver
calibration_dict = triangular_solver(parameters_definitions)
# now, dependencies between parameters are lost for ever

from dolo.numeric import CalibrationDict
calibration = CalibrationDict(symbols, calibration_dict)
# this object is cool, because it admits several indexation schemes:
# calibration['states', 'controls'] or calibration['k', 'l', 'β'] for instance


###
### Exogenous process
###

from dolo.numeric.processes_iid import UNormal
process = UNormal(σ=0.001)


###
### Grid construction
###

from dolo.numeric.grids import CartesianGrid


grid = CartesianGrid(min=[1.0, -0.1], max=[5.0, 0.1], n=[10, 10])

### Now the Python class is trivial.

class PythonModel:

    def __init__(self, symbols, functions, calibration, grid, exogenous):

        self.symbols = symbols
        self.functions = functions
        self.calibration = calibration
        self.grid = grid
        self.exogenous = exogenous



model = PythonModel(symbols, functions, calibration, grid, process)



# now, algoritms in dolo.algos, should be agnostic which respect to the origin of
# the above fields (maybe PythonModel should subclass model though)

from dolo.algos.time_iteration import time_iteration
dr = time_iteration(model)