from dataclasses import dataclass

from typing import Dict, List, Tuple

Vector = List[float]
from dolang.vectorize import standard_function
from dolo.numeric.processes import Process, DiscretizedProcess
from dolo.compiler.objects import Domain, CartesianDomain
from dolo.numeric.grids import Grid, ProductGrid

from dolo.compiler.misc import CalibrationDict, calibration_to_vector
from numba import jit

###
### Model API
###

# Here is a simple way to define a model
# which follows the model API


@dataclass
class PureModel:

    symbols: Dict[str, Vector]
    calibration: CalibrationDict
    functions: Dict[str, standard_function]

    # these are not accessed directly
    domain: Domain
    exogenous: Process

    def discretize(self, **kwargs) -> Tuple[Grid, DiscretizedProcess]:

        # returns a discretized grid for exogenous and endogenous variables
        # and a discretization of the exogenous process

        # it can be inherited from dolo.numeric.model.Model

        dp = exogenous.discretize()
        endo_grid = self.domain.discretize()
        exo_grid = dp.grid
        grid = ProductGrid(exo_grid, endo_grid, names=["exo", "endo"])
        return (grid, dp)


###
### Symbols definitions
###

# we define all variable names below
# to construct the calibration dictionary
# afterwards, it is only used to label outputs
# for instance in simulations


symbols = dict(
    states=["z", "k"],
    controls=["n", "i"],
    exogenous=["e_z"],
    parameters=["β", "δ", "α", "ρ", "σ", "η", "χ"],
)

###
### Calibration Definition
###

# ultimately, all is needed is a calibration object
# mapping variable names to values
# here we choose to rely on dolang methods
# to build this dictionary from the expressions
# of the steady-state/calibration

calibration_strings = dict(
    # parameters,
    β="0.99",
    δ="0.025",
    α="0.33",
    ρ="0.8",
    σ="5",
    η="1",
    zbar="0",
    χ="w/c**σ/n**η",
    c_i="1.5",
    c_y="0.5",
    e_z="0.0",
    # = "endogenous variables",
    n="0.33",
    z="zbar",
    rk="1/β-1+δ",
    w="(1-α)*exp(z)*(k/n)**(α)",
    k="n/(rk/α)**(1/(1-α))",
    y="exp(z)*k**α*n**(1-α)",
    i="δ*k",
    c="y - i",
    V="log(c)/(1-β)",
    u="c**(1-σ)/(1-σ) - χ*n**(1+η)/(1+η)",
    m="β/c**σ*(1-δ+rk)",
)


from dolang.triangular_solver import solve_triangular_system

calibration_dict = solve_triangular_system(calibration_strings)

calibration_vector = calibration_to_vector(symbols, calibration_dict)
calibration = CalibrationDict(symbols, calibration_vector)
calibration


###
### Define functions
###

# the basic formulation of the functions (aka kernel)

# take and return "tuples" of floats because tuples are supposed
# to be easy to be optimzed away by the compiler
# after the definitiexogenous, gridt


@jit(nopython=True)
def transition_(m, s, x, M, p):

    z, k = s
    n, i = x
    (e_z,) = m
    β, δ, α, ρ, σ, η, χ = p

    K = (1 - δ) * k + i
    Z = (1 - ρ) * z + e_z

    return (Z, K)


from math import exp


@jit(nopython=True)
def arbitrage_(m, s, x, M, S, X, p):

    z, k = s
    n, i = x
    (e_z,) = m
    β, δ, α, ρ, σ, η, χ = p

    Z, K = S
    N, I = X
    # E_z = M

    y = exp(z) * k**α * n ** (1 - α)
    c = y - i
    w = (1 - α) * y / n

    Y = exp(Z) * K**α * N ** (1 - α)
    C = Y - I
    R = α * Y / K
    # W = (1-α)*y/n

    res_1 = χ * n**η * c**σ - w
    res_2 = 1 - β * (c / C) ** (σ) * (1 - δ + R)

    return (res_1, res_2)


m, s, x, p = calibration["exogenous", "states", "controls", "parameters"]

transition_(m, s, x, m, p) - s
arbitrage_(m, s, x, m, s, x, p)


from numba import guvectorize


@guvectorize(
    ["void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])"],
    "(n_m),(n_s),(n_x),(n_m),(n_p)->(n_s)",
)
def transition_gu(m, s, x, M, p, out):
    tmp = transition_(m, s, x, M, p)
    out[0] = tmp[0]
    out[1] = tmp[1]
    return out


@guvectorize(
    [
        "void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])"
    ],
    "(n_m),(n_s),(n_x),(n_m),(n_s),(n_x_),(n_p)->(n_s)",
)
def arbitrage_gu(m, s, x, M, S, X, p, out):
    tmp = arbitrage_(m, s, x, M, S, X, p)
    out[0] = tmp[0]
    out[1] = tmp[1]
    return out


# now the functions transition_gu and arbitrage_gu
# are generelized ufuncs, which implement numpy-style
# broadcasting
transition_gu(m, s, x, m[None, :].repeat(10, axis=0), p)

# we use another convenience function to get a function which
# is able to computeexogenous, grid derivatives
# I don't like this way to proceed, this should be rewritten
# currently it is only used for perturbations anyway.

from dolang.vectorize import standard_function

transition = standard_function(transition_gu, 2)
arbitrage = standard_function(arbitrage_gu, 2)
[g, g_m, g_s, g_x, g_m] = transition(m, s, x, m, p, diff=True)


functions = {"transition": transition, "arbitrage": arbitrage}

###
### Exogenous process
###

from dolo.numeric.distribution import UNormal

exogenous = UNormal(σ=0.001)

# ###
# ### Discretized grid for endogenous states
# ###

# from dolo.numeric.grids import UniformCartesianGrid
# grid = UniformCartesianGrid(min=[-0.01, 5], max=[0.01, 15], n=[20, 20])

###
### Domain for endogenous states


domain = CartesianDomain(z=[-0.01, 0.01], k=[5, 15])


###
### Solve model
###

# now we should be able to solve the model using
# any of the availables methods

model = PureModel(symbols, calibration, functions, domain, exogenous)

from dolo.algos.time_iteration import time_iteration
from dolo.algos.perturbation import perturb
from dolo.algos.simulations import simulate
from dolo.algos.improved_time_iteration import improved_time_iteration

dr0 = perturb(model)

time_iteration(model)


simulate(model, dr0)
