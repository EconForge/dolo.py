import numpy
from numpy import array, zeros
from interpolation.smolyak import SmolyakGrid as SmolyakGrid0
from interpolation.smolyak import SmolyakInterp, build_B
from dolo.numeric.grids import cat_grids, n_nodes, node
from dolo.numeric.grids import UnstructuredGrid, CartesianGrid, SmolyakGrid, EmptyGrid
from dolo.numeric.misc import mlinspace
import scipy
from dolo.numeric.grids import *
# from dolo.numeric.decision_rule import CallableDecisionRule, cat_grids
import numpy as np

import numpy as np


def filter_controls(a,b,ndims,controls):

    from interpolation.splines.filter_cubic import filter_data, filter_mcoeffs

    dinv = (b-a)/(ndims-1)
    ndims = array(ndims)
    n_m, N, n_x = controls.shape
    coefs = zeros((n_m,) + tuple(ndims + 2) + (n_x,))
    for i_m in range(n_m):
        tt = filter_mcoeffs(a, b, ndims, controls[i_m, ...])
        # for i_x in range(n_x):
        coefs[i_m, ...] = tt
    return coefs

class Linear:
    pass

class Cubic:
    pass

class Chebychev:
    pass

interp_methods = {
    'cubic': Cubic(),
    'linear': Linear(),
    'multilinear': Linear(),
    'chebychev': Chebychev()
}

###

class CallableDecisionRule:

    def __call__(self, *args):
        args = [np.array(e) for e in args]
        if len(args)==1:
            if args[0].ndim==1:
                return self.eval_s(args[0][None,:])[0,:]
            else:
                return self.eval_s(args[0])
        elif len(args)==2:
            if args[0].dtype in ('int64','int32'):
                (i,s) = args
                if s.ndim == 1:
                    return self.eval_is(i, s[None,:])[0,:]
                else:
                    return self.eval_is(i, s)
                return self.eval_is()
            else:
                (m,s) = args[0],args[1]
                if s.ndim == 1 and m.ndim == 1:
                    return self.eval_ms(m[None,:], s[None,:])[0,:]
                elif m.ndim == 1:
                    m = m[None,:]
                elif s.ndim == 1:
                    s = s[None,:]
                return self.eval_ms(m,s)


class DecisionRule(CallableDecisionRule):

    exo_grid: Grid
    endo_grid: Grid

    def __init__(self, exo_grid: Grid, endo_grid: Grid, interp_method='cubic', dprocess=None, values=None):

        if interp_method not in interp_methods.keys():
            raise Exception(f"Unknown interpolation type: {interp_method}. Try one of: {tuple(interp_methods.keys())}")

        self.exo_grid = exo_grid
        self.endo_grid = endo_grid
        self.interp_method = interp_method

        self.__interp_method__ = interp_methods[interp_method]

        args = (self, exo_grid, endo_grid, interp_methods[interp_method])

        try:
            aa = args + (None, None)
            fun = eval_ms[tuple(map(type, aa))]
            self.__eval_ms__ = fun
        except Exception as exc:
            pass

        try:
            aa = args + (None, None)
            fun = eval_is[tuple(map(type, aa))]
            self.__eval_is__ = fun
        except Exception as exc:
            pass

        try:
            aa = args + (None, None)
            fun = eval_s[tuple(map(type, aa))]
            self.__eval_s__ = fun
        except Exception as exc:
            pass

        fun = get_coefficients[tuple(map(type, args))]
        self.__get_coefficients__ = fun

        if values is not None:
            self.set_values(values)

    def set_values(self, x):
        self.coefficients = self.__get_coefficients__(self, self.exo_grid, self.endo_grid, self.__interp_method__, x)

    def eval_ms(self, m, s):
        return self.__eval_ms__(self, self.exo_grid, self.endo_grid, self.__interp_method__, m, s)

    def eval_is(self, i, s):
        return self.__eval_is__(self, self.exo_grid, self.endo_grid, self.__interp_method__, i, s)

    def eval_s(self, s):
        return self.__eval_s__(self, self.exo_grid, self.endo_grid, self.__interp_method__, s)


# this is *not* meant to be used by users

from multimethod import multimethod

# Cartesian x Cartesian x Linear

@multimethod
def get_coefficients(itp, exo_grid: CartesianGrid, endo_grid: CartesianGrid, interp_type: Linear, x):
    grid = cat_grids(exo_grid, endo_grid) # one single CartesianGrid
    xx = x.reshape(tuple(grid.n)+(-1,))
    return xx


@multimethod
def eval_ms(itp, exo_grid: CartesianGrid, endo_grid: CartesianGrid, interp_type: Linear, m, s):

    assert(m.ndim==s.ndim==2)

    grid = cat_grids(exo_grid, endo_grid) # one single CartesianGrid
    coeffs = itp.coefficients
    d = len(grid.n)
    gg = tuple( [(grid.min[i], grid.max[i], grid.n[i]) for i in range(d)] )
    from interpolation.splines import eval_linear

    x = np.concatenate([m, s], axis=1)

    return eval_linear(gg, coeffs, x)


@multimethod
def eval_is(itp, exo_grid: CartesianGrid, endo_grid: CartesianGrid, interp_type: Linear, i, s):
    m = exo_grid.node(i)[None,:]
    return eval_ms(itp, exo_grid, endo_grid, interp_type, m, s)


# Cartesian x Cartesian x Cubic

@multimethod
def get_coefficients(itp, exo_grid: CartesianGrid, endo_grid: CartesianGrid, interp_type: Cubic, x):

    from interpolation.splines.prefilter_cubic import prefilter_cubic
    grid = cat_grids(exo_grid, endo_grid) # one single CartesianGrid
    x = x.reshape(tuple(grid.n)+(-1,))
    d = len(grid.n)
    # this gg could be stored as a member of itp
    gg = tuple( [(grid.min[i], grid.max[i], grid.n[i]) for i in range(d)] )
    return prefilter_cubic(gg, x)

@multimethod
def eval_ms(itp, exo_grid: CartesianGrid, endo_grid: CartesianGrid, interp_type: Cubic, m, s):

    from interpolation.splines import eval_cubic

    assert(m.ndim==s.ndim==2)

    grid = cat_grids(exo_grid, endo_grid) # one single CartesianGrid
    coeffs = itp.coefficients
    d = len(grid.n)
    gg = tuple( [(grid.min[i], grid.max[i], grid.n[i]) for i in range(d)] )

    x = np.concatenate([m, s], axis=1)

    return eval_cubic(gg, coeffs, x)


@multimethod
def eval_is(itp, exo_grid: CartesianGrid, endo_grid: CartesianGrid, interp_type: Cubic, i, s):
    m = exo_grid.node(i)[None,:]
    return eval_ms(itp, exo_grid, endo_grid, interp_type, m, s)



# UnstructuredGrid x Cartesian x Linear

@multimethod
def get_coefficients(itp, exo_grid: UnstructuredGrid, endo_grid: CartesianGrid, interp_type: Linear, x):
    return [x[i].copy() for i in range(x.shape[0])]

@multimethod
def eval_is(itp, exo_grid: UnstructuredGrid, endo_grid: CartesianGrid, interp_type: Linear, i, s):

    from interpolation.splines import eval_linear
    assert(s.ndim==2)

    grid = endo_grid # one single CartesianGrid
    coeffs = itp.coefficients[i]
    d = len(grid.n)
    gg = tuple( [(grid.min[i], grid.max[i], grid.n[i]) for i in range(d)] )

    return eval_linear(gg, coeffs, s)


# UnstructuredGrid x Cartesian x Cubic

@multimethod
def get_coefficients(itp, exo_grid: UnstructuredGrid, endo_grid: CartesianGrid, interp_type: Cubic, x):
    from interpolation.splines.prefilter_cubic import prefilter_cubic
    grid = endo_grid # one single CartesianGrid
    d = len(grid.n)
    # this gg could be stored as a member of itp
    gg = tuple( [(grid.min[i], grid.max[i], grid.n[i]) for i in range(d)] )
    return [prefilter_cubic(gg, x[i]) for i in range(x.shape[0])]


@multimethod
def eval_is(itp, exo_grid: UnstructuredGrid, endo_grid: CartesianGrid, interp_type: Cubic, i, s):

    from interpolation.splines import eval_cubic
    assert(s.ndim==2)

    grid = endo_grid # one single CartesianGrid
    coeffs = itp.coefficients[i]
    d = len(grid.n)
    gg = tuple( [(grid.min[i], grid.max[i], grid.n[i]) for i in range(d)] )

    return eval_cubic(gg, coeffs, s)


# UnstructuredGrid x Cartesian x Linear

@multimethod
def get_coefficients(itp, exo_grid: UnstructuredGrid, endo_grid: CartesianGrid, interp_type: Linear, x):
    return [x[i].copy() for i in range(x.shape[0])]

@multimethod
def eval_is(itp, exo_grid: UnstructuredGrid, endo_grid: CartesianGrid, interp_type: Linear, i, s):

    from interpolation.splines import eval_linear
    assert(s.ndim==2)

    grid = endo_grid # one single CartesianGrid
    coeffs = itp.coefficients[i]
    d = len(grid.n)
    gg = tuple( [(grid.min[i], grid.max[i], grid.n[i]) for i in range(d)] )

    return eval_linear(gg, coeffs, s)


# Empty x Cartesian x Cubic

@multimethod
def get_coefficients(itp, exo_grid: EmptyGrid, endo_grid: CartesianGrid, interp_type: Cubic, x):
    from interpolation.splines.prefilter_cubic import prefilter_cubic
    grid = endo_grid # one single CartesianGrid
    d = len(grid.n)
    # this gg could be stored as a member of itp
    gg = tuple( [(grid.min[i], grid.max[i], grid.n[i]) for i in range(d)] )
    return prefilter_cubic(gg, x)


@multimethod
def eval_s(itp, exo_grid: EmptyGrid, endo_grid: CartesianGrid, interp_type: Cubic, s):
    from interpolation.splines import eval_cubic
    assert(s.ndim==2)
    grid = endo_grid # one single CartesianGrid
    coeffs = itp.coefficients
    d = len(grid.n)
    gg = tuple( [(grid.min[i], grid.max[i], grid.n[i]) for i in range(d)] )

    return eval_cubic(gg, coeffs, s)


####

class ConstantDecisionRule(CallableDecisionRule):

    def __init__(self, x0):
        self.x0 = x0

    def eval_s(self, s):
        if s.ndim==1:
            return self.x0
        else:
            N = s.shape[0]
            return self.x0[None,:].repeat(N,axis=0)

    def eval_is(self, i, s):
        return self.eval_s(s)

    def eval_ms(self, m, s):
        return self.eval_s(s)



import dolang
import dolang.symbolic
from dolang.symbolic import stringify_symbol
from dolo.numeric.decision_rule import CallableDecisionRule
from dolang.factory import FlatFunctionFactory
from dolang.function_compiler import make_method_from_factory

class CustomDR(CallableDecisionRule):

    def __init__(self, values, model=None):

        from dolang.symbolic import sanitize, stringify

        exogenous = model.symbols['exogenous']
        states = model.symbols['states']
        controls = model.symbols['controls']
        parameters = model.symbols['parameters']

        preamble = dict([(s, values[s]) for s in values.keys() if s not in controls])
        equations = [values[s] for s in controls]

        variables = exogenous + states + controls + [*preamble.keys()]

        preamble_str = dict()

        for k in [*preamble.keys()]:
            v = preamble[k]
            if '(' not in k:
                vv = f'{k}(0)'
            else:
                vv = k

            preamble_str[stringify(vv)] = stringify(sanitize(v, variables))

        # let's reorder the preamble
        from dolang.triangular_solver import get_incidence, triangular_solver
        incidence = get_incidence(preamble_str)
        sol = triangular_solver(incidence)
        kk = [*preamble_str.keys()]
        preamble_str = dict([(kk[k], preamble_str[kk[k]]) for k in sol])

        equations = [dolang.symbolic.sanitize(eq, variables) for eq in equations]
        equations_strings = [dolang.stringify(eq, variables) for eq in equations]

        args = dict([
            ('m', [(e,0) for e in exogenous]),
            ('s', [(e,0) for e in states]),
            ('p', [e for e in parameters])
        ])

        args = dict( [(k,[stringify_symbol(e) for e in v]) for k,v in args.items()] )

        targets = [stringify_symbol((e,0)) for e in controls]

        eqs = dict([ (targets[i], eq) for i, eq in enumerate(equations_strings) ])

        fff = FlatFunctionFactory(preamble_str, eqs, args, 'custom_dr')

        fun, gufun = make_method_from_factory(fff)

        self.p = model.calibration['parameters']
        self.exo_grid = model.exogenous.discretize() # this is never used
        self.endo_grid = model.get_grid()
        self.gufun = gufun

    def eval_ms(self, m, s):

        return self.gufun(m, s, self.p)
