from typing import Any, Optional, Union
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


def filter_controls(a, b, ndims, controls):

    from interpolation.splines.filter_cubic import filter_data, filter_mcoeffs

    dinv = (b - a) / (ndims - 1)
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
    "cubic": Cubic(),
    "linear": Linear(),
    "multilinear": Linear(),
    "chebychev": Chebychev(),
}

###


class CallableDecisionRule:
    def __call__(self, *args):
        # 1. Fast path: delegate directly to approximator when present
        if hasattr(self, "approximator") and self.approximator is not None:
            return self.approximator.evaluate(self.coefficients, *args)

        # 2. Fallback for custom / legacy decision rules without Approximator
        from dolo.numeric.garray import QP, SliceQP
        from dolo.numeric.grids import EmptyGrid

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, SliceQP):
                is_empty_exo = hasattr(self, "exo_grid") and isinstance(
                    self.exo_grid, EmptyGrid
                )
                if arg.loc is not None and not is_empty_exo:
                    return self.eval_is(arg.loc, arg.endo)
                return self.eval_s(arg.endo)

            if isinstance(arg, QP):
                is_empty_exo = hasattr(self, "exo_grid") and isinstance(
                    self.exo_grid, EmptyGrid
                )
                if (
                    isinstance(arg.loc, tuple)
                    and len(arg.loc) >= 2
                    and not is_empty_exo
                ):
                    i, s_endo = arg.loc[0], arg.loc[1]
                    s_arr = np.asarray(s_endo)
                    if s_arr.ndim == 1:
                        return self.eval_is(i, s_arr[None, :])[0, :]
                    return self.eval_is(i, s_arr)
                val_arr = np.asarray(arg.val)
                if val_arr.ndim == 1:
                    return self.eval_s(val_arr[None, :])[0, :]
                return self.eval_s(val_arr)

            s_arr = np.asarray(arg)
            if s_arr.ndim == 1:
                return self.eval_s(s_arr[None, :])[0, :]
            return self.eval_s(s_arr)

        elif len(args) == 2:
            a0, a1 = args
            if isinstance(a0, (int, np.integer)):
                s_arr = np.asarray(a1)
                if s_arr.ndim == 1:
                    return self.eval_is(int(a0), s_arr[None, :])[0, :]
                return self.eval_is(int(a0), s_arr)

            a0_arr = np.asarray(a0)
            a1_arr = np.asarray(a1)
            if a0_arr.dtype.kind in ("i", "u") and a0_arr.ndim == 0:
                if a1_arr.ndim == 1:
                    return self.eval_is(int(a0_arr), a1_arr[None, :])[0, :]
                return self.eval_is(int(a0_arr), a1_arr)

            if a0_arr.ndim == 1 and a1_arr.ndim == 1:
                return self.eval_ms(a0_arr[None, :], a1_arr[None, :])[0, :]
            return self.eval_ms(a0_arr, a1_arr)

        raise ValueError(
            f"DecisionRule cannot be evaluated with {len(args)} arguments."
        )


class DecisionRule(CallableDecisionRule):

    exo_grid: Grid
    endo_grid: Grid

    def __init__(
        self,
        exo_grid: Any,
        endo_grid: Any = None,
        interp_method="cubic",
        dprocess=None,
        values=None,
    ):
        from dolo.numeric.spaces import Space, ProductSpace, CartesianSpace
        from dolo.numeric.approximators import Approximator, get_approximator
        from dolo.numeric.grids import ProductGrid, EmptyGrid

        if isinstance(exo_grid, Approximator):
            self.approximator = exo_grid
            if isinstance(self.approximator.grid, ProductGrid):
                self.exo_grid = self.approximator.grid.grids[0]
                self.endo_grid = self.approximator.grid.grids[1]
            else:
                self.exo_grid = EmptyGrid()
                self.endo_grid = self.approximator.grid
            self.interp_method = getattr(
                self.approximator, "interp_method", interp_method
            )
        elif isinstance(exo_grid, Space):
            space = exo_grid
            if isinstance(space, ProductSpace):
                grid = space.discretize()
                self.exo_grid = grid.grids[0]
                self.endo_grid = grid.grids[1]
            elif isinstance(space, CartesianSpace):
                self.exo_grid = EmptyGrid()
                self.endo_grid = space.discretize()
            else:
                self.exo_grid = EmptyGrid()
                self.endo_grid = space.discretize()
            self.interp_method = interp_method
            self.approximator = get_approximator(
                ProductGrid(self.exo_grid, self.endo_grid),
                interp_method=interp_method,
            )
        else:
            self.exo_grid = exo_grid
            self.endo_grid = endo_grid
            self.interp_method = interp_method
            self.approximator = get_approximator(
                ProductGrid(self.exo_grid, self.endo_grid),
                interp_method=interp_method,
            )

        if self.interp_method not in interp_methods.keys():
            raise Exception(
                f"Unknown interpolation type: {self.interp_method}. Try one of: {tuple(interp_methods.keys())}"
            )

        self.dprocess = dprocess
        self.coefficients = None

        if values is not None:
            self.fit(values)

    def fit(self, x):
        """Fit decision rule coefficients from values on the grid.

        Returns self for fluent chaining: `dr = DecisionRule(...).fit(x)`
        """
        from dolo.numeric.garray import GArray

        if isinstance(x, GArray):
            x = x.data
        self.coefficients = self.approximator.fit(x)
        return self

    # Alias for backwards compatibility
    set_values = fit

    def eval_ms(self, m, s):
        """Deprecated: use `dr(m, s)` instead."""
        import warnings

        warnings.warn(
            "dr.eval_ms(m, s) is deprecated, use dr(m, s) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self(m, s)

    def eval_is(self, i, s):
        """Deprecated: use `dr(i, s)` instead."""
        import warnings

        warnings.warn(
            "dr.eval_is(i, s) is deprecated, use dr(i, s) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self(i, s)

    def eval_s(self, s):
        """Deprecated: use `dr(s)` instead."""
        import warnings

        warnings.warn(
            "dr.eval_s(s) is deprecated, use dr(s) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self(s)

    def eval_ijs(self, i, j, s):
        """Evaluate policy at transitioned state for regime transition (i -> j)."""
        if isinstance(self.exo_grid, UnstructuredGrid):
            return self(j, s)
        elif isinstance(self.exo_grid, EmptyGrid):
            return self(s)
        elif isinstance(self.exo_grid, CartesianGrid):
            m = self.dprocess.inode(i, j)[None, :].repeat(s.shape[0], axis=0)
            return self(m, s)
        else:
            raise NotImplementedError()


####


class ConstantDecisionRule(CallableDecisionRule):
    def __init__(self, x0):
        self.x0 = x0

    def eval_s(self, s):
        if s.ndim == 1:
            return self.x0
        else:
            N = s.shape[0]
            return self.x0[None, :].repeat(N, axis=0)

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

        exogenous = model.symbols["exogenous"]
        states = model.symbols["states"]
        controls = model.symbols["controls"]
        parameters = model.symbols["parameters"]

        preamble = dict([(s, values[s]) for s in values.keys() if s not in controls])
        equations = [values[s] for s in controls]

        variables = exogenous + states + controls + [*preamble.keys()]

        preamble_str = dict()

        for k in [*preamble.keys()]:
            v = preamble[k]
            if "(" not in k:
                vv = f"{k}(0)"
            else:
                vv = k

            preamble_str[stringify(vv)] = stringify(sanitize(v, variables))

        # let's reorder the preamble
        from dolang.triangular_solver import get_incidence, triangular_solver

        incidence = get_incidence(preamble_str)
        sol = triangular_solver(incidence)
        kk = [*preamble_str.keys()]
        preamble_str = dict([(kk[k], preamble_str[kk[k]]) for k in sol])

        equations = [
            dolang.symbolic.sanitize(eq, variables=variables) for eq in equations
        ]
        equations_strings = [dolang.stringify(eq) for eq in equations]

        args = dict(
            [
                ("m", [(e, 0) for e in exogenous]),
                ("s", [(e, 0) for e in states]),
                ("p", [e for e in parameters]),
            ]
        )

        args = dict([(k, [stringify_symbol(e) for e in v]) for k, v in args.items()])

        targets = [stringify_symbol((e, 0)) for e in controls]

        eqs = dict([(targets[i], eq) for i, eq in enumerate(equations_strings)])

        fff = FlatFunctionFactory(preamble_str, eqs, args, "custom_dr")

        fun, gufun = make_method_from_factory(fff)

        self.p = model.calibration["parameters"]
        self.exo_grid = model.exogenous.discretize()  # this is never used
        self.endo_grid = model.endo_grid
        self.gufun = gufun

    def eval_ms(self, m, s):

        return self.gufun(m, s, self.p)
