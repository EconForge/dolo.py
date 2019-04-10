import numpy
from numpy import array, zeros
from interpolation.smolyak import SmolyakGrid as SmolyakGrid0
from interpolation.smolyak import SmolyakInterp, build_B
from dolo.numeric.grids import cat_grids, n_nodes, node
from dolo.numeric.grids import UnstructuredGrid, CartesianGrid, SmolyakGrid, EmptyGrid
from dolo.numeric.misc import mlinspace
import scipy

import numpy as np

class CallableDecisionRule:

    def __call__(self, *args):
        args = [np.array(e) for e in args]
        if len(args)==1:
            return self.eval_s(args[0])
        elif len(args)==2:
            if args[0].dtype in ('int64','int32'):
                return self.eval_is(args[0],args[1])
            else:
                return self.eval_ms(args[0],args[1])

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


class DecisionRule(CallableDecisionRule):

    def __init__(self, exo_grid, endo_grid, interp_type='cubic', dprocess=None):

        self.exo_grid = exo_grid
        self.endo_grid = endo_grid
        self.interp_type = interp_type
        self.dprocess = dprocess

        if isinstance(self.exo_grid, (UnstructuredGrid, EmptyGrid)) and isinstance(self.endo_grid, SmolyakGrid):
            min = self.endo_grid.min
            max = self.endo_grid.max
            d = len(min)
            mu = self.endo_grid.mu
            sg = SmolyakGrid0(d, mu, lb=min, ub=max) # That is really strange
            sg.lu_fact = scipy.linalg.lu_factor(sg.B)
            self.sg = sg
            self.interp_type = 'chebychev'
        else:
            self.interp_type = 'cubic'

    @property
    def full_grid(self):
        return cat_grids(self.exo_grid, self.endo_grid)

    def set_values(self, x):

        from interpolation.splines.filter_cubic import filter_data, filter_mcoeffs

        x = np.array(x)

        if isinstance(self.exo_grid, UnstructuredGrid) and isinstance(self.endo_grid, CartesianGrid):
            min = self.endo_grid.min
            max = self.endo_grid.max
            n = self.endo_grid.n
            coeffs = filter_controls(min, max, n, x)
            self.coefficients = coeffs
        elif isinstance(self.exo_grid, UnstructuredGrid) and isinstance(self.endo_grid, SmolyakGrid):
            from scipy.linalg import lu_solve
            from numpy import linalg
            self.thetas = [lu_solve( self.sg.lu_fact, v ) for v in x]
        elif isinstance(self.exo_grid, EmptyGrid) and isinstance(self.endo_grid, SmolyakGrid):
            from scipy.linalg import lu_solve
            from numpy import linalg
            self.thetas = lu_solve( self.sg.lu_fact, x[0] )
        elif isinstance(self.exo_grid, EmptyGrid) and isinstance(self.endo_grid, CartesianGrid):
            full_grid = self.full_grid
            min = full_grid.min
            max = full_grid.max
            n = full_grid.n
            coeffs = filter_mcoeffs(min, max, n, x)
            self.coefficients = coeffs
        else:
            raise Exception("Not implemented")
        self.n_x = x.shape[-1]

    def eval_is(self, i, s, out=None):

        from interpolation.splines.eval_cubic import vec_eval_cubic_splines

        if s.ndim == 1:
            return self.eval_is(i, s[None,:], out=out)[0,:]

        s = np.atleast_2d(s)

        if out is None:
            N = s.shape[0]
            out = zeros((N, self.n_x))

        if isinstance(self.exo_grid, UnstructuredGrid) and isinstance(self.endo_grid, CartesianGrid):
            coeffs = self.coefficients[i]
            min = self.endo_grid.min
            max = self.endo_grid.max
            n = self.endo_grid.n
            vec_eval_cubic_splines(min, max, n, coeffs, s, out)
        elif isinstance(self.exo_grid, EmptyGrid) and isinstance(self.endo_grid, CartesianGrid):
            self.eval_s(s, out=out)
        elif isinstance(self.exo_grid, UnstructuredGrid) and isinstance(self.endo_grid, SmolyakGrid):
            trans_points = self.sg.dom2cube(s)
            Phi = build_B(self.sg.d, self.sg.mu, trans_points, self.sg.pinds)
            theta = self.thetas[i]
            out[...] =  Phi@theta
        else:
            raise Exception("Not Implemented.")

        return out

    def eval_ms(self, m, s, out=None):

        from interpolation.splines.eval_cubic import vec_eval_cubic_splines

        if s.ndim==1 and m.ndim==1:
            return self.eval_ms(m[None,:], s[None,:], out=out)[0,:]

        s = np.atleast_2d(s)
        m = np.atleast_2d(m)
        if s.shape[0] == 1 and m.shape[0]>1:
            s = s.repeat(m.shape[0], axis=0)
        elif m.shape[0] == 1 and s.shape[0]>1:
            m = m.repeat(s.shape[0], axis=0)


        if out is None:
            out = np.zeros((s.shape[0], self.n_x))

        if isinstance(self.exo_grid, (EmptyGrid)) and isinstance(self.endo_grid, CartesianGrid):
            self.eval_s(s, out=out)
        elif isinstance(self.exo_grid, CartesianGrid) and isinstance(self.endo_grid, CartesianGrid):
            v = np.concatenate([m,s], axis=1)
            full_grid = self.full_grid
            min = full_grid.min
            max = full_grid.max
            n = full_grid.n
            coeffs = self.coefficients
            vec_eval_cubic_splines(min, max, n, coeffs, v, out)
        else:
            raise Exception("Not Implemented.")

        return out
        # raise Exception("Not Implemented.")


    def eval_s(self, s, out=None):

        from interpolation.splines.eval_cubic import vec_eval_cubic_splines

        if s.ndim==1:
            return self.eval_s(s[None,:])[0,:]

        s = np.atleast_2d(s)
        if out is None:
            out = np.zeros( (s.shape[0], self.n_x) )

        if isinstance(self.exo_grid, (EmptyGrid, CartesianGrid)) and isinstance(self.endo_grid, CartesianGrid):
            full_grid = self.full_grid
            min = self.full_grid.min
            max = self.full_grid.max
            n = self.full_grid.n
            coeffs = self.coefficients
            vec_eval_cubic_splines(min, max, n, coeffs, s, out)
        elif isinstance(self.exo_grid, EmptyGrid) and isinstance(self.endo_grid, SmolyakGrid):
            trans_points = self.sg.dom2cube(s)
            Phi = build_B(self.sg.d, self.sg.mu, trans_points, self.sg.pinds)
            theta = self.thetas
            out[...] = Phi@theta
        else:
            raise Exception("Not Implemented.")

        return out

    def eval_ijs(self, i, j, s, out=None):

        if out is None:
            out = np.zeros((s.shape[0], self.n_x))

        if isinstance(self.exo_grid, UnstructuredGrid) and isinstance(self.endo_grid, (CartesianGrid, SmolyakGrid)):
            self.eval_is(j, s, out=out)
        elif isinstance(self.exo_grid, EmptyGrid) and isinstance(self.endo_grid, (CartesianGrid,SmolyakGrid)):
            self.eval_s(s, out=out)
        elif isinstance(self.exo_grid, CartesianGrid) and isinstance(self.endo_grid, CartesianGrid):
            m = self.dprocess.inode(i, j)
            self.eval_ms(m, s, out=out)
        else:
            raise Exception("Not Implemented.")

        return out


import dolang
import dolang.symbolic
from dolang.symbolic import stringify_symbol
from dolo.numeric.decision_rule import CallableDecisionRule
from dolang.factory import FlatFunctionFactory
from dolang.function_compiler import make_method_from_factory

class CustomDR(CallableDecisionRule):

    def __init__(self, values, model=None):

        exogenous = model.symbols['exogenous']
        states = model.symbols['states']
        controls = model.symbols['controls']
        parameters = model.symbols['parameters']

        equations = [values[s] for s in controls]

        variables = exogenous + states + controls

        preamble = dict()

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

        fff = FlatFunctionFactory(preamble, eqs, args, 'custom_dr')

        fun, gufun = make_method_from_factory(fff)

        self.p = model.calibration['parameters']
        self.exo_grid = model.exogenous.discretize() # this is never used
        self.endo_grid = model.get_grid()
        self.gufun = gufun

    def eval_ms(self, m, s):

        return self.gufun(m, s, self.p)
