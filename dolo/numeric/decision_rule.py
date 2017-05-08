import numpy
from numpy import array, zeros
from dolo.numeric.grids import cat_grids, n_nodes, node
from dolo.numeric.grids import UnstructuredGrid, CartesianGrid, EmptyGrid
from dolo.numeric.misc import mlinspace

from interpolation.splines.eval_cubic import vec_eval_cubic_splines
from interpolation.splines.filter_cubic import filter_data, filter_mcoeffs

import numpy as np

class ConstantDecisionRule:

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


class DecisionRule:

    def __init__(self, exo_grid, endo_grid, interp_type='cubic', dprocess=None):

        self.exo_grid = exo_grid
        self.endo_grid = endo_grid
        self.interp_type = interp_type
        self.dprocess = dprocess

    @property
    def full_grid(self):
        return cat_grids(self.exo_grid, self.endo_grid)
    #
    # @property
    # def full_grid(self):
    #     return self.full_grid


    def set_values(self, x):

        if isinstance(self.exo_grid, UnstructuredGrid) and isinstance(self.endo_grid, CartesianGrid):
            min = self.endo_grid.min
            max = self.endo_grid.max
            n = self.endo_grid.n
            coeffs = filter_controls(min, max, n, x)
            self.coefficients = coeffs
            self.n_x = x.shape[-1]
        elif isinstance(self.exo_grid, EmptyGrid) and isinstance(self.endo_grid, CartesianGrid):
            full_grid = self.full_grid
            min = full_grid.min
            max = full_grid.max
            n = full_grid.n
            coeffs = filter_mcoeffs(min, max, n, x)
            self.coefficients = coeffs
            self.n_x = x.shape[-1]
        else:
            raise Exception("Not implemented")

    def eval_is(self, i, s, out=None):

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
        else:
            raise Exception("Not Implemented.")

        return out

    def eval_ms(self, m, s, out=None):

        if s.ndim==1 and m.ndim==1:
            return self.eval_ms(m[None,:], s[None,:], out=out)[0,:]

        s = np.atleast_2d(s)
        m = np.atleast_2d(m)
        if s.shape[0] == 1 and m.shape[0]>1:
            s = s.repeat(m.shape[0], axis=0)
        elif m.shape[0] == 1 and s.shape[0]>1:
            m = m.repeat(s.shape[0], axis=0)

        v = np.concatenate([m,s], axis=1)

        self.eval_s(v, out=out)

        raise Exception("Not Implemented.")


    def eval_s(self, s, out=None):

        if s.ndim==1:
            return self.eval_s(s[None,:], out=out)[0,:]

        s = np.atleast_2d(s)

        if isinstance(self.exo_grid, (EmptyGrid, CartesianGrid)) and isinstance(self.endo_grid, CartesianGrid):
            full_grid = self.full_grid
            min = self.full_grid.min
            max = self.full_grid.max
            n = self.full_grid.n
            coeffs = self.coefficients
            vec_eval_cubic_splines(min, max, n, coeffs, s, out)
        else:
            raise Exception("Not Implemented.")

        return out

    def eval_ijs(self, i, j, s, out=None):

        if out is None:
            out = np.zeros((s.shape[0], self.n_x))

        if isinstance(self.exo_grid, UnstructuredGrid) and isinstance(self.endo_grid, CartesianGrid):
            self.eval_is(j, s, out=out)
        elif isinstance(self.exo_grid, EmptyGrid) and isinstance(self.endo_grid, CartesianGrid):
            self.eval_s(s, out=out)
        elif isinstance(self.exo_grid, CartesianGrid) and isinstance(self.endo_grid, CartesianGrid):
            m = self.dprocess.inode(i, j)
            self.eval_ms(m, s, out=out)
        else:
            raise Exception("Not Implemented.")

        return out
