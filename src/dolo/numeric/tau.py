"""
Tau transition operator and trembling hand grid projection.

Following Dolo.jl:
- Tau: Generates the conditional transition distribution {(w_k, S_k)} over next states.
- trembling_hand: Multilinear grid projector mapping continuous next states onto grid nodes.
- tau_fit: Transition distribution projected onto discrete grid vertices.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterator, List, Tuple
import numpy as np

from dolo.numeric.garray import GArray, QP, SliceQP
from dolo.numeric.grids import UniformCartesianGrid, CartesianGrid


def trembling_hand_1d(
    min_val: float, max_val: float, n_points: int, points_1d: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1D trembling-hand multilinear decomposition.

    For an array of values `points_1d`, finds the two enclosing grid vertices
    `i0`, `i1` and their linear weights `w0 = 1 - lam`, `w1 = lam`.

    Returns
    -------
    i0 : np.ndarray (int)
        Left vertex index (0 to n_points - 2).
    i1 : np.ndarray (int)
        Right vertex index (1 to n_points - 1).
    w0 : np.ndarray (float)
        Weight for left vertex.
    w1 : np.ndarray (float)
        Weight for right vertex.
    """
    scale = (points_1d - min_val) / (max_val - min_val) * (n_points - 1)
    i0 = np.floor(scale).astype(np.int64)
    i0 = np.clip(i0, 0, n_points - 2)
    i1 = i0 + 1
    lam = np.clip(scale - i0, 0.0, 1.0)
    w0 = 1.0 - lam
    w1 = lam
    return i0, i1, w0, w1


def trembling_hand(
    grid: UniformCartesianGrid, points: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Multilinear projection of continuous points onto Cartesian grid vertices.

    Parameters
    ----------
    grid : UniformCartesianGrid
        The target Cartesian grid with attributes `min`, `max`, `n`.
    points : np.ndarray, shape (N, d)
        Continuous points to project onto the grid.

    Returns
    -------
    List of (weights, indices) tuples for each of the 2^d vertices:
    - weights : np.ndarray, shape (N,)
    - indices : np.ndarray, shape (N,) - linear indices on the Cartesian grid.
    """
    d = grid.d
    N = points.shape[0]

    dim_decomps = []
    for k in range(d):
        min_k = float(grid.min[k])
        max_k = float(grid.max[k])
        n_k = int(grid.n[k])
        i0, i1, w0, w1 = trembling_hand_1d(min_k, max_k, n_k, points[:, k])
        dim_decomps.append([(i0, w0), (i1, w1)])

    # Compute striding for linear indices
    strides = [1]
    for k in range(d - 1, 0, -1):
        strides.insert(0, strides[0] * int(grid.n[k]))

    results = []
    # 2^d vertex combinations
    for corner in itertools.product(*dim_decomps):
        # corner is a tuple of (idx_k, weight_k)
        corner_weight = np.ones(N, dtype=float)
        corner_idx = np.zeros(N, dtype=np.int64)
        for k, (idx_k, w_k) in enumerate(corner):
            corner_weight *= w_k
            corner_idx += idx_k * strides[k]
        results.append((corner_weight, corner_idx))

    return results


class Tau:
    """The Tau transition operator.

    Encapsulates the model's exogenous stochastic process and endogenous
    law of motion into a single conditional distribution operator:
        (s, x) -> {(w_k, S_k)}
    """

    def __init__(self, g_fun: Any, dprocess: Any, parms: np.ndarray):
        self.g = g_fun
        self.dprocess = dprocess
        self.parms = parms

    def slice_transitions(
        self, i_m: int, m_val: np.ndarray, s_endo: np.ndarray, xm: np.ndarray
    ) -> Iterator[Tuple[float, SliceQP]]:
        """Slice-level transition: vectorized across all N points of continuous grid.

        Yields (prob, SliceQP) where SliceQP contains:
        - loc: target discrete regime index
        - exo: next exogenous shock vector
        - endo: next endogenous state matrix (N, n_s)
        """
        N = s_endo.shape[0]
        n_inodes = self.dprocess.n_inodes(i_m)
        m_tiled = np.tile(m_val, (N, 1))

        for I_m in range(n_inodes):
            prob = float(self.dprocess.iweight(i_m, I_m))
            M_val = self.dprocess.inode(i_m, I_m)
            M_tiled = np.tile(M_val, (N, 1))

            S_endo = self.g(m_tiled, s_endo, xm, M_tiled, self.parms)
            target_regime = 0 if self.dprocess.n_nodes == 1 else I_m
            yield prob, SliceQP(loc=target_regime, exo=M_val, endo=S_endo)

    def point_transitions(self, s: QP, x: np.ndarray) -> Iterator[Tuple[float, QP]]:
        """Point-level transition (pure Julia-style).

        Given a single state point QP(loc, val) and control vector x,
        yields (prob, QP(loc_next, val_next)).
        """
        if isinstance(s.loc, tuple):
            i_m, s_endo = s.loc[0], s.loc[1]
        else:
            i_m, s_endo = 0, s.val

        m_val = self.dprocess.node(i_m)
        n_inodes = self.dprocess.n_inodes(i_m)

        m_1d = m_val[None, :]
        s_1d = s_endo[None, :]
        x_1d = x[None, :]

        for I_m in range(n_inodes):
            prob = float(self.dprocess.iweight(i_m, I_m))
            M_val = self.dprocess.inode(i_m, I_m)
            M_1d = M_val[None, :]

            S_endo = self.g(m_1d, s_1d, x_1d, M_1d, self.parms)[0]
            val_next = np.concatenate([M_val, S_endo])
            target_regime = 0 if self.dprocess.n_nodes == 1 else I_m
            yield prob, QP(loc=(target_regime, S_endo), val=val_next)

    def tau_fit(
        self,
        i_m: int,
        m_val: np.ndarray,
        s_endo: np.ndarray,
        xm: np.ndarray,
        endo_grid: UniformCartesianGrid,
    ) -> Iterator[Tuple[int, float, np.ndarray, np.ndarray]]:
        """Projected transitions on the grid vertices via trembling hand.

        For each shock branch and each multilinear vertex corner, yields:
        (target_regime_j, shock_prob, corner_weights, corner_grid_indices)
        where:
        - corner_weights : np.ndarray of shape (N,)
        - corner_grid_indices : np.ndarray of shape (N,) on endo_grid
        """
        for prob, S in self.slice_transitions(i_m, m_val, s_endo, xm):
            target_regime = S.loc
            projections = trembling_hand(endo_grid, S.endo)
            for w_corner, idx_corner in projections:
                yield target_regime, prob, w_corner, idx_corner


def euler_residuals_tau(dmodel: Any, controls: GArray, dr: Any) -> GArray:
    """Universal Euler residuals written entirely in terms of GArray and Tau."""
    res = GArray.zeros_like(controls)
    for i_m, m_val, s_endo in dmodel.regimes():
        xm = controls[i_m]
        for prob, S in dmodel.tau.slice_transitions(i_m, m_val, s_endo, xm):
            XM = dr(S)
            rr = dmodel.arbitrage(m_val, s_endo, xm, S.exo, S.endo, XM)
            res[i_m] += prob * rr
    return res
