"""
DiscretizedModel: The primary representation of a discretized economic model.

Following Dolo.jl:
A DiscretizedModel bundles:
- The continuous/symbolic Model
- The spatial Grid
- The transition operator Tau
- Methods for initial guess, bounds, arbitrage, and transition matrix generation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Tuple
import numpy as np

from dolo.numeric.garray import GArray, QP, SliceQP
from dolo.numeric.grids import ProductGrid, EmptyGrid, UniformCartesianGrid
from dolo.numeric.tau import Tau


class DiscretizedModel:
    """Discretized economic model.

    Encapsulates:
    - `model`: Symbolic Model / ModelV2
    - `grid`: Spatial Grid (ProductGrid or CartesianGrid)
    - `tau`: Tau transition operator
    - `dprocess`: Underlying discretized process
    """

    def __init__(
        self,
        model: Any,
        grid_options: Optional[Dict[str, Any]] = None,
        dprocess_options: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        dprocess_options = dprocess_options or {}

        # Discretize model domain and process
        self.grid, self.dprocess = model.discretize(grid_options, dprocess_options)
        self.parms = model.calibration["parameters"]
        self.f = model.functions["arbitrage"]
        self.g = model.functions["transition"]

        # Tau transition operator
        self.tau = Tau(self.g, self.dprocess, self.parms)

        # Grid characteristics
        self.is_mc = not isinstance(self.grid.grids[0], EmptyGrid)
        self.n_regimes = self.dprocess.n_nodes
        self.endo_grid = self.grid.grids[1]
        self.endo_nodes = self.endo_grid.nodes  # shape (N, n_s)
        self.N_endo = self.endo_nodes.shape[0]
        self.N_total = self.n_regimes * self.N_endo

    def regimes(self) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """Yields (regime_index, m_value, endo_nodes) for each discrete regime."""
        for i in range(self.n_regimes):
            m_val = self.dprocess.node(i)
            yield i, m_val, self.endo_nodes

    def arbitrage(
        self,
        m: np.ndarray,
        s: np.ndarray,
        x: np.ndarray,
        M: np.ndarray,
        S: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        """Arbitrage equation evaluation wrapper."""
        N = s.shape[0] if s.ndim == 2 else 1
        m_tiled = np.tile(m, (N, 1)) if m.ndim == 1 and N > 1 else m
        M_tiled = np.tile(M, (N, 1)) if M.ndim == 1 and N > 1 else M
        return self.f(m_tiled, s, x, M_tiled, S, X, self.parms)

    def initial_guess(self) -> GArray:
        """Initial GArray of controls from model calibration."""
        x0 = self.model.calibration["controls"]
        n_x = len(x0)
        controls = GArray.zeros(self.grid, n_x, self.n_regimes)
        controls.data[:, :, :] = x0[None, None, :]
        return controls

    def lower_bounds(self) -> Optional[GArray]:
        """Lower bounds on controls as GArray, if specified."""
        if "arbitrage_lb" not in self.model.functions:
            return None
        lb_fun = self.model.functions["arbitrage_lb"]
        controls_lb = GArray.zeros(
            self.grid, len(self.model.symbols["controls"]), self.n_regimes
        )
        N = self.N_endo
        p = np.repeat(self.parms[None, :], N, axis=0)

        for i, m_val, s_endo in self.regimes():
            m = np.repeat(m_val[None, :], N, axis=0)
            controls_lb[i] = lb_fun(m, s_endo, p)
        return controls_lb

    def upper_bounds(self) -> Optional[GArray]:
        """Upper bounds on controls as GArray, if specified."""
        if "arbitrage_ub" not in self.model.functions:
            return None
        ub_fun = self.model.functions["arbitrage_ub"]
        controls_ub = GArray.zeros(
            self.grid, len(self.model.symbols["controls"]), self.n_regimes
        )
        N = self.N_endo
        p = np.repeat(self.parms[None, :], N, axis=0)

        for i, m_val, s_endo in self.regimes():
            m = np.repeat(m_val[None, :], N, axis=0)
            controls_ub[i] = ub_fun(m, s_endo, p)
        return controls_ub

    def transition_matrix(self, controls: GArray) -> np.ndarray:
        """Compute the full Markov transition matrix P on the grid nodes using tau_fit.

        Parameters
        ----------
        controls : GArray
            Policy controls defined on the grid.

        Returns
        -------
        P : np.ndarray, shape (N_total, N_total)
            Stochastic matrix where P[i, j] is the transition probability from node i to node j.
        """
        P = np.zeros((self.N_total, self.N_total), dtype=float)
        N_e = self.N_endo

        for i_m, m_val, s_endo in self.regimes():
            xm = controls[i_m]
            row_offset = i_m * N_e

            for j_target, prob, w_corner, idx_corner in self.tau.tau_fit(
                i_m, m_val, s_endo, xm, self.endo_grid
            ):
                col_offset = j_target * N_e
                for n in range(N_e):
                    row = row_offset + n
                    col = col_offset + idx_corner[n]
                    P[row, col] += prob * w_corner[n]

        return P

    def ergodic_distribution(self, controls: GArray) -> GArray:
        """Compute the ergodic stationary distribution on the grid nodes.

        Parameters
        ----------
        controls : GArray
            Policy controls defined on the grid.

        Returns
        -------
        mu : GArray
            Stationary probability distribution across all grid nodes.
        """
        P = self.transition_matrix(controls)
        # Solve mu P = mu  <=>  (P^T - I) mu = 0 with sum(mu) = 1
        A = P.T - np.eye(self.N_total)
        A[-1, :] = 1.0
        b = np.zeros(self.N_total)
        b[-1] = 1.0
        mu_flat = np.linalg.solve(A, b)
        mu_flat = np.maximum(mu_flat, 0.0)
        mu_flat /= mu_flat.sum()

        mu = GArray.zeros(self.grid, 1, self.n_regimes)
        return mu.unravel(mu_flat)
