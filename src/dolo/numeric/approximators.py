"""
Approximator hierarchy for dolo.py.

Following Dolo.jl:
- Approximator: Operator knowing how to fit coefficients from values on a Grid,
  and evaluate approximations at query points (QualifiedPoint, QualifiedBatch, or coords).
- LinearApproximator: Piecewise multilinear interpolation on CartesianGrid.
- CubicSplineApproximator: Prefiltered cubic spline interpolation on CartesianGrid.
- SmolyakApproximator: Multivariate Chebyshev polynomial basis on SmolyakGrid.
- ProductApproximator: Regime-wise composite approximator over ProductGrid / ProductSpace.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Union
import numpy as np

from dolo.numeric.garray import GArray, QualifiedPoint, QP, QualifiedBatch, SliceQP
from dolo.numeric.grids import (
    Grid,
    CartesianGrid,
    UniformCartesianGrid,
    UnstructuredGrid,
    ProductGrid,
    EmptyGrid,
    SmolyakGrid,
)
from dolo.numeric.spaces import Space, CartesianSpace, ProductSpace, EmptySpace


class Approximator(ABC):
    """Abstract base class for approximation schemes."""

    grid: Grid
    space: Optional[Space]

    @abstractmethod
    def fit(self, values: Union[GArray, np.ndarray]) -> Any:
        """Compute and return approximation coefficients from field values on the grid."""
        pass

    @abstractmethod
    def evaluate_coords(self, coeffs: Any, coords: np.ndarray) -> np.ndarray:
        """Evaluate approximation at continuous coordinate matrix of shape (N, d)."""
        pass

    def evaluate(self, coeffs: Any, *args) -> np.ndarray:
        """Universal evaluation method for approximation schemes.

        Supports:
        - `evaluate(coeffs, qp)`: single QualifiedPoint / QP
        - `evaluate(coeffs, batch)`: QualifiedBatch / SliceQP
        - `evaluate(coeffs, coords)`: coordinate array (N, d) or (d,)
        - `evaluate(coeffs, *args)`: multi-argument forms implemented by subclasses
        """
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, QualifiedBatch):
                return self.evaluate_batch(coeffs, arg)
            if isinstance(arg, QualifiedPoint):
                coords = np.asarray(arg.val, dtype=float)
                if coords.ndim == 1:
                    return self.evaluate_coords(coeffs, coords[None, :])[0]
                return self.evaluate_coords(coeffs, coords)
            coords = np.asarray(arg, dtype=float)
            if coords.ndim == 1:
                return self.evaluate_coords(coeffs, coords[None, :])[0]
            return self.evaluate_coords(coeffs, coords)
        raise ValueError(f"Approximator.evaluate cannot handle {len(args)} arguments.")

    def evaluate_batch(self, coeffs: Any, batch: QualifiedBatch) -> np.ndarray:
        """Evaluate approximation over a QualifiedBatch / SliceQP."""
        return self.evaluate_coords(coeffs, batch.endo)


class LinearApproximator(Approximator):
    """Piecewise multilinear interpolation on CartesianGrid."""

    def __init__(
        self,
        grid: CartesianGrid,
        space: Optional[CartesianSpace] = None,
        backend: str = "jax",
    ):
        self.grid = grid
        self.space = space
        self.backend = backend

    def fit(self, values: Union[GArray, np.ndarray]) -> np.ndarray:
        data = values.data if isinstance(values, GArray) else np.asarray(values)
        return data.reshape(tuple(self.grid.n) + (-1,)).copy()

    def evaluate_coords(self, coeffs: np.ndarray, coords: np.ndarray) -> np.ndarray:
        if self.backend == "jax":
            from dolo.numeric.interpolation_jax import eval_linear_jax

            return np.asarray(eval_linear_jax(self.grid, coeffs, coords))

        from interpolation.splines import eval_linear

        pts = np.asarray(coords, dtype=float)
        if pts.ndim == 1:
            return eval_linear(self.grid.__numba_repr__(), coeffs, pts[None, :])[0]
        return eval_linear(self.grid.__numba_repr__(), coeffs, pts)


class CubicSplineApproximator(Approximator):
    """Prefiltered cubic spline interpolation on CartesianGrid."""

    def __init__(
        self,
        grid: CartesianGrid,
        space: Optional[CartesianSpace] = None,
        backend: str = "jax",
    ):
        self.grid = grid
        self.space = space
        self.backend = backend

    def fit(self, values: Union[GArray, np.ndarray]) -> np.ndarray:
        data = values.data if isinstance(values, GArray) else np.asarray(values)
        reshaped = data.reshape(tuple(self.grid.n) + (-1,))
        if self.backend == "jax":
            from dolo.numeric.interpolation_jax import prefilter_cubic_jax

            return np.asarray(prefilter_cubic_jax(self.grid, reshaped))

        from interpolation.splines.prefilter_cubic import prefilter_cubic

        return prefilter_cubic(self.grid.__numba_repr__(), reshaped)

    def evaluate_coords(self, coeffs: np.ndarray, coords: np.ndarray) -> np.ndarray:
        if self.backend == "jax":
            from dolo.numeric.interpolation_jax import eval_cubic_jax

            return np.asarray(eval_cubic_jax(self.grid, coeffs, coords))

        from interpolation.splines import eval_cubic

        pts = np.asarray(coords, dtype=float)
        if pts.ndim == 1:
            return eval_cubic(self.grid.__numba_repr__(), coeffs, pts[None, :])[0]
        return eval_cubic(self.grid.__numba_repr__(), coeffs, pts)


class SmolyakApproximator(Approximator):
    """Multivariate Chebyshev polynomial approximation on SmolyakGrid."""

    def __init__(self, grid: SmolyakGrid, space: Optional[Space] = None):
        self.grid = grid
        self.space = space

    def fit(self, values: Union[GArray, np.ndarray]) -> np.ndarray:
        data = values.data if isinstance(values, GArray) else np.asarray(values)
        from interpolation.smolyak import build_B

        B = build_B(self.grid.sg.grid, self.grid.sg.B)
        return np.linalg.solve(B, data)

    def evaluate_coords(self, coeffs: np.ndarray, coords: np.ndarray) -> np.ndarray:
        from interpolation.smolyak import SmolyakInterp

        si = SmolyakInterp(self.grid.sg, coeffs)
        pts = np.asarray(coords, dtype=float)
        if pts.ndim == 1:
            return si(pts[None, :])[0]
        return si(pts)


class ProductApproximator(Approximator):
    """Composite approximator over ProductGrid (e.g. exo × endo)."""

    def __init__(
        self,
        grid: ProductGrid,
        interp_method: str = "cubic",
        space: Optional[ProductSpace] = None,
    ):
        self.grid = grid
        self.space = space
        self.interp_method = interp_method
        self.exo_grid = grid.grids[0]
        self.endo_grid = grid.grids[1]

        # Build sub-approximator for continuous endogenous grid
        if isinstance(self.endo_grid, CartesianGrid):
            if interp_method in ("cubic", "spline"):
                self.sub_approximator = CubicSplineApproximator(self.endo_grid)
            elif interp_method in ("linear", "multilinear"):
                self.sub_approximator = LinearApproximator(self.endo_grid)
            else:
                raise ValueError(
                    f"Unsupported continuous interp_method: {interp_method}"
                )
        elif isinstance(self.endo_grid, SmolyakGrid):
            self.sub_approximator = SmolyakApproximator(self.endo_grid)
        else:
            raise ValueError(f"Unsupported endogenous grid: {type(self.endo_grid)}")

        # Check exogenous structure
        self.is_empty_exo = isinstance(self.exo_grid, EmptyGrid)
        self.is_discrete_exo = isinstance(self.exo_grid, UnstructuredGrid)
        self.is_cartesian_exo = isinstance(self.exo_grid, CartesianGrid)

        if self.is_cartesian_exo:
            joint_grid = self.exo_grid + self.endo_grid
            if interp_method in ("cubic", "spline"):
                self.joint_approximator = CubicSplineApproximator(joint_grid)
            else:
                self.joint_approximator = LinearApproximator(joint_grid)
        else:
            self.joint_approximator = None

    def fit(self, values: Union[GArray, np.ndarray]) -> Any:
        data = values.data if isinstance(values, GArray) else np.asarray(values)
        if self.is_empty_exo:
            if data.ndim == 3 and data.shape[0] == 1:
                data = data[0]
            return self.sub_approximator.fit(data)
        elif self.is_discrete_exo:
            n_regimes = data.shape[0]
            return [self.sub_approximator.fit(data[i]) for i in range(n_regimes)]
        elif self.is_cartesian_exo:
            return self.joint_approximator.fit(data)
        else:
            raise NotImplementedError(
                f"Unsupported exo grid type: {type(self.exo_grid)}"
            )

    def evaluate_s(self, coeffs: Any, s: np.ndarray) -> np.ndarray:
        """Evaluate continuous endogenous states `s` when exogenous state is empty or single-regime."""
        s_arr = np.asarray(s, dtype=float)
        if self.is_empty_exo:
            return self.sub_approximator.evaluate_coords(coeffs, s_arr)
        if self.is_discrete_exo and self.exo_grid.n_nodes == 1:
            return self.sub_approximator.evaluate_coords(coeffs[0], s_arr)
        raise ValueError(
            "Cannot evaluate 's' alone when exogenous states are non-empty."
        )

    def evaluate_ms(self, coeffs: Any, m: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Evaluate continuous endogenous states `s` and exogenous states `m`."""
        m_arr = np.asarray(m, dtype=float)
        s_arr = np.asarray(s, dtype=float)
        if self.is_empty_exo:
            return self.sub_approximator.evaluate_coords(coeffs, s_arr)
        elif self.is_cartesian_exo:
            combined = np.hstack([m_arr, s_arr])
            return self.joint_approximator.evaluate_coords(coeffs, combined)
        elif self.is_discrete_exo:
            if m_arr.ndim == 1:
                m_arr = m_arr[None, :]
            if s_arr.ndim == 1:
                s_arr = s_arr[None, :]
            if m_arr.shape[0] == 1 and s_arr.shape[0] > 1:
                m_arr = np.repeat(m_arr, s_arr.shape[0], axis=0)
            diffs = np.linalg.norm(
                self.exo_grid.nodes[:, None, :] - m_arr[None, :, :], axis=-1
            )
            regimes = np.argmin(diffs, axis=0)
            n_x = coeffs[0].shape[-1] if hasattr(coeffs[0], "shape") else 1
            out = np.empty((s_arr.shape[0], n_x), dtype=float)
            for r in np.unique(regimes):
                mask = regimes == r
                out[mask] = self.sub_approximator.evaluate_coords(
                    coeffs[r], s_arr[mask]
                )
            return out
        else:
            raise NotImplementedError()

    def evaluate_coords(self, coeffs: Any, coords: np.ndarray) -> np.ndarray:
        if self.is_empty_exo:
            return self.sub_approximator.evaluate_coords(coeffs, coords)
        elif self.is_cartesian_exo:
            return self.joint_approximator.evaluate_coords(coeffs, coords)
        else:
            raise ValueError(
                "Discrete exogenous states require regime index or QualifiedPoint for evaluation."
            )

    def evaluate(self, coeffs: Any, *args) -> np.ndarray:
        """Universal evaluation on ProductApproximator.

        Supports:
        - `evaluate(coeffs, qp)`: single QualifiedPoint / QP
        - `evaluate(coeffs, batch)`: QualifiedBatch / SliceQP
        - `evaluate(coeffs, s)`: continuous endogenous states alone (empty exo or single-regime)
        - `evaluate(coeffs, i, s)`: discrete regime index `i` + continuous states `s`
        - `evaluate(coeffs, m, s)`: continuous exogenous `m` + endogenous `s`
        """
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, QualifiedBatch):
                return self.evaluate_batch(coeffs, arg)
            if isinstance(arg, QualifiedPoint):
                return self._evaluate_qp(coeffs, arg)
            # Coordinates array: continuous endogenous states alone
            return self.evaluate_s(coeffs, np.asarray(arg, dtype=float))

        elif len(args) == 2:
            a0, a1 = args
            # Check for discrete regime index `i`
            if isinstance(a0, (int, np.integer)):
                s_arr = np.asarray(a1, dtype=float)
                if s_arr.ndim == 1:
                    return self.evaluate_regime(coeffs, int(a0), s_arr[None, :])[0]
                return self.evaluate_regime(coeffs, int(a0), s_arr)

            a0_arr = np.asarray(a0)
            a1_arr = np.asarray(a1, dtype=float)
            if a0_arr.dtype.kind in ("i", "u") and a0_arr.ndim == 0:
                if a1_arr.ndim == 1:
                    return self.evaluate_regime(coeffs, int(a0_arr), a1_arr[None, :])[0]
                return self.evaluate_regime(coeffs, int(a0_arr), a1_arr)

            # Continuous (m, s)
            if a0_arr.ndim == 1 and a1_arr.ndim == 1:
                return self.evaluate_ms(coeffs, a0_arr[None, :], a1_arr[None, :])[0]
            return self.evaluate_ms(coeffs, a0_arr, a1_arr)

        raise ValueError(
            f"ProductApproximator.evaluate cannot handle {len(args)} arguments."
        )

    def _evaluate_qp(self, coeffs: Any, qp: QualifiedPoint) -> np.ndarray:
        if self.is_empty_exo:
            coords = np.asarray(qp.val, dtype=float)
            if coords.ndim == 1:
                return self.sub_approximator.evaluate_coords(coeffs, coords[None, :])[0]
            return self.sub_approximator.evaluate_coords(coeffs, coords)
        elif self.is_discrete_exo:
            i = qp.index
            endo_coords = (
                qp.val[self.exo_grid.d :]
                if qp.val.ndim == 1 and qp.val.shape[0] > self.endo_grid.d
                else qp.val
            )
            if isinstance(qp.loc, tuple) and len(qp.loc) >= 2:
                endo_coords = qp.loc[1]
            c_arr = np.asarray(endo_coords, dtype=float)
            if c_arr.ndim == 1:
                return self.sub_approximator.evaluate_coords(coeffs[i], c_arr[None, :])[
                    0
                ]
            return self.sub_approximator.evaluate_coords(coeffs[i], c_arr)
        elif self.is_cartesian_exo:
            return self.joint_approximator.evaluate(coeffs, qp)
        else:
            raise NotImplementedError()

    def evaluate_batch(self, coeffs: Any, batch: QualifiedBatch) -> np.ndarray:
        if self.is_empty_exo:
            return self.sub_approximator.evaluate_coords(coeffs, batch.endo)
        elif self.is_discrete_exo:
            i = batch.loc
            return self.sub_approximator.evaluate_coords(coeffs[i], batch.endo)
        elif self.is_cartesian_exo:
            combined = np.hstack([batch.exo, batch.endo])
            return self.joint_approximator.evaluate_coords(coeffs, combined)
        else:
            raise NotImplementedError()

    def evaluate_regime(self, coeffs: Any, i: int, s: np.ndarray) -> np.ndarray:
        """Evaluate continuous endogenous states `s` within discrete regime `i`."""
        if self.is_empty_exo:
            return self.sub_approximator.evaluate_coords(coeffs, s)
        elif self.is_discrete_exo:
            return self.sub_approximator.evaluate_coords(coeffs[i], s)
        elif self.is_cartesian_exo:
            m = np.tile(self.exo_grid.node(i), (s.shape[0], 1))
            combined = np.hstack([m, s])
            return self.joint_approximator.evaluate_coords(coeffs, combined)
        else:
            raise NotImplementedError()


def get_approximator(
    target: Union[Grid, Space, ProductGrid, ProductSpace],
    interp_method: str = "cubic",
    **kwargs,
) -> Approximator:
    """Factory function returning the canonical Approximator for a given Grid or Space."""
    grid = target.discretize(**kwargs) if isinstance(target, Space) else target
    space = target if isinstance(target, Space) else None

    if isinstance(grid, ProductGrid):
        return ProductApproximator(grid, interp_method=interp_method, space=space)
    elif isinstance(grid, CartesianGrid):
        if interp_method in ("cubic", "spline"):
            return CubicSplineApproximator(grid, space=space)
        elif interp_method in ("linear", "multilinear"):
            return LinearApproximator(grid, space=space)
        else:
            raise ValueError(
                f"Unknown interp_method for CartesianGrid: {interp_method}"
            )
    elif isinstance(grid, SmolyakGrid):
        return SmolyakApproximator(grid, space=space)
    else:
        raise ValueError(f"Cannot construct Approximator for grid type: {type(grid)}")
