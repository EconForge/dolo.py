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

    def evaluate(self, coeffs: Any, qp: QualifiedPoint) -> np.ndarray:
        """Evaluate approximation at a single QualifiedPoint."""
        coords = np.asarray(qp.val, dtype=float)
        if coords.ndim == 1:
            res = self.evaluate_coords(coeffs, coords[None, :])
            return res[0]
        return self.evaluate_coords(coeffs, coords)

    def evaluate_batch(self, coeffs: Any, batch: QualifiedBatch) -> np.ndarray:
        """Evaluate approximation over a QualifiedBatch / SliceQP."""
        return self.evaluate_coords(coeffs, batch.endo)


class LinearApproximator(Approximator):
    """Piecewise multilinear interpolation on CartesianGrid."""

    def __init__(self, grid: CartesianGrid, space: Optional[CartesianSpace] = None):
        self.grid = grid
        self.space = space

    def fit(self, values: Union[GArray, np.ndarray]) -> np.ndarray:
        data = values.data if isinstance(values, GArray) else np.asarray(values)
        return data.reshape(tuple(self.grid.n) + (-1,)).copy()

    def evaluate_coords(self, coeffs: np.ndarray, coords: np.ndarray) -> np.ndarray:
        from interpolation.splines import eval_linear

        pts = np.asarray(coords, dtype=float)
        if pts.ndim == 1:
            return eval_linear(self.grid.__numba_repr__(), coeffs, pts[None, :])[0]
        return eval_linear(self.grid.__numba_repr__(), coeffs, pts)


class CubicSplineApproximator(Approximator):
    """Prefiltered cubic spline interpolation on CartesianGrid."""

    def __init__(self, grid: CartesianGrid, space: Optional[CartesianSpace] = None):
        self.grid = grid
        self.space = space

    def fit(self, values: Union[GArray, np.ndarray]) -> np.ndarray:
        from interpolation.splines.prefilter_cubic import prefilter_cubic

        data = values.data if isinstance(values, GArray) else np.asarray(values)
        reshaped = data.reshape(tuple(self.grid.n) + (-1,))
        return prefilter_cubic(self.grid.__numba_repr__(), reshaped)

    def evaluate_coords(self, coeffs: np.ndarray, coords: np.ndarray) -> np.ndarray:
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

    def evaluate_coords(self, coeffs: Any, coords: np.ndarray) -> np.ndarray:
        if self.is_empty_exo:
            return self.sub_approximator.evaluate_coords(coeffs, coords)
        elif self.is_cartesian_exo:
            return self.joint_approximator.evaluate_coords(coeffs, coords)
        else:
            raise ValueError(
                "Discrete exogenous states require regime index or QualifiedPoint for evaluation."
            )

    def evaluate(self, coeffs: Any, qp: QualifiedPoint) -> np.ndarray:
        if self.is_empty_exo:
            coords = np.asarray(qp.val, dtype=float)
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
            return self.sub_approximator.evaluate_coords(
                coeffs[i], np.asarray(endo_coords, dtype=float)
            )
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
