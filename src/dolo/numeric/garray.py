"""
GArray and QP abstractions for dolo.py.

Following Dolo.jl:
- QP: Quasi-Point separating discrete grid locators from continuous economic coordinates.
- SliceQP: A batch/slice of state points for an entire continuous sub-grid.
- GArray: A discrete field bundling a Grid with underlying Data, supporting both
  geometric indexing and a flat 1D view for numerical solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np

from dolo.numeric.grids import Grid, ProductGrid, EmptyGrid


@dataclass
class QualifiedPoint:
    """A point in a mathematical space qualified with locator context and evaluated values.

    A QualifiedPoint bundles:
    - `loc` (Locator / Address): How the point is addressed or indexed in the
      computational domain. This can be discrete (e.g. a regime index `i` or grid
      coordinate `(i, j)` for table lookups) or continuous / differentiable
      (e.g. normalized coordinates for spline evaluation).
    - `val` (Values / Payload): The evaluated economic coordinates and derived quantities
      (e.g. state vector `[z, k]`, coordinate products, or precomputed basis features)
      passed to economic equations (arbitrage, transition, utility).

    Attributes
    ----------
    loc : Any
        Computational locator / index.
    val : np.ndarray
        Evaluated economic coordinates / values.
    """

    loc: Any
    val: np.ndarray

    @property
    def coords(self) -> np.ndarray:
        """Alias for economic coordinates `val`."""
        return self.val

    @property
    def index(self) -> Any:
        """Discrete index, if `loc` is a tuple or scalar index."""
        if isinstance(self.loc, tuple) and len(self.loc) > 0:
            return self.loc[0]
        return self.loc

    def __repr__(self) -> str:
        return f"QualifiedPoint(loc={self.loc}, val={np.round(self.val, 4)})"


# Conciseness alias matching Dolo.jl
QP = QualifiedPoint


@dataclass
class QualifiedBatch:
    """A batch/slice of qualified state points for an entire continuous sub-grid.

    Attributes
    ----------
    loc : Any
        Target discrete regime index / locator (or None for pure continuous models).
    exo : np.ndarray
        Exogenous state values for this branch, shape (N, n_m) or (n_m,).
    endo : np.ndarray
        Endogenous state matrix, shape (N, n_s).
    """

    loc: Any
    exo: np.ndarray
    endo: np.ndarray

    @property
    def val(self) -> np.ndarray:
        """Full economic state matrix [exo, endo] of shape (N, n_m + n_s)."""
        if self.exo.ndim == 1:
            exo_tiled = np.tile(self.exo, (self.endo.shape[0], 1))
            return np.hstack([exo_tiled, self.endo])
        return np.hstack([self.exo, self.endo])

    @property
    def coords(self) -> np.ndarray:
        """Full economic coordinates matrix."""
        return self.val


# Aliases for convenience and backward compatibility
QualifiedSlice = QualifiedBatch
SliceQP = QualifiedBatch


class GArray:
    """Represents a discrete field of values living on a Grid.

    Supports:
    - Geometric indexing: garray[loc] accessing a node or slice.
    - Duality: flat 1D view via ravel() / unravel() for solvers.
    - Standard array arithmetic and norm.
    """

    def __init__(self, grid: Any, data: np.ndarray):
        self.grid = grid
        self.data = np.asarray(data)

    @classmethod
    def zeros(cls, grid: Any, n_vars: int, n_regimes: Optional[int] = None) -> GArray:
        """Create a zero GArray conforming to grid structure."""
        if isinstance(grid, ProductGrid) and not isinstance(grid.grids[0], EmptyGrid):
            n_m = grid.grids[0].n_nodes
            n_s = grid.grids[1].n_nodes
            data = np.zeros((n_m, n_s, n_vars))
        elif isinstance(grid, ProductGrid) and isinstance(grid.grids[0], EmptyGrid):
            n_s = grid.grids[1].n_nodes
            data = np.zeros((1, n_s, n_vars))
        elif hasattr(grid, "n_nodes"):
            n_s = grid.n_nodes
            data = np.zeros((1, n_s, n_vars))
        else:
            n_reg = n_regimes if n_regimes is not None else 1
            data = np.zeros((n_reg, len(grid), n_vars))
        return cls(grid, data)

    @classmethod
    def zeros_like(cls, other: GArray) -> GArray:
        """Create a zero GArray matching the grid and shape of another GArray."""
        return cls(other.grid, np.zeros_like(other.data))

    @classmethod
    def ones_like(cls, other: GArray) -> GArray:
        """Create a ones GArray matching the grid and shape of another GArray."""
        return cls(other.grid, np.ones_like(other.data))

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __getitem__(self, loc: Any) -> np.ndarray:
        if isinstance(loc, tuple):
            if len(loc) == 1:
                return self.data[loc[0]]
            return self.data[loc]
        return self.data[loc]

    def __setitem__(self, loc: Any, value: Any) -> None:
        if isinstance(loc, tuple):
            if len(loc) == 1:
                self.data[loc[0]] = value
            else:
                self.data[loc] = value
        else:
            self.data[loc] = value

    def ravel(self) -> np.ndarray:
        """Flat 1D view of the degrees of freedom for numerical solvers."""
        return self.data.ravel()

    def unravel(self, flat_data: np.ndarray) -> GArray:
        """Wrap flat solver iterate back into a structured GArray."""
        return GArray(self.grid, np.asarray(flat_data).reshape(self.data.shape))

    def norm(self) -> float:
        """Maximum absolute norm (L_infinity) across all nodes and variables."""
        return float(np.max(np.abs(self.data)))

    def copy(self) -> GArray:
        return GArray(self.grid, self.data.copy())

    def __sub__(self, other: Union[GArray, np.ndarray, float]) -> GArray:
        other_data = other.data if isinstance(other, GArray) else other
        return GArray(self.grid, self.data - other_data)

    def __add__(self, other: Union[GArray, np.ndarray, float]) -> GArray:
        other_data = other.data if isinstance(other, GArray) else other
        return GArray(self.grid, self.data + other_data)

    def __mul__(self, other: Union[GArray, np.ndarray, float]) -> GArray:
        other_data = other.data if isinstance(other, GArray) else other
        return GArray(self.grid, self.data * other_data)

    def __rmul__(self, scalar: float) -> GArray:
        return GArray(self.grid, self.data * scalar)

    def __truediv__(self, other: Union[GArray, np.ndarray, float]) -> GArray:
        other_data = other.data if isinstance(other, GArray) else other
        return GArray(self.grid, self.data / other_data)

    def __neg__(self) -> GArray:
        return GArray(self.grid, -self.data)

    def __repr__(self) -> str:
        grid_name = type(self.grid).__name__
        return f"GArray(grid={grid_name}, shape={self.data.shape})"
