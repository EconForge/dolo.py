"""
Space hierarchy for dolo.py.

Following Dolo.jl:
- Space: Mathematical domain for economic variables (continuous or discrete).
- CartesianSpace: Continuous bounded hypercube [min, max] (loc = val).
- DiscreteSpace (GridSpace): Finite set of discrete points/regimes (loc = index).
- ProductSpace: Cartesian product of multiple spaces (e.g. exo × endo).
- EmptySpace: Neutral empty domain (discretizes to EmptyGrid).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np

from dolo.numeric.garray import QualifiedPoint, QP
from dolo.numeric.grids import (
    Grid,
    UniformCartesianGrid,
    UnstructuredGrid,
    ProductGrid,
    EmptyGrid,
    SmolyakGrid,
    NonUniformCartesianGrid,
)


class Space(ABC):
    """Abstract base class representing a mathematical space for economic variables."""

    variables: Tuple[str, ...]

    @property
    def d(self) -> int:
        """Dimension of the space."""
        return len(self.variables)

    @abstractmethod
    def contains(self, point: Any) -> bool:
        """Check if a point is within this space."""
        pass

    def __contains__(self, point: Any) -> bool:
        return self.contains(point)

    @abstractmethod
    def qp(self, loc: Any, val: Optional[Any] = None) -> QualifiedPoint:
        """Construct a QualifiedPoint bound to this space."""
        pass

    @abstractmethod
    def draw(self, n: int = 1) -> Union[QualifiedPoint, List[QualifiedPoint]]:
        """Sample random QualifiedPoint(s) from this space."""
        pass

    @abstractmethod
    def discretize(self, **options) -> Grid:
        """Discretize the continuous or discrete space into a computational Grid."""
        pass

    def __mul__(self, other: Space) -> ProductSpace:
        if isinstance(other, ProductSpace):
            return ProductSpace((self, *other.spaces))
        return ProductSpace((self, other))


class CartesianSpace(Space, dict):
    """Continuous hypercube domain bounded by [min, max].

    Supports dict-like access `space['k']` for backwards compatibility with
    legacy `CartesianDomain`.
    """

    def __init__(
        self,
        bounds: Optional[Dict[str, Sequence[float]]] = None,
        *,
        min: Optional[Sequence[float]] = None,
        max: Optional[Sequence[float]] = None,
        variables: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        dict.__init__(self)

        # Merge dict kwargs or bounds dict
        all_bounds: Dict[str, np.ndarray] = {}
        if bounds is not None:
            for k, v in bounds.items():
                all_bounds[k] = np.array(v, dtype=float)
        for k, v in kwargs.items():
            all_bounds[k] = np.array(v, dtype=float)

        if all_bounds:
            self.variables = tuple(all_bounds.keys())
            self._min = np.array(
                [all_bounds[k][0] for k in self.variables], dtype=float
            )
            self._max = np.array(
                [all_bounds[k][1] for k in self.variables], dtype=float
            )
            for k, v in all_bounds.items():
                self[k] = v
        else:
            if min is None or max is None:
                raise ValueError(
                    "Must specify either bounds dict or min and max arrays."
                )
            self._min = np.array(min, dtype=float)
            self._max = np.array(max, dtype=float)
            if variables is not None:
                self.variables = tuple(variables)
            else:
                self.variables = tuple(f"x_{i}" for i in range(len(self._min)))
            for i, k in enumerate(self.variables):
                self[k] = np.array([self._min[i], self._max[i]], dtype=float)

    @property
    def min(self) -> np.ndarray:
        return self._min

    @property
    def max(self) -> np.ndarray:
        return self._max

    @property
    def states(self) -> Tuple[str, ...]:
        """Backwards compatibility alias for variables."""
        return self.variables

    def contains(self, point: Any) -> bool:
        if isinstance(point, str):
            return point in self.variables
        try:
            pt = np.asarray(point, dtype=float)
            if pt.ndim == 1:
                return bool(np.all(pt >= self._min) and np.all(pt <= self._max))
            return bool(
                np.all(pt >= self._min, axis=-1).all()
                and np.all(pt <= self._max, axis=-1).all()
            )
        except (ValueError, TypeError):
            return False

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, str):
            return dict.__contains__(self, item)
        return self.contains(item)

    def qp(self, loc: Any, val: Optional[Any] = None) -> QualifiedPoint:
        """In CartesianSpace, location and continuous coordinate values coincide."""
        coords = np.asarray(loc if val is None else val, dtype=float)
        return QualifiedPoint(loc=coords, val=coords)

    def draw(self, n: int = 1) -> Union[QualifiedPoint, List[QualifiedPoint]]:
        pts = np.random.uniform(self._min, self._max, size=(n, self.d))
        qps = [self.qp(pts[i]) for i in range(n)]
        return qps[0] if n == 1 else qps

    def discretize(
        self,
        n: Optional[Union[int, Sequence[int]]] = None,
        type: str = "cartesian",
        **kwargs,
    ) -> Grid:
        grid_type = str(type).lower()
        if n is None:
            n = kwargs.get("orders", kwargs.get("n", None))
        if n is None:
            n = [20] * self.d
        elif isinstance(n, int):
            n = [n] * self.d
        else:
            n = list(n)

        if grid_type in ("cartesian", "uniform", "uniformcartesian"):
            return UniformCartesianGrid(self.min, self.max, n)
        elif grid_type in ("smolyak", "smolyakgrid"):
            mu = kwargs.get("mu", 2)
            return SmolyakGrid(self.min, self.max, mu=mu)
        elif grid_type in ("nonuniform", "nonuniformcartesian"):
            nodes = kwargs.get("nodes")
            if nodes is None:
                raise ValueError("Non-uniform Cartesian grid requires explicit nodes.")
            return NonUniformCartesianGrid(nodes)
        else:
            raise ValueError(f"Unknown grid type: {grid_type}")

    def __repr__(self) -> str:
        b_str = ", ".join(
            f"{k}: [{self[k][0]:.3g}, {self[k][1]:.3g}]" for k in self.variables
        )
        return f"CartesianSpace({b_str})"


class DiscreteSpace(Space):
    """Finite discrete set of points or regimes (e.g. Markov states).

    In a DiscreteSpace:
    - `loc` is the integer regime index `i` (0 <= i < n_points).
    - `val` is the discrete coordinate vector `points[i]`.
    """

    def __init__(
        self,
        points: Union[Sequence[Sequence[float]], np.ndarray],
        variables: Optional[Sequence[str]] = None,
    ):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim == 1:
            self.points = self.points[:, None]
        self.n_points = self.points.shape[0]

        if variables is not None:
            self.variables = tuple(variables)
        else:
            self.variables = tuple(f"m_{i}" for i in range(self.points.shape[1]))

    def contains(self, point: Any) -> bool:
        if isinstance(point, (int, np.integer)):
            return 0 <= int(point) < self.n_points
        pt = np.asarray(point, dtype=float)
        diffs = np.linalg.norm(self.points - pt, axis=-1)
        return bool(np.any(diffs < 1e-12))

    def qp(self, loc: Any, val: Optional[Any] = None) -> QualifiedPoint:
        if isinstance(loc, (int, np.integer)):
            idx = int(loc)
            coord = self.points[idx] if val is None else np.asarray(val, dtype=float)
            return QualifiedPoint(loc=idx, val=coord)
        pt = np.asarray(loc, dtype=float)
        diffs = np.linalg.norm(self.points - pt, axis=-1)
        idx = int(np.argmin(diffs))
        return QualifiedPoint(
            loc=idx, val=pt if val is None else np.asarray(val, dtype=float)
        )

    def draw(self, n: int = 1) -> Union[QualifiedPoint, List[QualifiedPoint]]:
        indices = np.random.choice(self.n_points, size=n)
        qps = [self.qp(int(idx)) for idx in indices]
        return qps[0] if n == 1 else qps

    def discretize(self, **options) -> UnstructuredGrid:
        return UnstructuredGrid(self.points)

    def __repr__(self) -> str:
        return f"DiscreteSpace(n_points={self.n_points}, vars={self.variables})"


# Alias matching Dolo.jl
GridSpace = DiscreteSpace


class EmptySpace(Space):
    """Empty / Neutral space (used when exogenous shocks are absent)."""

    variables: Tuple[str, ...] = ()

    def contains(self, point: Any) -> bool:
        return True

    def qp(self, loc: Any = None, val: Optional[Any] = None) -> QualifiedPoint:
        return QualifiedPoint(loc=None, val=np.empty((0,)))

    def draw(self, n: int = 1) -> Union[QualifiedPoint, List[QualifiedPoint]]:
        qps = [self.qp() for _ in range(n)]
        return qps[0] if n == 1 else qps

    def discretize(self, **options) -> EmptyGrid:
        return EmptyGrid()

    def __repr__(self) -> str:
        return "EmptySpace()"


class ProductSpace(Space):
    """Product of two or more spaces (e.g. Exogenous × Endogenous)."""

    def __init__(
        self,
        spaces: Sequence[Space],
        names: Optional[Sequence[str]] = None,
    ):
        self.spaces = tuple(spaces)
        if names is not None:
            self.names = tuple(names)
        elif len(self.spaces) == 2:
            self.names = ("exo", "endo")
        else:
            self.names = tuple(f"space_{i}" for i in range(len(self.spaces)))

        all_vars: List[str] = []
        for s in self.spaces:
            all_vars.extend(s.variables)
        self.variables = tuple(all_vars)

    @property
    def exo(self) -> Space:
        return self.spaces[0]

    @property
    def endo(self) -> Space:
        return self.spaces[1]

    def __getitem__(self, key: Union[int, str]) -> Space:
        if isinstance(key, str):
            idx = self.names.index(key)
            return self.spaces[idx]
        return self.spaces[key]

    def contains(self, point: Any) -> bool:
        if isinstance(point, (tuple, list)) and len(point) == len(self.spaces):
            return all(s.contains(p) for s, p in zip(self.spaces, point))
        return False

    def qp(self, loc: Any, val: Optional[Any] = None) -> QualifiedPoint:
        if isinstance(loc, tuple) and len(loc) == len(self.spaces):
            loc_tuple = tuple(loc)
            if val is not None:
                val_arr = np.asarray(val, dtype=float)
            else:
                qps = [s.qp(l) for s, l in zip(self.spaces, loc)]
                val_arr = np.concatenate([q.val for q in qps])
            return QualifiedPoint(loc=loc_tuple, val=val_arr)
        val_arr = np.asarray(val if val is not None else loc, dtype=float)
        return QualifiedPoint(loc=loc, val=val_arr)

    def draw(self, n: int = 1) -> Union[QualifiedPoint, List[QualifiedPoint]]:
        res = []
        for _ in range(n):
            sub_qps = [s.draw(1) for s in self.spaces]
            loc = tuple(q.loc for q in sub_qps)
            val = np.concatenate([q.val for q in sub_qps])
            res.append(QualifiedPoint(loc=loc, val=val))
        return res[0] if n == 1 else res

    def discretize(self, **options) -> ProductGrid:
        grids = []
        for name, space in zip(self.names, self.spaces):
            sub_opts = (
                options.get(name, {}) if isinstance(options.get(name), dict) else {}
            )
            grids.append(space.discretize(**sub_opts))
        return ProductGrid(grids[0], grids[1], names=list(self.names))

    def __repr__(self) -> str:
        return " × ".join(repr(s) for s in self.spaces)
