# Model Representation & Numeric Core

!!! warning "Work in Progress"
    This chapter documents a **work-in-progress** redesign of the numeric core
    of `dolo.py`, following the architecture of **Dolo.jl**. Several concepts
    described here (in particular *spaces* and *approximators*) are only
    partially implemented, and the public API may still change. The current,
    stable API used by the solution algorithms is documented in the
    "Model API" chapter.

For global nonlinear solution methods, a continuous model (written in YAML or
`.dyno`) must be turned into a fully discretized numerical object that both
solution and simulation algorithms can iterate on. This chapter describes the
layered abstractions that perform this conversion:

1. the **space** — the (abstract) domain on which the model lives;
2. the **grids** — concrete sets of nodes that discretize the space;
3. the **qualified points** (`QP`) — points that carry both an *address* in
   the grid and *values* in economic coordinates;
4. the **approximators** — objects that interpolate a policy across the grid;
5. the **`DiscretizedModel`** — the bundle tying the model, the grid and the
   transition operator `tau` together.

---

## 1. Spaces and Grids

### The `Space` Hierarchy

In economic modeling, variables live on a **mathematical space** (the continuous or discrete domain) rather than a computational grid. The **`Space`** hierarchy in `dolo.numeric.spaces` provides first-class objects for these domains:

- **`CartesianSpace`**: A continuous bounded hypercube $[\underline{s}, \overline{s}] \subset \mathbb{R}^d$.
  - In a `CartesianSpace`, location and continuous coordinates coincide (`loc = val = coords`).
  - Supports domain membership checks (`pt in space`), random sampling (`space.draw()`), and discretization (`space.discretize(n=20)`).
  - Preserves dict-like indexing `space['k']` for backward compatibility with `model.domain`.
- **`DiscreteSpace`** (alias **`GridSpace`**): A finite discrete set of points $\{p_1, \dots, p_N\}$ (e.g. Markov states).
  - In a `DiscreteSpace`, `loc` is the integer regime index $i \in \{0, \dots, N-1\}$, while `val = points[i]`.
  - Discretizes to an `UnstructuredGrid`.
- **`ProductSpace`**: The Cartesian product of multiple spaces (e.g. $S_{exo} \times S_{endo}$).
  - `loc` is a composite tuple `(loc_exo, loc_endo)`, and `val` is concatenated $[m, s]$.
  - Discretizes to a `ProductGrid`.
- **`EmptySpace`**: The neutral empty space for models without discrete exogenous states (discretizes to `EmptyGrid`).

```python
from dolo.numeric import CartesianSpace, DiscreteSpace, ProductSpace

# Continuous state space
endo_space = CartesianSpace(min=[0.1, 1.0], max=[0.5, 5.0], variables=["z", "k"])
assert [0.2, 2.0] in endo_space

# Discrete regime space
exo_space = DiscreteSpace([[-0.01], [0.01]], variables=["e"])

# Product space
space = exo_space * endo_space
grid = space.discretize(endo={"n": [20, 40]})   # -> ProductGrid
```

### Grids (Discretizations of Spaces)

A **`Grid`** represents a concrete, discrete set of nodes sampling a `Space`:

$$\mathcal{G} = \mathcal{M} \times \mathcal{S}$$

- $\mathcal{M}$ is the grid of the discretized exogenous process (regimes / shock nodes);
- $\mathcal{S}$ is the grid of endogenous states (a Cartesian product of intervals).

A `ProductGrid` bundles the two sub-grids and supports lookup by name:

```python
grid.grids      # -> [exo_grid, endo_grid]
grid["endo"]    # -> the endogenous sub-grid
```

### Grid types

Grids are defined in `dolo.numeric.grids`. All grid types expose `nodes`
(the array of node coordinates, shape `(n_nodes, d)`) and `node(i)` (the
`i`-th node):

| Class | Description |
| --- | --- |
| `EmptyGrid` | The neutral grid (`n_nodes == 0`); used as the exogenous grid of purely deterministic / i.i.d. models. |
| `PointGrid` | A single point. |
| `UnstructuredGrid` | An arbitrary, user-provided list of nodes (e.g. the values of a Markov chain). |
| `UniformCartesianGrid` | A Cartesian product of uniform 1-D grids, defined by `min`, `max` and per-dimension counts `n`. |
| `NonUniformCartesianGrid` | A Cartesian product of arbitrary per-dimension node lists. |
| `SmolyakGrid` | A sparse Smolyak grid (for high-dimensional interpolation). |
| `ProductGrid` | The product of two sub-grids (exogenous × endogenous). |

```python
from dolo.numeric.grids import UniformCartesianGrid, UnstructuredGrid, ProductGrid

k_grid   = UniformCartesianGrid(min=[0.0], max=[10.0], n=[50])
z_values = UnstructuredGrid([[-0.02, 0.0], [0.02, 0.0]])

grid = ProductGrid(z_values, k_grid)
grid.nodes.shape   # (2 * 50, 2)  — one node per (regime, k-node) pair
```

!!! note
    The same `Grid` interface is used both for the exogenous and endogenous
    parts, which is what lets the transition operator `tau` and the decision
    rules treat Markov regimes, i.i.d. quadrature and pure-continuous models
    uniformly.

---

## 2. Qualified Points (`QP`) and Batches

A raw array of coordinates says *where* a point is but not *how it is
addressed* in the discretized domain. Dolo separates those two roles.

### `QualifiedPoint` / `QP`

A **`QualifiedPoint`** (alias **`QP`**) bundles:

- `loc` — the *locator / address*: a regime index, a grid coordinate `(i, j)`,
  or any other token used to index a lookup table or a spline;
- `val` — the *evaluated payload*: the continuous economic coordinates
  $[m, s]$ (and derived quantities) actually passed to the model equations.

```python
from dolo.numeric import QP

qp = QP(loc=(i_m, s_endo), val=np.array([m, s]))
qp.coords   # -> np.array([m, s])       (alias for val)
qp.index    # -> i_m                    (first element of a tuple loc)
```

### `QualifiedBatch` / `SliceQP`

For vectorized operations over a whole continuous sub-grid, a
**`QualifiedBatch`** (aliases `QualifiedSlice`, `SliceQP`) groups:

- `loc` — the target discrete regime index (or `None` for pure-continuous
  models);
- `exo` — the exogenous state values for the branch, shape `(N, n_m)`;
- `endo` — the endogenous state matrix, shape `(N, n_s)`.

```python
from dolo.numeric import SliceQP

batch = SliceQP(loc=j, exo=M_val, endo=S_endo)
batch.val    # -> hstack([exo, endo]), shape (N, n_m + n_s)
```

---

## 3. The Transition Operator `tau`

### Unified transition kernel

The redesign collapses the model's exogenous stochastic process and its
endogenous law of motion into a **single conditional distribution operator**
`tau`:

$$S_{t+1} \sim \tau(S_t, x_t)$$

Given a state $S_t$ (a `QP`) and a control vector $x_t$, `tau` yields the set
of next states with their probabilities:

$$\tau(S_t, x_t) = \left\{ \big(w_k, \, S^{(k)}_{t+1}\big) \right\}_{k=1}^K,
\qquad \sum_{k=1}^K w_k = 1$$

This lets the Euler residuals be written identically for every kind of
process:

$$0 = \mathbb{E}_{\tau}\left[ f\left(S_t, x_t, S_{t+1},\varphi(S_{t+1})\right)\right]
= \sum_{(w_k, S'_k) \in \tau(S_t, x_t)}
w_k \cdot f\left(S_t, x_t, S'_k, \varphi(S'_k)\right)$$

### The `Tau` object

`Tau` wraps the transition equation $g$ and the discretized process:

```python
from dolo.numeric import Tau
tau = Tau(g_fun, dprocess, parms)
```

It offers three levels of granularity, all of which are generators over
`(weight, next_point)` pairs:

- `tau.point_transitions(s: QP, x)` — a single state point; yields
  `(w, QP)` for each shock branch.
- `tau.slice_transitions(i_m, m_val, s_endo, xm)` — the whole continuous
  sub-grid at once (vectorized); yields `(w, SliceQP)`.
- `tau.tau_fit(i_m, m_val, s_endo, xm, endo_grid)` — transitions **projected
  onto the grid vertices** via the *trembling-hand* multilinear projector
  (see below); yields `(target_regime, prob, corner_weights, corner_indices)`.

### Trembling-hand projection

To build the finite Markov transition matrix $P$ on the grid nodes, `tau_fit`
projects each continuous next-state $S_{t+1}$ onto the $2^d$ enclosing
vertices of the Cartesian endogenous grid, using standard multilinear
(bilinear/trilinear) weights:

```python
from dolo.numeric import trembling_hand
projections = trembling_hand(endo_grid, S_endo)   # [(corner_weight, corner_idx), ...]
```

Each projected branch $(w_k, S'_k)$ is split into $2^d$ corner contributions,
so that the total transition matrix remains row-stochastic. This is the
"trembling hand" that lets the ergodic distribution and Markov-chain
simulations be computed directly on the grid.

---

## 4. The `DiscretizedModel`: the new model representation

The **`DiscretizedModel`** is the primary numerical representation of a model.
It bundles the continuous/symbolic model, the grid (the *space*), and the
transition operator:

$$\langle \text{Model}, \; \mathcal{G}, \; \tau \rangle$$

```python
from dolo import yaml_import
from dolo.numeric import DiscretizedModel

model  = yaml_import("examples/models/rbc.yaml")
dmodel = DiscretizedModel(model)
```

Attributes:

- `dmodel.model` — the symbolic model (`Model` / `ModelV2`);
- `dmodel.grid` — the product grid $\mathcal{G}$;
- `dmodel.tau` — the transition operator $\tau$;
- `dmodel.dprocess` — the underlying discretized process;
- `dmodel.n_regimes`, `dmodel.endo_grid`, `dmodel.endo_nodes`,
  `dmodel.N_endo`, `dmodel.N_total` — derived grid dimensions.

Convenience methods:

- `dmodel.regimes()` — iterates over `(regime_index, m_value, endo_nodes)`;
- `dmodel.arbitrage(...)` — evaluates the arbitrage equation across regimes
  and batches;
- `dmodel.initial_guess()` — a `GArray` of calibrated controls over the grid;
- `dmodel.lower_bounds()`, `dmodel.upper_bounds()` — `GArray` bounds;
- `dmodel.transition_matrix(controls)` — the full row-stochastic transition
  matrix $P$ on grid nodes, built from `tau.tau_fit`;
- `dmodel.ergodic_distribution(controls)` — the stationary distribution
  solving $\mu P = \mu$.

---

## 5. `GArray`: grid-bound fields

A **`GArray`** binds an array of values to a grid $\mathcal{G}$, and exposes a
dual interface:

- **geometric indexing** — `g[i_m]` returns the slice of the field living on
  the sub-grid of discrete regime `i_m`;
- **solvers' flat view** — `g.ravel()` returns a flat 1-D view of the degrees
  of freedom for SciPy / Newton solvers, and `g.unravel(flat)` reconstructs
  the structured field, so solvers never touch tensor reshaping.

```python
x0 = dmodel.initial_guess()       # GArray of controls
x0[i_m]                            # (N_endo, n_x) slice for regime i_m
flat = x0.ravel()                  # 1-D solver iterate
x1 = x0.unravel(flat)              # back to a GArray
x0.norm()                          # L_infinity norm
```

`GArray` implements the usual array arithmetic (`+`, `-`, `*`, `/`, unary
`-`) and `copy()`/`zeros_like()`/`ones_like()` helpers.

---

## 6. Approximators

The solution algorithms do not return raw point values on the grid; they return a decision rule $x = \varphi(S)$ that can be evaluated *anywhere* in the continuous space. The objects that interpolate or approximate the policy are collectively the **approximators** in `dolo.numeric.approximators`:

- **`LinearApproximator`**: Piecewise multilinear interpolation on Cartesian grids.
- **`CubicSplineApproximator`**: Prefiltered cubic spline interpolation on Cartesian grids.
- **`SmolyakApproximator`**: Multivariate Chebyshev polynomial approximation on Smolyak grids.
- **`ProductApproximator`**: Regime-wise composite approximator over `ProductGrid` / `ProductSpace` (delegating to continuous approximators per discrete regime).

An `Approximator` provides:
- `app.grid`: The compatible discretization grid.
- `app.fit(values)`: Fits coefficients from field values on the grid.
- `app.evaluate_coords(coeffs, s)`: Evaluates approximation at continuous coordinate matrix.
- `app.evaluate(coeffs, qp)`: Evaluates approximation at a `QualifiedPoint`.
- `app.evaluate_batch(coeffs, batch)`: Evaluates approximation over a `QualifiedBatch` / `SliceQP`.

### DecisionRule and `fit()`

A **`DecisionRule`** bundles the space, grid, approximator, and fitted coefficients into a callable object:

```python
from dolo.numeric import DecisionRule

# Fluent construction and fitting (matching Julia's fit!):
dr = DecisionRule(exo_grid, endo_grid, interp_method="cubic").fit(controls_ga)

# dr.set_values is preserved as an alias for backwards compatibility:
dr.set_values(controls_ga)

# Callable over QP, SliceQP, or regime + continuous coordinates:
dr(qp)
dr(slice_qp)
dr(i_m, s)
```

You can also construct a `DecisionRule` directly from a `DiscretizedModel`:

```python
dr = dmodel.decision_rule(controls_ga, interp_method="cubic")
```

---

## 7. Roadmap

With first-class **`Space`** and **`Approximator`** hierarchies now landed, the remaining items on the redesign roadmap are:

- **`tau`-based simulation and ergodic analysis**: using $\tau$ directly inside time-series simulation loops (replacing ad-hoc process branching);
- **Higher-order Taylor approximators**: local perturbation expansions around the steady state implementing the unified `Approximator` interface.

The stable, algorithm-facing `DiscretizedModel` / `tau` / `GArray` / `QP` / `Space` / `Approximator` abstractions are used by the global solvers (see the "Time Iteration" chapter).
