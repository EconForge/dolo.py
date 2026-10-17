# Time Iteration

Time iteration is the core nonlinear policy-iteration algorithm in Dolo for solving dynamic stochastic general equilibrium (DSGE) models, optionally subject to occasionally binding inequality constraints.

---

## 1. Mathematical Formulation

### Classical Structural Form

In its classical structural formulation, a discrete-time continuous-state economic model is represented by:

$$\begin{aligned}
s_t & = g\left(m_{t-1}, s_{t-1}, x_{t-1}, m_t \right) \\
0   & = \mathbb{E}_t \left[ f\left(m_t, s_t, x_t, m_{t+1}, s_{t+1}, x_{t+1} \right) \right] \quad \perp \quad \underline{x}(m_t, s_t) \le x_t \le \overline{x}(m_t, s_t)
\end{aligned}$$

where:
- $m_t$ is the vector of exogenous shocks,
- $s_t$ is the vector of endogenous states,
- $x_t$ is the vector of controls (decisions),
- $g$ is the state transition law,
- $f$ is the arbitrage equation (Euler residuals),
- $\underline{x}, \overline{x}$ are lower and upper bounds on controls, with the complementarity condition $\perp$.

### The Unified $\tau$ Transition Operator

Following the design of **Dolo.jl**, `dolo.py` generalizes this formulation by abstracting the state transition and stochastic process into a unified transition kernel $\tau$. 

Instead of treating exogenous Markov chains, VAR(1) processes, and i.i.d. quadrature shocks through disparate control flows, the next state distribution is written universally as:

$$S_{t+1} \sim \tau(S_t, x_t)$$

where a state $S$ is represented as a **`QualifiedPoint`** (or `QP`) bundling:
1. **`loc` (Locator / Address):** The discrete regime index or computational address in the discretized domain (e.g. regime index $i_m$ for Markov chains, or coordinate indices for table lookups).
2. **`val` (Evaluated Payload):** The continuous economic state coordinates $[m, s]$ passed into equations.

Given a policy rule $x = \varphi(S)$, the conditional transition $\tau(S_t, x_t)$ yields a set of discrete branches with weights and target qualified states:

$$\tau(S_t, x_t) = \left\{ \big(w_k, \, S^{(k)}_{t+1}\big) \right\}_{k=1}^K, \quad \sum_{k=1}^K w_k = 1$$

The arbitrage equation is formulated universally as:

$$0 = \mathbb{E}_{\tau} \left[ f\left(S_t, x_t, S_{t+1}, \varphi(S_{t+1})\right) \right] = \sum_{(w_k, S'_k) \in \tau(S_t, x_t)} w_k \cdot f\left(S_t, x_t, S'_k, \varphi(S'_k)\right)$$

This abstraction ensures that the Euler residuals and solution algorithms are identical whether the model features pure continuous states, discrete Markov regimes, or mixed exogenous processes.

---

## 2. Discrete Model & Core Abstractions

### `DiscretizedModel`

A **`DiscretizedModel`** bundles the continuous model specification, the computational grid $\mathcal{G}$, and the discrete transition operator $\tau$:

$$\langle \text{Model}, \; \mathcal{G}, \; \tau \rangle$$

```python
from dolo import yaml_import
from dolo.numeric import DiscretizedModel

model = yaml_import("examples/models/rbc.yaml")
dmodel = DiscretizedModel(model)
```

It provides:
- `dmodel.initial_guess()`: Returns a `GArray` initialized from steady-state calibration.
- `dmodel.lower_bounds()`, `dmodel.upper_bounds()`: Returns `GArray` bounds.
- `dmodel.arbitrage(...)`: Evaluates arbitrage equations across regimes and batches.
- `dmodel.transition_matrix(dr)`: Computes the full row-stochastic Markov transition matrix $P$ on grid nodes.
- `dmodel.ergodic_distribution(dr)`: Computes the stationary distribution $\mu P = \mu$.

### `GArray` (Grid-Bound Fields)

A `GArray` represents a discrete field of values living on a grid $\mathcal{G}$. It exhibits a powerful duality:
- **Geometric Indexing:** Slice by discrete regime or subgrid, e.g., `controls[i_m]`.
- **Vectorized Solver Interface:** Directly provides `ravel()` (flat 1D view for SciPy/Newton solvers) and `unravel(flat_x)` to reconstruct structured fields without manual tensor reshaping.
- **Linear Algebra:** Implements vectorized addition, scaling, and $L_\infty$ distance norm (`ga.norm()`).

---

## 3. The Time Iteration Algorithm

The algorithm proceeds as follows:

1. **Initial Guess:** Start with an initial policy field $x^{(0)} = \varphi^{(0)} \in \mathcal{G}$ (typically the calibrated steady-state controls).
2. **Policy Update (Euler Equation Solve):**
   Given future policy $\varphi^{(n)}$, solve for current period controls $\varphi^{(n+1)}(S)$ at each grid node $S \in \mathcal{G}$:
   $$0 = \sum_{(w_k, S'_k) \in \tau(S, x)} w_k \cdot f\left(S, x, S'_k, \varphi^{(n)}(S'_k)\right) \quad \perp \quad \underline{x}(S) \le x \le \overline{x}(S)$$
   This step uses a vectorized serial Newton-Raphson or Non-Linear Complementarity Problem (NCP) solver.
3. **Policy Acceleration (Optional Multi-Step Lookahead):**
   To accelerate convergence, an intermediate policy evaluation step can be applied using Neumann series or GMRES linear solves (similar to Howard improvement in value iteration):
   $$\varphi^{(n+1)} \leftarrow \varphi^{(n+1)} + (I - J_x)^{-1} \cdot R\big(\varphi^{(n+1)}\big)$$
4. **Convergence Check:**
   Compute the residual norm $\epsilon_{n+1} = \|R(\varphi^{(n+1)})\|_\infty$ and the policy step error $\eta_{n+1} = \|\varphi^{(n+1)} - \varphi^{(n)}\|_\infty$:
   - If $\epsilon_{n+1} < \tau_\epsilon$ and $\eta_{n+1} < \tau_\eta$, terminate and return the converged decision rule $\varphi^*$.
   - Otherwise, update $\varphi^{(n)} \leftarrow \varphi^{(n+1)}$ and return to Step 2.

---

## 4. Usage & Options

### Basic Usage

```python
from dolo import yaml_import, time_iteration

# Load model
model = yaml_import("examples/models/rbc.yaml")

# Solve using time iteration
sol = time_iteration(model)

# Access decision rule
dr = sol.dr
print("Converged:", sol.converged)
print("Iterations:", sol.iterations)
```

### Accelerated Time Iteration

For models with high persistence or slow policy adjustment, acceleration drastically reduces the number of outer iterations:

```python
# Enable Neumann policy acceleration
sol = time_iteration(model, accelerate=True)

# Custom acceleration configuration
sol = time_iteration(
    model,
    accelerate={
        "linsolve": "neumann",  # or "gmres"
        "maxit": 50,
        "wait": 5,             # wait 5 iterations before starting acceleration
        "tol": 1e-8,
    }
)
```

---

## 5. API Reference

::: dolo.algos.time_iteration.time_iteration

::: dolo.algos.results.TimeIterationResult

