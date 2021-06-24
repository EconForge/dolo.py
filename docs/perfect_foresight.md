Perfect foresight
=================

Consider a series for the exogenous process $(m_t)_{0 \leq t \leq T}$ given exogenously.

The perfect foresight problem consists in finding the path of optimal
controls $(x_t)_{0 \leq t \leq T}$ and corresponding states $(s_t)_{0 \leq t \leq T}$ such that:

$$\begin{aligned}
s_t & = & g\left(m_{t-1}, s_{t-1}, x_{t-1}, m_t \right) & \\
0 & = & f\left(m_{t}, s_{t}, x_{t}, m_{t+1}, s_{t+1}, x_{t+1}\right) & \ \perp \ \underline{u} <= x_t <= \overline{u}
\end{aligned}$$

Special conditions apply for the initial state and controls. Initial
state (${m_0}, {s_0})$ is given exogenously. Final states and controls are
determined by assuming the exogenous process satisfies $m_t=m_T$ for all
$t\geq T$ and optimality conditions are satisfied in the last period:

$$f(m_T, s_T, x_T, m_T, s_T, x_T) \perp \underline{u} \leq x_T \leq \overline{u}$$

We assume that $\underline{u}$ and $\overline{u}$ are constants. This is
not a big restriction since the model can always be reformulated in
order to meet that constraint, by adding more equations.

The stacked system of equations satisfied by the solution is:

| Transitions | Arbitrage | 
|-------------|------------|
| $s_0$ exogenous |  $f(m_0, s_0, x_0, m_1, s_1, x_1) \perp \underline{u} <= x_0 <= \overline{u}$ |
| $s_1 = g(m_0, s_0, x_0, m_1)$ | $f(s_1, x_1, s_2, x_2) \perp \underline{u} <= x_1 <= \overline{u}$ |
| .... | ... |
| $s_T = g(m_{T-1}, s_{T-1}, x_{T-1}, m_T)$ | $f(m_T, s_T, x_T, m_T, s_T, x_T) \perp \underline{u} <= x_T <= \overline{u}$ |

The system is solved using a nonlinear solver.


![mkapi](dolo.algos.perfect_foresight.deterministic_solve)

