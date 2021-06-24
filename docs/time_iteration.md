Time iteration
==============

We consider a model with the form:

$$\begin{aligned}
s_t & = & g\left(m_{t-1}, s_{t-1}, x_{t-1}, m_t \right) \\
0   & = & E_t \left[ f\left(m_t, s_{t}, x_{t}, m_{t+1}, s_{t+1}, x_{t+1} \right) \right]
\end{aligned}$$

where $g$ is the state transition function, and $f$ is the arbitrage equation.

The time iteration algorithm consists in approximating the optimal controls as a function $\varphi$ of exogenous and endogenous controls $x_t = \varphi(m_t,s_t)$.

- At step $n$, the current guess for the control, $x(s_t) = \varphi^n(m_t, s_t)$, serves as the control being exercised next period :
    - Taking $\varphi^n$ as the initial guess, find the current period's controls $\varphi^{n+1}(m_t,s_t)$  for any $(m_t,s_t)$ by solving the arbitrage equation :
$0 = E_t \left[ f\left(m_t, s_{t}, \varphi^{n+1}(m_t, s_t), g(m_t, s_t, \varphi^{n+1}(m_t, s_t), m_{t+1}), \varphi^{n}(m_{t+1},g(m_t, s_t, \varphi^{n+1}(m_t, s_t), m_{t+1})) \right) \right]$
- Repeat until $\eta_{n+1} = \max_{m,s}\left |\varphi^{n+1}(m,s) - \varphi^{n}(m,s) \right|$ is smaller than prespecified criterium $\tau_{η}$

![mkapi](dolo.algos.time_iteration.time_iteration)

![mkapi](dolo.algos.results.TimeIterationResult)

