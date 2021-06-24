Steady-state
============

The deterministic state of a model corresponds to steady-state values
$\overline{m}$ of the exogenous process. States and controls satisfy:

> $\overline{s} = g\left(\overline{m}, \overline{s}, \overline{x}, \overline{m} \right)$
>
> $0 = \left[ f\left(\overline{m}, \overline{s}, \overline{x}, \overline{m}, \overline{s}, \overline{x} \right) \right]$

where $g$ is the state transition function, and $f$ is the arbitrage
equation. Note that the shocks, $\epsilon$, are held at their
deterministic mean.

The steady state function consists in solving the system of arbitrage
equations for the steady state values of the controls, $\overline{x}$,
which can then be used along with the transition function to find the
steady state values of the state variables, $\overline{s}$.

![mkapi](dolo.algos.steady_state.residuals)
