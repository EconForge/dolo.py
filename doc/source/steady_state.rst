Steady-state
............

The deterministic state of a model corresponds to steady-state values :math:`\overline{m}`
of the exogenous process. States and controls satisfy:

 :math:`\overline{s} = g\left(\overline{m}, \overline{s}, \overline{x}, \overline{m} \right)`

 :math:`0 = \left[ f\left(\overline{m}, \overline{s}, \overline{x}, \overline{m}, \overline{s}, \overline{x} \right) \right]`

where :math:`g` is the state transition function, and :math:`f` is the arbitrage equation.
Note that the shocks, :math:`\epsilon`, are held at their deterministic mean.

The steady state function consists in solving the system of arbitrage equations for
the steady state values of the controls, :math:`\overline{x}`, which can then be used along
with the transition function to find the steady state values of the state variables,
:math:`\overline{s}`.

.. autofunction:: dolo.algos.steady_state.residuals
