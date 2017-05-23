Steady-state
............

We consider an `fg` model at the deterministic steady state:

 :math:`s = g\left(s, x, \epsilon \right)`

 :math:`0 = \left[ f\left(s, x, s, x \right) \right]`

where :math:`g` is the state transition function, and :math:`f` is the arbitrage equation.
Note that the shocks, :math:`\epsilon`, are held at their deterministic mean.

The steady state function consists in solving the system of arbitrage equations for
the steady state values of the controls, :math:`x`, which can then be used along
with the transition function to find the steady state values of the state variables,
:math:`s`.

.. autofunction:: dolo.algos.steady_state.residuals
