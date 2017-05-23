Time iteration
..............

We consider an `fg` model, that is a model with the form:

 :math:`s_t = g\left(s_{t-1}, x_{t-1}, \epsilon_t \right)`

 :math:`0 = E_t \left[ f\left(s_{t}, x_{t}, s_{t+1}, x_{t+1} \right) \right]`

where :math:`g` is the state transition function, and :math:`f` is the arbitrage equation.

The time iteration algorithm consists in approximating the optimal controls, :math:`x_t = x(s_t)`.

At step :math:`n`, the current guess for the control, :math:`x(s_t) = \varphi^n(s_t)`, serves as the control being exercised next period  :
  - Given current guess, find the current period's control by solving the arbitrage equation : :math:`0 = E_t \left[ f\left(s_{t}, \varphi^{n+1}(s_t), g(s_t, \varphi^{n+1}(s_t)), \varphi^{n}(g(s_t, \varphi^{n+1}(s_t))) \right) \right]`

.. autofunction:: dolo.algos.time_iteration.time_iteration

.. autofunction:: dolo.algos.time_iteration.residuals_simple
