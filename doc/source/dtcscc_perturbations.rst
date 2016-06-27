Perturbation
............

We consider an `fg` model, that is a model with the form:

 :math:`s_t = g\left(s_{t-1}, x_{t-1}, \epsilon_t \right)`

 :math:`0 = E_t \left[ f\left(s_{t}, x_{t}, s_{t+1}, x_{t+1} \right) \right]`

where :math:`g` is the state transition function, and :math:`f` is the arbitrage equation.

The perturbation method consists in take an :math:`n^{th}` order Taylor approximation to the arbitrage equation, and solving for the :math:`n^{th}` order Taylor approximation to the control functions.

.. autofunction:: dolo.algos.dtcscc.perturbations.approximate_controls
