Perfect foresight algorithm
===========================

The perfect foresight algorithm is implemented for models of type "fg". Recall that ths type of model is specified by :math:`g` and :math:`f` such that:

:math:`s_t = g \left( s_{t-1}, x_{t-1}, \epsilon_t \right)`

:math:`E_t \left[ f \left( s_t, x_t, s_{t+1}, x_{t+1} \right) \right]=0`

In this exercise, the exogenous shocks are supposed to take a predetermined series of values :math:`(\epsilon_0,\epsilon_1,...,\epsilon_K)`. We assume :math:`\forall t<0, \epsilon_t=\epsilon_0` and :math:`\forall t>K, \epsilon_t=\epsilon_K`. 

We compute the transition of the economy specified by :math:`fg`, from an equilibrium with :math:`\epsilon=\epsilon_0` to an equilibrim with :math:`\epsilon=\epsilon_K`.

This transition happens under under perfect foresight, in the following sense. For :math:`t<0` agents, expect the economy to remain at its initial steady-state, but suddenly, at :math:`t=0`, they know the exact values, that innovations will take until the end of times.

.. autofunction:: dolo.algos.perfect_foresight.deterministic_solve
.. autofunction:: dolo.algos.perfect_foresight.find_steady_state
