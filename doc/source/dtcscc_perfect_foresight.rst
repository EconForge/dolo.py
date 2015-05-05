Perfect foresight
.................


We consider an `fg` model, that is a model with in the form:

 :math:`s_t = g\left(s_{t-1}, x_{t-1}, \epsilon_t \right)`

 :math:`0 = E_t \left( f\left(s_{t}, x_{t}, s_{t+1}, x_{t+1}\right) \right) \ \perp \ \underline{u} <= x_t <= \overline{u}`

We assume that :math:`\underline{u}` and :math:`\overline{u}` are constants. This is not a big restriction since the model can always be reformulated in order to meet that constraint, by adding more equations.

Given a realization of the shocks :math:`(\epsilon_i)_{i>=1}` and an initial state :math:`s_0`, the perfect foresight
problem consists in finding the path of optimal controls  :math:`(x_t)_{t>=0}` and the corresponding
evolution of states :math:`(s_t)_{t>=0}`.

In practice, we find a solution over a finite horizon :math:`T>0` by assuming that the last state is constant forever.
The stacked system of equations satisfied by the solution is:




 +-------------------------------------------------+------------------------------------------------------------------------------+
 +-------------------------------------------------+------------------------------------------------------------------------------+
 |  :math:`s_0 = s_0`                              |   :math:`f(s_0, x_0, s_1, x_1) \perp \underline{u} <= x_0 <= \overline{u}`   |
 +-------------------------------------------------+-----------------+------------------------------------------------------------+
 |  :math:`s_1 = g(s_0, x_0, \epsilon_1)`          |   :math:`f(s_1, x_1, s_2, x_2) \perp \underline{u} <= x_1 <= \overline{u}`   |
 +-------------------------------------------------+------------------------------------------------------------------------------+
 |                                                 |                                                                              |
 +-------------------------------------------------+------------------------------------------------------------------------------+
 |  :math:`s_T = g(s_{T-1}, x_{T-1}, \epsilon_T)`  |   :math:`f(s_T, x_T, s_T, x_T) \perp \underline{u} <= x_T <= \overline{u}`   |
 +-------------------------------------------------+------------------------------------------------------------------------------+




.. autofunction:: dolo.algos.dtcscc.perfect_foresight.deterministic_solve
