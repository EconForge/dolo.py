Perfect foresight
.................

Consider a series for the exogenous process :math:`(m_t)_{0 \leq t \leq T}`.
The perfect foresight problem consists in finding the path of optimal controls :math:`(x_t)_{0 \leq t \leq T}`
and corresponding states :math:`(s_t)_{0 \leq t \leq T}` such that:

 :math:`s_t = g\left(m_{t-1}, s_{t-1}, x_{t-1}, \epsilon_t \right)`

 :math:`0 = E_t \left( f\left(m_{t}, s_{t}, x_{t}, m_{t+1}, s_{t+1}, x_{t+1}\right) \right) \ \perp \ \underline{u} <= x_t <= \overline{u}`

Special conditions apply for the initial state and controls. Initial state :math:`s_0` is given exogenously.
Final states and controls are determined by assuming the exogenous process satisfies :math:`m_t=m_T` for all :math:`t\leq T` and optimality conditions are satisfied in the last period:

:math:`f(m_T, s_T, x_T, m_T,s_T, x_T) \perp \underline{u} <= x_T <= \overline{u}`.

We assume that :math:`\underline{u}` and :math:`\overline{u}` are constants. This is not a big restriction since the model can always be reformulated in order to meet that constraint, by adding more equations.

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

The system is solved using a nonlinear solver.

.. note:: TODO: explain the subtle timing convention converning the exogenous shock.

.. autofunction:: dolo.algos.perfect_foresight.deterministic_solve
