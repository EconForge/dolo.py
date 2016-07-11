Parameterized expectations
..............

We consider an `fgh` model, that is a model with the form:

 :math:`s_t = g\left(s_{t-1}, x_{t-1}, \epsilon_t \right)`

 :math:`0 = f\left(s_{t}, x_{t}, E_t[h(s_{t+1}, x_{t+1})] \right)`

where :math:`g` is the state transition function, :math:`f` is the arbitrage equation, and :math:`h` is the expectations function (more accurately, it is the function over which expectations are taken).

The parameterized expectations algorithm consists in approximating the expectations function, :math:`h`, and solving for the associated optimal controls, :math:`x_t = x(s_t)`.

At step :math:`n`, with a current guess of the control, :math:`x(s_t) = \varphi^n(s_t)`, and expectations function, :math:`h(s_t,x_t) = \psi^n(s_t)` :
  - Compute the conditional expectation :math:`z_t = E_t[\varphi^n(s_t)]`
  - Given the expectation, update the optimal control by solving :math:`0 = \left( f\left(s_{t}, \varphi^{n+1}(s), z_t \right) \right)`
  - Given the optimal control, update the expectations function :math:`\psi^{n+1}(s_t) = h(s_t, \varphi^{n+1}(s_t))`


.. autofunction:: dolo.algos.dtcscc.parameterized_expectations.parameterized_expectations
