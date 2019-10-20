Model Specification
===================


Variable types
--------------

The following types of variables can be used in DTCSCC models:

    -  ``shocks`` (``e``) (can be autocorrelated)
    -  ``states`` (``s``)
    -  ``controls`` (``x``)
    -  ``auxiliaries`` (``y``)
    -  ``rewards`` (``r``)
    -  ``values`` (``v``)
    -  ``expectations`` (``z``)
    -  ``parameters`` (``p``)

Symbol types that are present in a model are always listed in that order.

State-space
~~~~~~~~~~~

The unknown vector of controls :math:`x` is a function :math:`\varphi` of the states, both exogenous (:math`e`) and endogenous (:math`s`)
In general we have:

.. math::

    x = \varphi(e, s)

.. math::

    In case shocks are iid, dolo looks for a decision rule :math`x=\varphi(s)`/

The function :math:`\varphi` is typically approximated by the solution algorithm. It can be either a Taylor expansion, or an intepolating object (splines, smolyak). In both cases, it behaves like a numpy gufunc and can be called on a vector or a list of points:

.. code:: python

    # for an iid model
    dr = approximate_controls(model)
    s0 = model.calibration['states']
    dr(s0)                               # evaluates on a vector
    dr(s0[None,:].repeat(10, axis=0) )   # works on a list of points too


Valid equations
~~~~~~~~~~~~~~~

The various equations that can be defined using these symbol types is summarized in the following table. They are also reviewed below with more details.



+-----------------------------------+-------------------------------+------------+-----------------------------------------------+
|    Function                       | Standard name                 | Short name | Definition                                    |
+-----------------------------------+-------------------------------+------------+-----------------------------------------------+
| Transitions                       | ``transition``                | ``g``      | ``s = g(e(-1), s(-1),x(-1),e)``                      |
+-----------------------------------+-------------------------------+------------+-----------------------------------------------+
| Lower bound                       | ``controls_lb``               | ``lb``     | ``x_lb = lb(e, s)``                              |
+-----------------------------------+-------------------------------+------------+-----------------------------------------------+
| Upper bound                       | ``controls_ub``               | ``ub``     | ``x_ub = ub(e, s)``                              |
+-----------------------------------+-------------------------------+------------+-----------------------------------------------+
| Auxiliary  (special block)        | ``auxiliary``                 | ``a``      | ``y = a(e,s,x)``                                |
+-----------------------------------+-------------------------------+------------+-----------------------------------------------+
| Utility                           | ``utility``                   | ``u``      | ``r = u(e,s,x)``                                |
+-----------------------------------+-------------------------------+------------+-----------------------------------------------+
| Value updating                    | ``value_updating``            | ``w``      | ``v = w(s,x,v,s(1),x(1),w(1))``               |
+-----------------------------------+-------------------------------+------------+-----------------------------------------------+
| Arbitrage                         | ``arbitrage``                 | ``f``      | ``0=f(e,s,x,e(1),s(1),x(1))``                    |
+-----------------------------------+-------------------------------+------------+-----------------------------------------------+
| Expectations                      | ``expectation``               | ``h``      | ``z=h(s(1),x(1))``                            |
+-----------------------------------+-------------------------------+------------+-----------------------------------------------+
| Generalized expectations          | ``expectation_2``             | ``h_2``    | ``z=h_2(s,x,e(1),s(1),x(1))``                 |
+-----------------------------------+-------------------------------+------------+-----------------------------------------------+
| Arbitrage (explicit expectations) | ``arbitrage_2``               | ``f_2``    | ``0=f_2(s,x,z)``                              |
+-----------------------------------+-------------------------------+------------+-----------------------------------------------+
| Direct response                   | ``direct_response``           | ``d``      | ``x=d(s,z)``                                  |
+-----------------------------------+-------------------------------+------------+-----------------------------------------------+
| Terminal conditions               | ``terminal``                  | ``f_T``    | ``0=f_T(s,x)``                                |
+-----------------------------------+-------------------------------+------------+-----------------------------------------------+
| Explicit terminal conditions      | ``direct_terminal``           | ``d_T``    | ``s=d_T(s)``                                  |
+-----------------------------------+-------------------------------+------------+-----------------------------------------------+

When present these functions can be accessed from the ``model.functions`` dictionary by using the standard name. For instance to compute the auxiliary variables at the steady-state one can compute:

.. code:: python

    # recover steady-state values
    e = model.calibration['exogenous']
    s = model.calibration['states']
    x = model.calibration['controls']
    p = model.calibration['parameters']

    # compute the vector of auxiliary variables
    a = model.functions['auxiliary']
    y = a(e,s,x,p)

    # it should correspond to the calibrated values:
    calib_y = model.calibration['auxiliaries']
    assert( abs(y - calib_y).max() < 0.0000001 )


Transitions
...........

::

    - name: `transition`
    - short name: `g`

Transitions are given by a function :math:`g` such that at all times:

.. math::

    s_t = g(s_{t-1}, x_{t-1}, \epsilon_t)

where :math:`\epsilon_t` is a vector of i.i.d. shocks.

.. note::

    In the RBC model, the vector of states is :math:`s_t=(a_t,k_t)`.
    The transitions are:

        .. math::

            a_t = \rho a_{t-1} + \epsilon_t
            k_t = (1-\delta)*k_{t-1} + i_{t-1}


    The yaml file is amended with:

    .. code:: yaml

        symbols:
            states: [a,k]
            controls: [i]
            shocks: [epsilon]
            ...
        equations:
            transition:
                a = rho*a(-1) + e
                k = k(-1)*(1-delta) + i(-1)

    Note that the transitions are given in the declaration order.

Auxiliary variables
...................

::

    - name: `auxiliary`
    - short name: `a`

In order to reduce the number of variables, it is useful to define auxiliary variables :math:`y_t` using a function :math:`a` such that:

.. math::

    y_t = a(s_t, x_t)

.. note

    Auxiliaries are now defined in the `definitions` block, separately from other equations.

When they appear in an equation they are automatically substituted by
the corresponding expression in :math:`s_t` and :math:`x_t`.
Note that auxiliary variables are not explicitely listed in the following definition. Implicitly, wherever states and controls are allowed with the same date in an equation type, then auxiliary variable are also allowed with the same date.

.. note::

    In the RBC model, three auxiliary variables are defined :math:`y_t, c_t, r_{k,t}` and :math:`w_t`. They are a closed form function of :math:`a_t, k_t, i_t, n_t`. Defining these variables speeds up computation since they are don't need to be solved for or interpolated.



Utility function and Bellman equation
.....................................

::

    - name: `utility`
    - short name: `u`

The (separable) value equation defines the value :math:`v_t` of a given policy as:

.. math::

    v_t = u(s_t,x_t) + \beta E_t \left[ v_{t+1} \right]

This gives rise to the Bellman equation:

    .. math::

        v_t = \max_{x_t} \left( u(s_t,x_t) + \beta E_t \left[ v_{t+1} \right] \right)

These two equations are characterized by the reward function :math:`u` and the discount rate :math:`\beta`. Function :math:`u` defines the vector of symbols ``rewards``.
Since the definition of :math:`u` alone is not sufficient, the parameter used for the discount factor must be given to routines that compute the value. Several values can be computed at once, if :math:`U` is a vector function and :math:`\beta` a vector of discount factors, but in that case in cannot be used to solve for the Bellman equation.

.. note::

    Our RBC example defines the value as :math:`v_t = \frac{(c_t)^{1-\gamma}}{1-\gamma} + \beta E_t v_{t+1}`. This information is coded using:   ## TODO add labour to utility

    .. code:: yaml

        symbols:
            ...
            rewards: [r]

        equations:
            ...
            utility:
                - r = c^(1-gamma)/(1-gamma)

        calibration:
            ...
            beta: 0.96   # beta is the default name of the discount


Value
.....

::

    - name: `value`
    - short name: `w`

A more general updating equation can be useful to express non-separable utilities or prices.  the vector of (generalized) values :math:`v^{*}` are defined by a function ``w`` such that:

.. math::

    v_t = w(s_t,x_t,v_t,s_{t+1},x_{t+1},v_{t+1})

As in the separable case, this function can either be used to compute the value of a given policy :math:`x=\varphi()` or in order solve the generalized Bellman equation:

.. math::

    v_t = \max_{x_t} \left( w(s_t,x_t,v_t,s_{t+1},x_{t+1},v_{t+1}) \right)


.. note::

    Instead of defining the rewards of the RBC example, one can instead define a value updating equation instead:

    .. code:: yaml

        symbols:
            ...
            values: [v]

        equations:
            ...
            value:
                - v = c^(1-gamma)/(1-gamma)*(1-n...) + beta*v(1)



Boundaries
..........

::

    - name: `controls_lb` and `controls_ub`
    - short name: `lb` and `ub`

The optimal controls must also satisfy bounds that are function of states. There are two functions :math:`\underline{b}()` and :math:`\overline{b}()` such that:

.. math::

    \underline{b}(e_t, s_t) \leq x_t \leq \overline{b}(s_t)

.. note::

    In our formulation of the RBC model we have excluded negative investment, implying :math:`i_t \geq 0`. On the other hand, labour cannot be negative so that we add lower bounds to the model:

    .. code:: yaml

        equations:
            ...
            controls_lb:
                i = 0
                n = 0

    Specifying the lower bound on labour actually has no effect since agents endogeneously choose to work a positive amount of time in order to produce some consumption goods.
    As for upper bounds, it is not necessary to impose some: the maximum amount of investment is limited by the Inada conditions on consumption. As for labour ``n``, it can be arbitrarily large without creating any paradox. Thus the upper bounds are omitted (and internally treated as infinite values).

Euler equation
..............

::

    - name: `arbitrage` (`equilibrium`)
    - short name: `f`

A general formulation of the Euler equation is:

.. math::

    0 = E_t \left[ f(s_t, x_t, s_{t+1}, x_{t+1}) \right]

Note that the Euler equation and the boundaries interact via
"complementarity equations". Evaluated at one given state, with
the vector of controls :math:`x=(x_1, ..., x_i, ..., x_{n_x})`, the
Euler equation gives us the residuals :math:`r=(f_1, ..., f_i, ...,
f_{n_x})`.
Suppose that the :math:`i`-th control :math:`x_i` is supposed to lie in the
interval
:math:`[ \underline{b}_i, \overline{b}_i ]`. Then one of the following
conditions
must be true:

-  :math:`f_i` = 0
-  :math:`f_i<0` and :math:`x_i=\overline{b}_i`
-  :math:`f_i>0` and :math:`x_i=\underline{b}_i`


By definition, this set of conditions is denoted by:

-  :math:`f_i = 0 \perp \underline{b}_i \leq x_i \leq \overline{b}_i`

These notations extend to a vector setting so that the Euler
equations can also be written:

.. math::

    0 = E_t \left[ f(s_t, x_t, s_{t+1}, x_{t+1}) \right] \perp \underline{b}(s_t) \leq x_t \leq \overline{b}(s_t)

Specifying the boundaries together with Euler equation, or providing them separately is exactly equivalent. In any case, when the boundaries are finite and occasionally binding, some attention should be devoted to write the Euler equations in a consistent manner. In particular, note that the Euler equations are order-sensitive.

The Euler conditions, together with the complementarity conditions typically often come from Kuhn-Tucker conditions associated with the Bellman problem, but that is not true in general.

.. note::

    The RBC model has two Euler equations associated with investment and labour supply respectively. They are added to the model as:

    .. code:: yaml

        arbitrage:
            - 1 - beta*(c/c(1))^(sigma)*(1-delta+rk(1))   | 0 <= i <= inf
            - w - chi*n^eta*c^sigma                       | 0 <= n <= inf

    Putting the complementarity conditions close to the Euler equations, instead of entering them as separate equations, helps to check the sign of the Euler residuals when constraints are binding. Here, when investment is less desirable, the first expression becomes bigger. When the representative is prevented to invest less due to the constraint (i.e. :math:`i_t=0`), the expression is then *positive* consistently with the complementarity conventions.


Expectations
............

::

    - name: `expectation`
    - short name: `h`

The vector of explicit expectations :math:`z_t` is defined by a function  :math:`h` such that:

.. math::

    z_t = E_t \left[ h(s_{t+1},x_{t+1}) \right]

.. code::

    In the RBC example, one can define. the expected value tomorrow of one additional unit invested tomorrow:

    .. math::

        m_t=\beta*(c_{t+1}^(-\sigma)*(1-\delta+r_{k,t+1})

     It is a pure expectational variable in the sense that it is solely determined by future states and decisions. In the model file, it would be defined as:

    .. code: yaml

        symbols:
            ...
            expectations: [z]

        equations:
            ...
            - z = beta*(c(1))^(-sigma)*(1-delta+rk(1))


Generalized expectations
........................

::

    - name: `expectation_2`
    - short name: `h_2`

The vector of generalized explicit expectations :math:`z_t` is defined by a function :math:`h^{\star}` such that:

.. math::

    z_t = E_t \left[ h^{\star}(s_t,x_t,\epsilon_{t+1},s_{t+1},x_{t+1}) \right]

Euler equation with expectations
.....................................

::

    - name: `arbitrage_2` (`equilibrium_2`)
    - short name: `f_2`

If expectations are defined using one of the two preceding
definitions, the Euler equation can be rewritten as:

.. math::

    0 = f(s_t, x_t, z_t) \perp \underline{b}(s_t) \leq x_t \leq \overline{b}(s_t)

.. note::

    Given the definition of the expectation variable :math:`m_t`, today's consumption is given by: :math:`c_t = z_t^(-\frac{1}{sigma})` so the Euler equations are rewritten as:


    .. code:: yaml

        arbitrage_2:
            - 1 - beta*(c)^(sigma)/m   | 0 <= i <= inf
            - w - chi*n^eta*c^sigma    | 0 <= n <= inf

    Note the type of the arbitrage equation (``arbitrage_2`` instead of ``arbitrage``).

    However :math:`c_t` is not a control itself,




     but the controls :math:`i_t, n_t` can be easily deduced:

    ..math::

        n_t = ((1-alpha)*z_t*k_t^alpha*m_t/chi)^(1/(eta+alpha))
        i_t = z_t*k_t^\alpha*n_t^(1-\alpha) - (m_t)^(-1/sigma)

    This translates into the following YAML code:

    .. code:: yaml

        equations:
            - n = ((1-alpha)*a*k^alpha*m/chi)^(1/(eta+alpha))
            - i = z*k^alpha*n^(1-alpha) - m^(-1/sigma)




Direct response function
........................

::

    - name: `direct_response`
    - short name: `d`

In some simple cases, there a function :math:`d()` giving an explicit
definition of the controls:

.. math::

    x_t = d(s_t, z_t)

Compared to the preceding Euler equation, this formulation saves
computational time by removing the need to solve a nonlinear system to recover the controls implicitly defined by the Euler equation.

Terminal conditions
...................

::

    - name: `terminal_condition`
    - short name: `f_T`

When solving a model over a finite number :math:`T` of periods, there must
be a terminal condition defining the controls for the last period.
This is a function :math:`f^T` such that:

.. math::

    0 = f^T(s_T, x_T)

Terminal conditions
...................

::

    - name: `terminal_condition`
    - short name: `f_T_2`

Sometimes the terminal condition is given as an explicit choice for the controls in the last period. This defines function :math:`f^{T,\star}` such that:

.. math::

    x_T = f^{T,\star}(s_T)





..
..
..
.. Discrete Time - Mixed States - Continuous Controls models (DTMSCC)
.. ------------------------------------------------------------------
..
.. The definitions for this class of models differ from the former ones
.. by the fact that states are split into exogenous and discrete markov states,
.. and endogenous continuous states as before. Most of the definition can be readily
.. transposed by replacing only the state variables.
..
.. State-space and solution
.. ~~~~~~~~~~~~~~~~~~~~~~~~
..
.. For this kind of problem, the state-space, is the cartesian product
.. of a vector of "markov states" :math:`m_t` that can take a finite number of
.. values and a vector of "continuous states" :math:`s_t` which takes
.. continuous values.
..
.. The unknown controls :math:`x_t` is a function :math:`\varphi` such that:
..
.. .. math::
..
..     x_t =\varphi (m_t, s_t)
..
.. Transitions
.. ~~~~~~~~~~~
..
.. ::
..
..     - name: `transition`
..     - short name: `g`
..
.. :math:`(m_t)` follows an exogenous and discrete markov chain.
.. The whole markov chain is specified by two matrices :math:`P,Q` where each
.. line of :math:`P` is one admissible value for :math:`m_t` and where each element
.. :math:`Q(i,j)` is the conditional probability to go from state :math:`i` to state :math:`j`.
..
.. The continuous states :math:`s_t` evolve after the law of motion:
..
.. .. math::
..
..     s_t = g(m_{t-1}, s_{t-1}, x_{t-1}, m_t)
..
..
.. Boundaries
.. ~~~~~~~~~~
..
.. ::
..
..     - name: `controls_lb`, `controls_ub`
..     - short name: `lb`, `ub`
..
.. The optimal controls must satisfy bounds that are function of states.
.. There are two functions :math:`\underline{b}()`
.. and :math:`\overline{b}()` such that:
..
.. .. math::
..
..     \underline{b}(m_t, s_t) \leq x_t \leq \overline{b}(m_t, s_t)
..
.. Value Equation
.. ~~~~~~~~~~~~~~
..
.. ::
..
..     - name: `value`
..     - short name: `v`
..
.. The (separable) Bellman equation defines a value :math:`v_t` as:
..
.. .. math::
..
..     v_t = U(m_t,s_t,x_t) + \beta E_t \left[v_{t+1}\right]
..
.. It is completely characterized by the reward function :math:`U` and
.. the discount rate :math:`\beta`.
..
.. Generalized Value Equation
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~
..
.. ::
..
..     - name: `value_2`
..     - short name: `v_2`
..
.. The generalized value equation defines a value :math:`v^{\star}_t` as:
..
.. .. math::
..
..     :math:`v^{\star}_t = U^{\star}(m_t,s_t,x_t,v^{\star},m_{t+1},s_{t+1},x_{t+1})`
..
.. Euler equation
.. ~~~~~~~~~~~~~~
..
.. ::
..
..     - name: `arbitrage` (`equilibrium`)
..     - short name: `f`
..
.. Many Euler equations can be defined a function :math:`f` such that:
..
.. .. math::
..
..     0 = E_t \left( f(m_t,s_t,x_t,m_{t+1},s_{t+1},x_{t+1})
..     \right) \perp \underline{b}(m_t, s_t) \leq x_t \leq
..     \overline{b}(m_t, s_t)
..
.. See discussion about complementarity equations in the Continuous States
.. - Continuous Controls section.
..
.. Expectations
.. ~~~~~~~~~~~~
..
.. ::
..
..     - name: `expectation`
..     - short name: `h`
..
.. The vector of explicit expectations :math:`z_t` is defined by a function :math:`h` such that:
..
.. .. math::
..
..     z_t = E_t \left[ h(m_{t+1},s_{t+1},x_{t+1}) \right]
..
.. Generalized expectations
.. ~~~~~~~~~~~~~~~~~~~~~~~~
..
.. ::
..
..     - name: `expectation_2`
..     - short name: `h_2`
..
.. The vector of generalized explicit expectations :math:`z_t` is defined by a
.. function :math:`h^{\star}` such that:
..
.. .. math::
..
..     z_t = E_t \left[ h^{\star}(m_t,s_t,x_t,m_{t+1},s_{t+1},x_{t+1}) \right]
..
.. Euler equation with explicit equations
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
..
.. ::
..
..     - name: `arbitrage_2` (`equilibrium_2`)
..     - short name: `f_2`
..
.. If expectations are defined using one of the two preceding
.. definitions, the Euler equation can be rewritten as:
..
.. .. math::
..
..     0 = f(m_t, s_t, x_t, z_t) \perp \underline{b}(s_t) \leq x_t \leq \overline{b}(s_t)
..
.. Direct response function
.. ~~~~~~~~~~~~~~~~~~~~~~~~
..
.. ::
..
..     - name: `direct_response`
..     - short name: `d`
..
.. In some simple cases, there a function :math:`d()` giving an explicit
.. definition of the controls:
..
.. .. math::
..
..     x_t = d(s_t, z_t)
..
.. Compared to the preceding Euler equation, this formulation saves
.. computational time by removing to solve a nonlinear to get the controls implicitly
.. defined by the Euler equation.
..
.. Direct states function
.. ~~~~~~~~~~~~~~~~~~~~~~
..
.. ::
..
..     - name: `direct_states`
..     - short name: `d_s`
..
.. For some applications, it is also useful to have a function
.. :math:`d{\star}` which gives the endogenous states as a function of the controls and
.. the exogenous markov states:
..
.. .. math::
..
..     s_t = d^{\star}(m_t, x_t)
..
.. Auxiliary variables
.. ~~~~~~~~~~~~~~~~~~~
..
.. ::
..
..     - name: `auxiliary`
..     - short name: `a`
..
.. In order to reduce the number of variables, it is useful to define
.. auxiliary variables :math:`y_t$ using a function $a` such that:
..
.. .. math::
..
..     y_t = a(m_t,s_t, x_t)
..
.. Terminal conditions
.. ~~~~~~~~~~~~~~~~~~~
..
.. ::
..
..     - name: `terminal_control`
..     - short name: `f_T`
..
.. When solving a model over a finite number :math:`T` of periods, there must
.. be a terminal condition defining the controls for the last period.
.. This is a function :math:`f^T` such that:
..
.. .. math::
..
..     x_T = f^T(m_T, s_T)
..
.. Terminal conditions (explicit)
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
..
.. ::
..
..     - name: `terminal_control`
..     - short name: `f_T_2`
..
.. When solving a model over a finite number :math:`T` of periods, there must
.. be a terminal condition defining the controls for the last period.
.. This is a function :math:`f^{T,\star}` such that:
..
.. .. math::
..
..     f^{T,\star}(m_T, s_T, x_T)
..
..
..
.. Misc
.. ----
..
.. Variables
.. ~~~~~~~~~
..
.. For DTCSCC and DTMSCC models, the following list variable types can be
.. used (abbreviation in parenthesis):
.. Required:
..
.. -  ``states`` (``s``)
.. -  ``controls`` (``x``)
..    For DTCSCC only:
.. -  ``shocks`` (``e``)
..    For DTMSCC only:
.. -  ``markov_states`` (``m``)
..    Optional:
.. -  ``auxiliaries`` (``y``)
.. -  ``values`` (``v``)
.. -  ``values_2`` (``v_2``)
.. -  ``expectations`` (``z``)
.. -  ``expectations_2`` (``z_2``)
..
.. Algorithms
.. ~~~~~~~~~~
..
.. Several algorithm are available to solve a model,
.. depending no the functions that are specified.
..
.. +----------------------------------+----------------+-----------------+-----------------+
.. |                                  | Dynare model   | DTCSCC          | DTMSCC          |
.. +==================================+================+=================+=================+
.. | Perturbations                    | yes            | (f,g)           | no              |
.. +----------------------------------+----------------+-----------------+-----------------+
.. | Perturbations (higher order)     | yes            | (f,g)           | no              |
.. +----------------------------------+----------------+-----------------+-----------------+
.. | Value function iteration         |                | (v,g)           | (v,g)           |
.. +----------------------------------+----------------+-----------------+-----------------+
.. | Time iteration                   |                | (f,g),(f,g,h)   | (f,g),(f,g,h)   |
.. +----------------------------------+----------------+-----------------+-----------------+
.. | Parameterized expectations       |                | (f,g,h)         | (f,g,h)         |
.. +----------------------------------+----------------+-----------------+-----------------+
.. | Parameterized expectations (2)   |                | (f_2,g,h_2)     | (f_2,g,h_2)     |
.. +----------------------------------+----------------+-----------------+-----------------+
.. | Parameterized expectations (3)   |                | (d,g,h)         | (d,g,h)         |
.. +----------------------------------+----------------+-----------------+-----------------+
.. | Endogeneous gridpoints           |                |                 | (d,d_s,g,h)     |
.. +----------------------------------+----------------+-----------------+-----------------+
..
.. Additional informations
.. -----------------------
..
.. calibration
.. ~~~~~~~~~~~
..
.. In general, the models will depend on a series of scalar parameters.
.. A reference value for the endogeneous variables is also used, for
.. instance to define the steady-state. We call a "calibration" a list of values
.. for all parameters and steady-state.
..
.. state-space
.. ~~~~~~~~~~~
..
.. When a global solution is computed, continuous states need to be
.. bounded.
.. This can be done by specifying an n-dimensional box for them.
..
.. Usually one also want to specify a finite grid, included in this grid
.. and the interpolation method used to evaluate between the grid points.
..
.. specification of the shocks
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~
..
.. For DTCSCC models, the shocks follow an i.i.d. series of random
.. variables.
.. If the shock is normal, this one is characterized by a covariance
.. matrix.
..
.. For DTMSCC models, exogenous shocks are specified by a two matrices P
.. and Q,
.. containing respectively a list of nodes and the transition
.. probabilities.
..
.. Remarks
.. ~~~~~~~
..
.. Some autodetection is possible. For instance, some equations appearing
.. in
.. ``f`` functions, can be promoted (or downgraded) to expectational
.. equation, based
.. on incidence analysis.
