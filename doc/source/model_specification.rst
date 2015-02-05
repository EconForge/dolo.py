Dolo Model Classification
=========================

Dynare models
-------------

A Dynare model is specified by a set of equations captured by
one vector function :math:`f` so that the vector of endogeneous states
:math:`y_t` satisfies

.. math::

    0 = E_t f(y_{t-1}, y_t, y_{t+1}, \epsilon_t)

where :math:`\epsilon_t` is a vector of exogenous variables
following an i.i.d. series of shocks.

In this formulation, any variable appearing at date :math:`t-1` in the
system (a predetermined variable) is part of the state-space as
well as all shocks. Hence, the solution of the system is
a function :math:`g` such that:

.. math::

    y_t = f(y_{t-1}, \epsilon_t)

Discrete Time - Continuous States - Continuous Controls models (DTCSCC)
-----------------------------------------------------------------------

Solution
~~~~~~~~

State-space is characterized by a vector :math:`s` of continuous variables.
The unknown vector of controls :math:`x` is expressed by a function
:math:`\varphi`
such that:

.. math::

    x = \varphi(s)

Transitions
~~~~~~~~~~~

::

    - name: `transition`
    - short name: `g`

Transitions are given by a function :math:`g` such that at all times:

.. math::

    s_t = g(s_{t-1}, x_{t-1}, \epsilon_t)

where :math:`\epsilon_t` is a vector of i.i.d. shocks.

Value equation
~~~~~~~~~~~~~~

::

    - name: `value`
    - short name: `v`

The (separable) value equation defines a value :math:`v_t` as:

.. math::

    v_t = U(s_t,x_t) + \beta E_t \left[ v_{t+1} \right]

In general the Bellman equations are completely characterized by the
reward function :math:`U` and
the discount rate :math:`\beta`.

Note that several values can be computed at once, if :math:`U` is a vector
function
and :math:`\beta` a vector of discount factors.

Generalized Value Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    - name: `value_2`
    - short name: `v_2`

A more general updating equation can be useful to express
non-separable utilities or prices.
In that case, we define a function :math:`U^{*}` such that

.. math::

    v^{*}_t = U^{*}(s_t,x_t,v^{*}_t,s_{t+1},x_{t+1},v^{*}_{t+1})

This equation defines the vector of (generalized) values :math:`v^{*}`

Boundaries
~~~~~~~~~~

::

    - name: `controls_lb` and `controls_ub`
    - short name: `lb` and `ub`

The optimal controls must also satisfy bounds
that are function of states. There are two functions
:math:`\underline{b}()` and :math:`\overline{b}()` such that:

.. math::

    \underline{b}(s_t) \leq x_t \leq \overline{b}(s_t)

Euler equation
~~~~~~~~~~~~~~

::

    - name: `arbitrage` (`equilibrium`)
    - short name: `f`

A general formulation of the Euler equation is:

.. math::

    0 = E_t [ f(s_t, x_t, s_{t+1}, x_{t+1}) ]

Note that the Euler equation and the boundaries interact via
"complentarity equations". Evaluated at one given state, with
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

-  :math:`f_i = 0 \perp \underline{b}_i \leq x*\ i \leq \overline{b}_i`

These notations extend to a vector setting so that the Euler
equations can also be written:

.. math::

    0 = E_t [ f(s_t, x_t, s_{t+1}, x_{t+1}) ] \perp \underline{b}(s_t) \leq x_t \leq \overline{b}(s_t)

Specifying the boundaries together with Euler equation, or specifying
them as independent equations is (read: should be) equivalent.
In any case, when the boundaries are finite and occasionally binding,
some attention should be devoted to write the Euler equations in a
consistent manner.
In particular, note, that the Euler equations are order-sensitive.

The Euler conditions, together with the complementarity conditions
typically come from the Kuhn-Tucker conditions associated
with the maximization problem, but that is not true in general.

Expectations
~~~~~~~~~~~~

::

    - name: `expectation`
    - short name: `h`

    The vector of explicit expectations :math:`z_t` is defined by a function  :math:`h` such that:

.. math::

    z_t = E_t \left[ h(s_{t+1},x_{t+1}) \right]

Generalized expectations
~~~~~~~~~~~~~~~~~~~~~~~~

::

    - name: `expectation_2`
    - short name: `h_2`

The vector of generalized explicit expectations :math:`z_t` is defined by a
function :math:`h^{\star}` such that:

.. math::

    z_t = E_t \left[ h^{\star}(s_t,x_t,\epsilon_{t+1},s_{t+1},x_{t+1}) \right]

Euler equation with explicit equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    - name: `arbitrage_2` (`equilibrium_2`)
    - short name: `f_2`

If expectations are defined using one of the two preceding
definitions,
the Euler equation can be rewritten as:

.. math::

    0 = f(s_t, x_t, z_t) \perp \underline{b}(s_t) \leq x_t \leq \overline{b}(s_t)

Direct response function
~~~~~~~~~~~~~~~~~~~~~~~~

::

    - name: `direct_response`
    - short name: `d`

In some simple cases, there a function :math:`d()` giving an explicit
definition of the controls:

.. math::

    x_t = d(s_t, z_t)

Compared to the preceding Euler equation, this formulation saves
computational time by removing to solve a nonlinear to get the controls implicitly
defined by the Euler equation.

Terminal conditions
~~~~~~~~~~~~~~~~~~~

::

    - name: `terminal_control`
    - short name: `f_T`

When solving a model over a finite number :math:`T` of periods, there must
be a terminal condition defining the controls for the last period.
This is a function :math:`f^T` such that:

.. math::

    0 = f^T(s_T, x_T)

Terminal conditions
~~~~~~~~~~~~~~~~~~~

::

    - name: `terminal_control_2`
    - short name: `f_T_2`

When solving a model over a finite number :math:`T` of periods, there must
be a terminal condition defining the controls for the last period.
This is a function :math:`f^{T,\star}` such that:

.. math::

    x_T = f^{T,\star}(s_T)

Auxiliary variables
~~~~~~~~~~~~~~~~~~~

::

    - name: `auxiliary`
    - short name: `a`

In order to reduce the number of variables, it is useful to define
auxiliary variables :math:`y_t` using a function :math:`a` such that:

.. math::

    y_t = a(s_t, x_t)

When they appear in an equation they are automatically substituted by
the corresponding expression in :math:`s_t` and :math:`x_t`.

Discrete Time - Mixed States - Continuous Controls models (DTMSCC)
------------------------------------------------------------------

The definitions for this class of models differ from the former ones
by the fact that states are split into exogenous and discrete markov states,
and endogenous continous states as before. Most of the definition can be readily
transposed by replacing only the state variables.

State-space and solution
~~~~~~~~~~~~~~~~~~~~~~~~

For this kind of problem, the state-space, is the cartesian product
of a vector of "markov states" :math:`m_t` that can take a finite number of
values and a vector of "continuous states" :math:`s_t` which takes
continuous values.

The unknown controls :math:`x_t` is a function :math:`\varphi` such that:

.. math::

    x_t =\varphi (m_t, s_t)

Transitions
~~~~~~~~~~~

::

    - name: `transition`
    - short name: `g`

:math:`(m_t)` follows an exogenous and discrete markov chain.
The whole markov chain is specified by two matrices :math:`P,Q` where each
line of :math:`P` is one admissible value for :math:`m_t` and where each element
:math:`Q(i,j)` is the conditional probability to go from state :math:`i` to state :math:`j`.

The continuous states :math:`s_t` evolve after the law of motion:

.. math::

    s_t = g(m_{t-1}, s_{t-1}, x_{t-1}, m_t)


Boundaries
~~~~~~~~~~

::

    - name: `controls_lb`, `controls_ub`
    - short name: `lb`, `ub`

The optimal controls must satisfy bounds that are function of states.
There are two functions :math:`\underline{b}()`
and :math:`\overline{b}()` such that:

.. math::

    \underline{b}(m_t, s_t) \leq x_t \leq \overline{b}(m_t, s_t)

Value Equation
~~~~~~~~~~~~~~

::

    - name: `value`
    - short name: `v`

The (separable) Bellman equation defines a value :math:`v_t` as:

.. math::

    v_t = U(m_t,s_t,x_t) + \beta E_t \left[v_{t+1}\right]

It is completely characterized by the reward function :math:`U` and
the discount rate :math:`\beta`.

Generalized Value Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    - name: `value_2`
    - short name: `v_2`

The generalized value equation defines a value :math:`v^{\star}_t` as:

.. math::

    :math:`v^{\star}_t = U^{\star}(m_t,s_t,x_t,v^{\star},m_{t+1},s_{t+1},x_{t+1})`

Euler equation
~~~~~~~~~~~~~~

::

    - name: `arbitrage` (`equilibrium`)
    - short name: `f`

Many Euler equations can be defined a function :math:`f` such that:

.. math::

    0 = E_t \left( f(m_t,s_t,x_t,m_{t+1},s_{t+1},x_{t+1})
    \right) \perp \underline{b}(m_t, s_t) \leq x_t \leq
    \overline{b}(m_t, s_t)

See discussion about complementarity equations in the Continuous States
- Continuous Controls section.

Expectations
~~~~~~~~~~~~

::

    - name: `expectation`
    - short name: `h`

The vector of explicit expectations :math:`z_t` is defined by a function :math:`h` such that:

.. math::

    z_t = E_t \left[ h(m_{t+1},s_{t+1},x_{t+1}) \right]

Generalized expectations
~~~~~~~~~~~~~~~~~~~~~~~~

::

    - name: `expectation_2`
    - short name: `h_2`

The vector of generalized explicit expectations :math:`z_t` is defined by a
function :math:`h^{\star}` such that:

.. math::

    z_t = E_t \left[ h^{\star}(m_t,s_t,x_t,m_{t+1},s_{t+1},x_{t+1}) \right]

Euler equation with explicit equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    - name: `arbitrage_2` (`equilibrium_2`)
    - short name: `f_2`

If expectations are defined using one of the two preceding
definitions, the Euler equation can be rewritten as:

.. math::

    0 = f(m_t, s_t, x_t, z_t) \perp \underline{b}(s_t) \leq x_t \leq \overline{b}(s_t)

Direct response function
~~~~~~~~~~~~~~~~~~~~~~~~

::

    - name: `direct_response`
    - short name: `d`

In some simple cases, there a function :math:`d()` giving an explicit
definition of the controls:

.. math::

    x_t = d(s_t, z_t)

Compared to the preceding Euler equation, this formulation saves
computational time by removing to solve a nonlinear to get the controls implicitly
defined by the Euler equation.

Direct states function
~~~~~~~~~~~~~~~~~~~~~~

::

    - name: `direct_states`
    - short name: `d_s`

For some applications, it is also useful to have a function
:math:`d{\star}` which gives the endogenous states as a function of the controls and
the exogenous markov states:

.. math::

    s_t = d^{\star}(m_t, x_t)

Auxiliary variables
~~~~~~~~~~~~~~~~~~~

::

    - name: `auxiliary`
    - short name: `a`

In order to reduce the number of variables, it is useful to define
auxiliary variables :math:`y_t$ using a function $a` such that:

.. math::

    y_t = a(m_t,s_t, x_t)

Terminal conditions
~~~~~~~~~~~~~~~~~~~

::

    - name: `terminal_control`
    - short name: `f_T`

When solving a model over a finite number :math:`T` of periods, there must
be a terminal condition defining the controls for the last period.
This is a function :math:`f^T` such that:

.. math::

    x_T = f^T(m_T, s_T)

Terminal conditions (explicit)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    - name: `terminal_control`
    - short name: `f_T_2`

When solving a model over a finite number :math:`T` of periods, there must
be a terminal condition defining the controls for the last period.
This is a function :math:`f^{T,\star}` such that:

.. math::

    f^{T,\star}(m_T, s_T, x_T)

Misc
----

Variables
~~~~~~~~~

For DTCSCC and DTMSCC models, the following list variable types can be
used (abbreviation in parenthesis):
Required:

-  ``states`` (``s``)
-  ``controls`` (``x``)
   For DTCSCC only:
-  ``shocks`` (``e``)
   For DTMSCC only:
-  ``markov_states`` (``m``)
   Optional:
-  ``auxiliaries`` (``y``)
-  ``values`` (``v``)
-  ``values_2`` (``v_2``)
-  ``expectations`` (``z``)
-  ``expectations_2`` (``z_2``)

Algorithms
~~~~~~~~~~

Several algorithm are available to solve a model,
depending no the functions that are specified.

+----------------------------------+----------------+-----------------+-----------------+
|                                  | Dynare model   | DTCSCC          | DTMSCC          |
+==================================+================+=================+=================+
| Perturbations                    | yes            | (f,g)           | no              |
+----------------------------------+----------------+-----------------+-----------------+
| Perturbations (higher order)     | yes            | (f,g)           | no              |
+----------------------------------+----------------+-----------------+-----------------+
| Value function iteration         |                | (v,g)           | (v,g)           |
+----------------------------------+----------------+-----------------+-----------------+
| Time iteration                   |                | (f,g),(f,g,h)   | (f,g),(f,g,h)   |
+----------------------------------+----------------+-----------------+-----------------+
| Parameterized expectations       |                | (f,g,h)         | (f,g,h)         |
+----------------------------------+----------------+-----------------+-----------------+
| Parameterized expectations (2)   |                | (f_2,g,h_2)     | (f_2,g,h_2)     |
+----------------------------------+----------------+-----------------+-----------------+
| Parameterized expectations (3)   |                | (d,g,h)         | (d,g,h)         |
+----------------------------------+----------------+-----------------+-----------------+
| Endogeneous gridpoints           |                |                 | (d,d_s,g,h)     |
+----------------------------------+----------------+-----------------+-----------------+

Additional informations
-----------------------

calibration
~~~~~~~~~~~

In general, the models will depend on a series of scalar parameters.
A reference value for the endogeneous variables is also used, for
instance to define the steady-state. We call a "calibration" a list of values
for all parameters and steady-state.

state-space
~~~~~~~~~~~

When a global solution is computed, continuous states need to be
bounded.
This can be done by specifying an n-dimensional box for them.

Usually one also want to specify a finite grid, included in this grid
and the interpolation method used to evaluate between the grid points.

specification of the shocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For DTCSCC models, the shocks follow an i.i.d. series of random
variables.
If the shock is normal, this one is characterized by a covariance
matrix.

For DTMSCC models, exogenous shocks are specified by a two matrices P
and Q,
containing respectively a list of nodes and the transition
probabilities.

Remarks
~~~~~~~

Some autodetection is possible. For instance, some equations appearing
in
``f`` fonctions, can be promoted (or downgraded) to expectational
equation, based
on incidence analysis.
