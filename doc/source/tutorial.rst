Quick tutorial
==============

Here we illustrate how to solve the RBC model.


Write a model
-------------

Models are defined in YAML, which is a very readable standard for coding native data structures (see http://yaml.org/). This makes the model definition file quite easy to read. Take a look at the ``rbc.yaml`` from the ``examples/models`` directory. It is a valid YAML file. In particular, indentation defines nesting, colons define key-value associations that generate Python dicts, dashes generate Python lists, and the file must not contain any tabs. Here is its content:

.. literalinclude:: ../../examples/models/rbc.yaml
    :language: yaml
    :linenos:

It consists in several part:

1. First the model type is given (here ``fga``). This instructs dolo about the kind of model it has to expect.
   More information on model types (to-be-done).

2. The ``declarations`` block contains all names of variables/parameters to be used in the model. Here the model contains several
   kinds of variables: ``controls``, ``states`` and ``auxiliaries`` (which are are basically definitions
   that can be substituted everywhere else). There are also exogenously distributed innovations named ``shocks``
   and ``parameters``.

3. The model part consists of a list of equations sorted by type.  In these equations, variables and shocks are indexed
   by time: ``A``, ``A(1)`` and ``A(-1)`` denote variable ``A`` at date ``t``, ``(t-1)``, and ``t+1`` respectively.
   By assumption al equations are taken in expectation at date t (explanation).

   ``transition`` and ``auxiliary`` are *definition* equations, You must define variables in the same order as in the declaration header.
   Definition equation can be defined recursively, meaning that you can use a just defined variable on the right hand side.


   ``arbitrage`` equations can receive complementarity constraints separated by ``|``.
   The meaning of ``f | a<=x<=b`` is interpreted as follows: either ``f=0`` or ``f>0`` and ``x=b``
   or ``f<0`` and ``x=a``. This is very useful when representing the langrangian positivity conditions coming from an
   objective maximization. In that case the lagrangian would always be equal to ``f``. Complementarity conditions must
   be expressed directly as a function of the state.

4. The calibration part contains

        * The values of the parameters.
        * The steady-state values of endogenous variables.

        Values can depend upon each other and the declaration is not order dependent. In particular, parameters
        allowed to depend on steady-state values.

        * The covariance matrix of the shocks.


Solving the RBC model
---------------------

Here we present an example where we solve the RBC model and performs irfs, and stochastic simulation.

.. seealso:: This example is better viewed as an IPython `notebook <http://nbviewer.ipython.org/github/EconForge/dolo/blob/master/examples/notebooks/rbc_model.ipynb>`_ that you can run interactively.

Importing the model :
+++++++++++++++++++++

Import dolo:

.. code-block:: python

   from dolo import *

Import the example file provided with dolo in ``examples/models`` subdirectory and display it.

.. code-block:: python

   model = yaml_import('examples/models/rbc.yaml')
   display(model) # this prints the model equations

Solving the model :
+++++++++++++++++++

Get a first order approximation of the decision rule,

.. code-block:: python

   dr_1 = approximate_controls(model, order=1)

... For a second order approximation pass order=2

Compute the global solution. Unless bounds have been given in the yaml file, this will use the first order solution
to approximate the asymptotic distribution. Then the state-space is defined as 2 standard deviations of this
distribution around the deterministic steady-state. By default the solution algorithm uses time-iteration to determine
the decision rules and Smolyak collocation to interpolation future decision rules.

.. code-block:: python

   dr_s = global_solve(model)


Simulate the solution
+++++++++++++++++++++

Take the deterministic steady-state from the perturbation solution and consider a 1% initial shock to productivity.

.. code-block:: python

   s0 = dr_1.S_bar.copy() # deterministic steady-state is the fixed point of 1st order d.r.
   s0[0] += 0.01

Compute irfs for the global solution using this state as the starting point

.. code-block:: python

   irf = simulate(model, dr_s, s0)
   display(irf.shape)
   display(model.variables)

Now irf is an array of dimension ``n_v x 40`` where ``n_v`` is the number of variables of the model. It is possible
to change the number of observations by setting the ``horizon=`` argument (``40`` by default).


Plot the adjustment of consumption :

.. code-block:: python

   i_C = model.variables.index( Variable('c') ) # get index of consumption
   i_I = model.variables.index( Variable('i') ) # get index of investment

   plot( irf[i_C,:], label='consumption' )
   plot( irf[i_I,:], label='investment' )
   title('Productivity shock (impulse response function)')
   legend()

We can also plot stochastic simulations by setting a number of simulations ``n_exp>1``. In the following line, we
compute ``1000`` random simulations, each simulation lasting ``50`` periods.

.. code-block:: python

   sims = simulate(model, dr, s0, n_exp=100, horizon=50)
   display(sims)

The resulting object is a ``n_x x 1000 x 50`` array. The first index is the variable, the second is the simulation number
the last, the time. To plot only the first simulation :

.. code-block:: python

    i_C = model.variables.index( Variable('c') ) # get index of consumption
    i_I = model.variables.index( Variable('i') ) # get index of investment

    plot( irf[i_C,0,:], label='consumption' )
    plot( irf[i_I,0,:], label='investment' )
    title('Productivity shock (stochastic simulation)')
    legend()

If we want to plot all simulations on the same plot :

.. code-block:: python

    i_I = model.variables.index( Variable('i') )             # get index of investment
    for i in range(100):
        plot( irf[i_I,i,:], color='red', alpha=0.1 )        # transparent lines makes accumulation of draws clearer
    title('Productivity shock (stochastic simulation)')
