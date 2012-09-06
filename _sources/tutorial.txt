Quick tutorial
==============

-----------------
Quick quick quick
-----------------

Open an IPython session.

Import dolo:

.. code-block:: python

   from dolo import *

Import the example file provided with dolo in ``examples/global_models`` subdirectory and display it.

.. code-block:: python

   model = yaml_import('../examples/global_models/optimal_growth')
   display(model) # this prints the model equations

Solve the model at first order...

.. code-block:: python

   dr_1 = approximate_controls(model, order=1)

... and at second order ...

.. code-block:: python

   dr_2 = approximate_controls(model, order=2)

... and using smolyak colocation (global solution):

.. code-block:: python

   dr_s = global_solve(model, order=1)

Compute irfs for each of the methods, using deterministic steady-state as a starting point.

.. code-block:: python

   s0 = dr_1.S_bar # deterministic steady-state is the fixed point of 1st order d.r.

   irfs = [ simulate(model, dr, s0) for dr in [dr_1,dr_2_dr_s] ]

Plot the path of consumption for each solution method:

.. code-block:: python

   i_C = model.variables.index( Variable(c,0) ) # get index of consumption ( it is equal to 2 )

   for irf in irfs:
       plot( irf[i_C,:] )


-------------
More details
-------------

Write a model
\\\\\\\\\\\\\

Here is the content of the ``optimal_growth.yaml`` model:

.. literalinclude:: ../../examples/global_models/optimal_growth.yaml
   :language: yaml


It is a valid YAML file (see http://yaml.org/). In particular, it is sensitive to indentation and cannot contain tabs.
It consists in several part:

* The declaration block contains all names of variables/parameters to be used in the model. Here the model contains several
  kinds of variables: *controls*, *states* and *expectations*. There are also exogenously distributed innovations (*shocks*)
  and *parameters*.

* The model part contains a list of equations sorted by type.  In these equations, variables and shocks are indexed
  by time: A, A(1) and A(-1) denote variable A at date t, (t-1), and (t+1) respectively. By assumption all equations are taken
  in expectation at date t (explanation). Optionnaly, arbitrage equations can receive complementarity constraints (see
  relevant section)

* The calibration part contains the value of the parameters, and and initial value for endogenous variables. Parameters
  can depend upon each other and the declaration is not order dependent. Parameters are also allowed to depend on steady-state
  values.

* The covariance matrix of the shock can also be specified in the calibration part




