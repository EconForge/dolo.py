Inspecting the solution
***********************

The output of most solution methods is a decision rule for the controls as a
function of the exogenous and endogenous states: ``dr``. This decision rule
can be called using one of the following methods:

- ``dr.eval_s(s: array)``: function of endogenous state. Works only if exgogenous process is i.i.d.

- ``dr.eval_ms(m: array,s: array)``: function of exogenous and endogenous values. Works only if exogenous process is continuous.

- ``dr.eval_is(i: int,s: array)``: function of exognous index and endogenous values. Works only if some indexed discrete values are associated with exogenous process.

There is also a __call__ function, which tries to make the sensible call based on argument types. Hence ``dr(0, s)`` will behave as the third example.


Tabulating a decision rule
@@@@@@@@@@@@@@@@@@@@@@@@@@

Dolo provides a convenience function to plot the values of a decision rule against different values of a state:

.. autofunction:: dolo.algos.simulations.tabulate


Stochastic simulations
@@@@@@@@@@@@@@@@@@@@@@

Given a model object and a corresponding decision rule, one can get a ``N`` stochastic simulation for ``T`` periods,
using the ``simulate`` function. The resulting object is an 3-dimensional *DataArray*, with the following labelled axes:
- T: date of the simulation (``range(0,T)``)
- N: index of the simulation (``range(0,N)``)
- V: variables of the model (``model.variables``)

.. autofunction:: dolo.algos.simulations.simulate

Impulse response functions
@@@@@@@@@@@@@@@@@@@@@@@@@@

For continuously valued exogenous shocks, one can perform an impulse response function:

.. autofunction:: dolo.algos.simulations.response


Graphing nonstochastic simulations
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Given one or many nonstochstic simulations of a model, obtained with ``response``, or ``deterministic_solve``
it is possible to quickly create an irf for multiple variables.

.. autofunction:: dolo.misc.graphs.plot_irfs
