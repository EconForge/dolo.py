Solution algorithms
*******************

Solution algorithms operate on model objects to determine
the optimal decision rules for the model agents.

Each algorithm requires its target model to have functions
defined in its specification.

This is a mapping between solution algorithm and necessary functions.
Description of these functions, including their type signature,
can be found in the Model Specification.

- Improved time iteration: `transition` and `arbitrage`
- Parameterized expectations : `transition`, `expectation`, and `direct response`
- Perfect foresight: `transition` and `arbitrage`
- Perturbation : `transition` and `arbitrage`
- Steady-state : `transition` and `arbitrage`
- Time iteration: `transition` and `arbitrage`
- Value function iteration : `transition`, and either `felicity` or `value`

.. toctree::
    :maxdepth: 2

    steady_state
    perturbations
    parameterized_expectations
    perfect_foresight
    time_iteration
    value_iteration
