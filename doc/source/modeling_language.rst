The dolo language
=================

The easiest way to code a model in dolo consists in using specialized Yaml files also referred to as dolo model files.

YAML format
-----------

YAML stands for Yet Another Markup Language. It is a serialization language that
allows complex data structures in a human-readable way.
Atomic elements are floats, integers and strings.
An ordered list can be defined by separating elements with commas and enclosing them with squere brackets:

.. code:: yaml

    [1,2,3]

Equivalently, it can be done on several line, by prepending `-` to each line

.. code:: yaml

    - 'element'
    - element         # quotes are optional there is no ambiguity
    - third element   # this is interpreted as ``'third element'``



Associative arrays map keys to (simple strings to arbitrary values) as in the following example:

.. code:: yaml

    {age: 18, name: peter}

Mappings can also be defined on severaly lines, and as in Python, structures can be nested by using indentation (use spaces no tabs):

.. code:: yaml

  age: 18
  name: peter
  occupations:
    - school
    - guitar
  friends:
    paula: {age: 18}

The correspondance between the yaml definition and the resulting Python object
is very transparent. YAML mappings and lists are converted to Python  dictionaries and lists respectiveley.

.. note::

    In dolo, we use the additional convention that a dictionary key is interpreted as a Python objects if:

        - it begins with an uppercase
        - it is at least two characters long
        - its the only key in its dictionary.

    For instance,

    .. code:: yaml

        - AR1:
            rho: 0.9
            sigma: 0.1
        - TakeAList:
            - 0
            - 1
        - notanobject:
            a: 1
            b: 2

    will be interpreted in Python as:

    .. code:: python

        list(
            AR1(rho=0.9, sigma=0.1),
            TakeAList(0,1),
            {'notanobject':
                'a': 1,
                'b': 2
            }
        )



Any model file must be syntactically correct in the Yaml sense, before the
content is analysed further. More information about the YAML syntax can be found
on the `YAML website <http://www.yaml.org/>`_, especially from the `language specification <http://www.yaml.org/>`_.

Model types
-----------

Note, that dolo currently allows to define three types of models:

- `dynare models` for which a set of first order conditions are perturbated around a steady-state

- `continous states - continous controls` (CSCC models) that can be solve on a compact state-space, possibly with occasionally binding constraints

- `mixed states - continuous controls` (MSCC models) : a variant of the former category where some of the states follow an exogenous discrete markov process

Those models, differ by the type of equations they require, but the general principles are the same for all of them. Here we abstract from the differences and present only the common principles. Section [] presents these various models more in detail.
..
.. The compiler part of dolo takes a model written in a YAML format, and converts it to a Python object, that is compliant with a simple API. Hence, models can be written either using YAML files, or directly using Python syntax.

Example
-------

Here is an example model contained in the file ``examples\models\rbc.yaml``

.. literalinclude:: ../../examples/models/rbc.yaml
    :language: yaml
    :linenos:

This model can be loaded using the command:

.. code:: python

    model = yaml_import(`examples\global_models\example.yaml`)

The function `yaml_import` (cross) will raise errors until the model satisfies basic compliance tests. [more of it below]. In the following subsections, we describe the various syntaxic rules prevailing while writing yaml files.


Sections
--------

A dolo model consists in the following 4 or 5 parts:

- a `symbols` section where all symbols used in the model must be defined
- an `equations` containing the list of equations
- a `calibration` section providing numeric values for the symbols
- an `options` section containing additional informations
- a `covariances` or `markov_chain` section where exogenous shocks are defined

These section have context dependent rules. We now review each of them in detail:

Declaration section
~~~~~~~~~~~~~~~~~~~

This section is introduced by the `symbols` keyword. All symbols appearing in the model must be defined there.

Symbols must be valid Python identifiers (alphanumeric not beginning with a number) and are case sensitive. Greek letters (save for `lambda` which is a keyword) are recognized. Subscripts and superscripts can be denoted by `_` and `__` respectively. For instance `beta_i_1__d` will be pretty printed as :math:`beta_{i,1}^d`.

Symbols are sorted by type as in the following example:

.. code:: yaml

  symbols:
    variables: [a, b]
    shocks: [e]
    parameters: [rho]

Note that each type of symbol is associated with a symbol list (as `[a,b]`).


.. note::

  A common mistake consists in forgetting the commas, and use spaces only. This doesn't work since two symbols are recognized as one.

The expected types depend on the model that is being written:

- For Dynare models, all endogenous variables must be listed as `variables` with the exogenous shocks being listed as `shocks` (as in the example above).

.. note::

  The `variables`, `shocks` and `parameters` keywords correspond to the `var`, `varexo` and `param` keywords in Dynare respectively.

- Global models require the definition of the parameters, and to provide a list
of `states` and `controls`. Mixed states model also require `markov_states` that follow a discrete markov chain, while continuous states model need to identify the i.i.d `shocks` that hit the model. If the corresponding equations are given (see next subsection) optional symbols can also be defined. Among them: `values`, `expectations`.


Declaration of equations
~~~~~~~~~~~~~~~~~~~~~~~~

The `equations` section contains blocks of equations sorted by type.

Epxressions follow (roughly) the Dynare conventions. Common arithmetic operators (`+,-,*,/,^`) are allowed with conventional priorities as well as usual functions (`sqrt, log, exp, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh`). The definitions of these functions match the definitions from the `numpy` package. All symbols appearing in an expression must either be declared in the `symbols` section or be one of the predefined functions.
Any symbol `s` that is not a parameter is assumed to be considered at date `t`. Values at date `t+1` and `t-1` are denoted by `s(1)` and `s(-1)` respectively.

All equations are implicitly enclosed by the expectation operator :math:`E_t\left[\cdots \right]`. Consequently, the law of motion for the capital

.. math::
    k_{t+1} = (1-\delta) k_{t} +  i_{t} + \epsilon_t

is written as:

.. code:: yaml

    k = (1-delta)*k(-1) + i(-1)

while the Euler equation

.. math::

    E_t \left[ 1=\beta \left( \frac{c_{t+1}}{c_t} + (1-\delta)+r_{t+1} \right) \right]

is translated by:

.. code:: yaml

    1 = beta*(c/c(1))^(sigma)*(1-delta+rk(1))

.. note::

   In Python, the exponent operator is denoted by `**` while the caret operator `^` represents bitwise XOR. In dolo models, we ignore this distinction and interpret both as an exponent.

.. note::

    The default evaluator in dolo preserves the evaluation order. Thus       ``(c(1)/c)^(-gamma)`` is not evaluated in the same way (and is numerically more stable) than ``c(1)^(-gamma)/c^(-gamma)``. Currently, this is not true for symbolically computed derivatives, as expressions are automatically simplified, implying that execution order is not guaranteed. This impacts only higher order perturbations.

.. idea: we could allow for equations like a = E[ fsjlkjaskldf ] and completely ignore E[ ] unless it doesn't enclose the equation completely.

.. errors in parsing equations derive from the class X

An equation can consist of one expression, or two expressions separated by `=`.
There are two types of equation blocks:

- condition blocks

    In these blocks, each equation ``lhs = rhs`` define the scalar value ``(rhs)-(lhs)```. A list of of such equations, i.e a block, defines a multivariate function of the appearing symbols.
    Certain condition blocks, can be associated with complementarity conditions separated by ``|`` as in ``rhs-lhs | 0 < x < 1``. In this case it is advised to omit the equal sign in order to make it easier to interpret the complementarity. Also, when complementarity conditions are used, the ordering of variables appearing in the complementarities must match the declaration order (more in section Y).

- definition blocks

    Definition blocks differ from condition blocks in that they define a group of variables (``states`` or ``auxiliaries``) as a function of the right hand side.
The types of variables appearing on the right hand side depend on the block type. The variables enumerated on the left hand-side must appear in the declaration order.


.. note::

    In the RBC example, the ``auxiliary`` block defines variables (``y,c,rk,w``) that can be directly deduced from the states and the controls:

    .. code:: yaml

        auxiliary:
            - y = z*k^alpha*n^(1-alpha)
            - c = y - i
            - rk = alpha*y/k
            - w = (1-alpha)*y/w

    Note that the declaration order matches the order in which variables appear on the left hand side. Also, these variables are defined recursively: ``c``, ``rk`` and ``w`` depend on the value for ``y``. In contrast to the calibration block, the definition order matters. Assuming that variables where listed as (``c,y,rk,w``) the following block would provide incorrect result since ``y`` is not known when ``c`` is evaluated.

    .. code:: yaml

        auxiliary:
            - c = y - i
            - y = z*k^alpha*n^(1-alpha)
            - rk = alpha*y/k
            - w = (1-alpha)*y/w




Calibration section
~~~~~~~~~~~~~~~~~~~

The role of the calibration section consists in providing values for the parameters and the variables. The calibration of all parameters appearing in the equation is of course strictly necessary while the the calibration of other types of variables is useful to define the steady-state or an initial guess to the steady-state.

The calibrated values are also substituted in other sections, including the `shocks` and `options` section. This is particularly useful to make the covariance matrix depend on model parameters, or to adapt the state-space to the model's calibration.

The calibration is given by an associative dictionary mapping symbols to define with values. The values can be either a scalar or an expression. All symbols are treated in the same way, and values can depend upon each other as long as there is a way to resolve them recursively.

In particular, it is possible to define a parameter in order to target a special value of an endogenous variable at the steady-state. This is done in the RBC example where steady-state labour is targeted with ``n: 0.33`` and the parameter ``phi`` calibrated so that the optimal labour supply equation holds at the steady-state (``chi: w/c^sigma/n^eta``).

All symbols that are defined in the `symbols` section but do not appear in the calibration section are initialized with the value `nan` without issuing any warning.

.. note::

    No clear policy has been established yet about how to deal with undeclared symbols in the calibration section. Avoid them.

Shock specification
~~~~~~~~~~~~~~~~~~~

The way shocks are specified depends on the type of model. They are constructed using a the rules for mini-languages defined in section [ref].

Distribution
............

For Dynare and continuous-states models, one has to specifiy a multivariate distribution of the i.i.d. process for the vector of ``shocks`` (otherwise shocks are assumed to be constantly 0). This is done in the distribution section. A gaussian distrubution (only one supported so far), is specified by supplying the covariance matrix as a list of list as in the following example.

.. code:: yaml

    distribution:

        Normal: [
                [sigma_1, 0.0],
                [0.0, sigma_2]
            ]

Markov chains
.............

When the model is driven by an exogenous discrete markov chain, that is for DTMSCC models, shocks are defined in the ``discrete_transition`` section. The objects allowed in this section are: `MarkovChain, AR1, MarkovTensor`

 markov chain can be constructed in several ways:

   - by listing directly a list of states, and a transition matrix as in :

        .. code:: yaml

            discrete_transition:
                MarkovChain:   # a markov chain is defined by providing:
                    - [ [0.0, -0.02]           # a list of markov states
                        [0.0,  0.02]
                        [-0.1, 0.02]]
                    - [ [ 0.98, 0.01, 0.01],   # a transition matrix
                        [ 0.10, 0.01, 0.90],
                        [ 0.05, 0.05, 0.90] ]

    - by using primitives to construct a discretized process from an AR1:

        .. code:: yaml

            discrete_transition:
                AR1:
                    rho: 0.9
                    sigma: [
                            [0.01, 0.001]
                            [0.001, 0.02]
                        ]
                    N: 3
                    method: rouwenhorst   # the alternative is tauchen

    - by combining two processes together:

        .. code:: yaml

            discrete_transition:
                MarkovTensor:
                    - AR1:
                        rho: 0.9
                        sigma: [
                                [0.01, 0.001]
                                [0.001, 0.02]
                            ]
                        N: 3
                        method: rouwenhorst   # the alternative is tauchen
                    - AR1:
                        rho: 0.9
                        sigma: 0.01
                        N: 2
                        method: rouwenhorst   # the alternative is tauchen


Options
~~~~~~~

The `options` section contains all informations necessary to solve the model. It can also contain arbitrary additional informations. The section follows the mini-language convention, with all calibrated values replaced by scalars and all keywords allowed.

Global solutions require the definition of an approximation space. The lower, upper bounds and approximation orders (number of nodes in each dimension) are defined as in the following example:

.. code:: yaml

    options:
        Approximation:
            a: [ 1-2*asig_z, k*0.9 ]
            b: [ 1+2*asig_z, k*1.1 ]
            orders: [10, 50]
        arbitrary_information

This reads as follows: the upper and lower bounds for the productivity process are 1 minus and plus two times its asymptotic standard deviation. The boundaries for the capital level are defined by a 10% bracket around its steady-state value (or the one defined in the calibration section). 10 points are used to discretize the state-space for the productivity process and 50 are used for the capital level.
