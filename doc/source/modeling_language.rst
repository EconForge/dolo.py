Model Definition
================

Dolo currently allows to define three types of models:

- `dynare models` for which a set of first order conditions are perturbated around a steady-state

- `continous states - continous controls` (CSCC models) that can be solve on a compact state-space, possibly with occasionally binding constraints

- `mixed states - continuous controls` (MSCC models) : a variant of the former category where some of the states follow an exogenous discrete markov process

Those models, differ by the type of equations they require, but the general principles are the same for all of them. The compiler part of dolo takes a model written in a YAML format, and converts it to a Python object, that is compliant with a simple API. Hence, models can be written either using YAML files, or directly using Python syntax.


The dolo language
-----------------

The easiest way to code a model in dolo consists in using specialized Yaml files also referred to as dolo model files.

YAML format
~~~~~~~~~~~

YAML stands for Yet Another Markup Language. It is a serialization language that
allows complex data structure in a human-readable way. Like Python, nested structures are denoted by semi colon and/or identation.
Among other features, it understands floats, integers and strings as basic element, and has lists as well as associative dictionaries to structure them. The mapping between these structures and the Python object they are converted into is very transparent. For instance the following associative array in yaml:

.. code::

  age: 18
  name: peter
  occupations: [school, guitar]

would be equivalent to the python code:

.. code::

  dict(
    age=18,
    name='peter',
    occupations=['school','guitar']
  )

Any model file must be syntactically correct in the Yaml sense, before the
content is analysed further.

Sections
~~~~~~~~

A dolo model consists in 4 or 5 parts whose interpretation follows specific rules. Supported sections

- a `symbols` section where all symbols used in the model must be defined
- an `equations` containing the list of equations
- a `calibration` section providing numeric values for the symbols
- an `options` section containing additional informations
- a `covariances` or `markov_chain` section where exogenous shocks are defined

We now review all these sections in detail

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

The `equations` section contains blocks of equations sorted by type. A dynare model contains one single block `dynare_block`. Each equation is written using (broadly) the same convention as Dynare.


Calibration section
~~~~~~~~~~~~~~~~~~~

Shock specification
~~~~~~~~~~~~~~~~~~~

Normally distributed shocks
...........................

Markov chains mini-language
...........................

Options
~~~~~~~

Approximation space
...................

The model object
----------------
