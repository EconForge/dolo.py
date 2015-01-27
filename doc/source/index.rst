What is dolo ?
##############

.. only:: html

    .. This is only for the webpage.

    Dolo is a tool to assist researchers in solving several types of DSGE models, using either local of global approximation methods.

    Users are can separate the definition of their models from the solution algorithm. A simple syntax is
    provided in YAML files to define variables and equations of several types. This syntax integrates the specification
    of occasionally binding constraints (a.k.a. complementarity conditions)

    Dolo then checks consistency of the model and computes an efficient numerical representation. This includes producing
    vectorized code with symbolic derivatives (if needed).

    The user can then implement his own preferred solution method using one of the provided tools (various types of interpolation,
    solvers, ...) or use one of the already implemented procedures. Currently, only time-iteration is supported but in the near future,
    there will also be value function iteration and parametrized expectations. High level functions are provided so that
    should be extremely easy in most of the cases.

    Dolo is written in python and so are his solution routines. If you prefer or need to use another language for the solution
    (such as a legacy scientific Fortran wrapper edited by Mathworks...), you can always use dolo as a preprocessor. In
    that case, dolo will just translate the model file into a numerical file usable by your sofware. Currently, Matlab and
    Julia are supported.

.. 
.. Index
.. #####

.. toctree::
    :maxdepth: 2

    introduction
    installation
    quick_tutorial
    modeling_language
    model_specification
    algos
    examples
    faq
    api
