Custom model types
==================

Recipes
-------

Different kinds of solution algorithm are associated with several
ways of writing the model.
Dolo provides a simple way to define your own type of models. These models
can then be translated into numerical models.

A custom definition of a model type is called a recipe. Recipes can be
described in a YAML file or directly as a Python object. Let's inspect the
file which define the ``fga`` type is used by the default time-iteration
algorithm.

.. literalinclude:: ../../examples/recipes/recipe_fga.yaml
    :language: yaml
    :linenos:

- First the ``model_type`` key defines a short name for the recipe.

- Second, a list of symbol types is declared. Two types of symbols are automatically
added to that list : ``shocks`` and ``parameters``.

- Third, all blocks of equation are described in the equation_type part
  by supplying a list of allowed symbol types, with the date at which they occur.

  Currently, two types of blocks are recognized by dolo:

    - regular blocks (like ``arbitrage``) consist of a list of expressions. When an equal
      sign is supplied in the model file, the right hand side is substracted from the left hand side.

    - definition blocks (with ``definition: True`` as in ``transition`` and ``auxiliary``) define the left hand
      side as a function of the right hand side. The left hand side must be a series of variable
      at date 0.

    .. seealso::

        When an actual model is checked, dolo will check that these variables are defined
        in the same order as in the declaration header of the model file. It also allows for recursive definitions as long
        as the block of equations forms a triangular system of the left hand side variables.


Checking a model's validity
---------------------------

A recipe can be used inside Python to check the validity of a symbolic model.
For instance the following code checks that the ``rbc.yaml`` file is valid.

    .. code-block:: python

        model = yaml_import('rbc.yaml')
        from dolo.symbolic.recipes import recipe_fga
        from dolo.symbolic.validator import validate

        validate( model, recipe_fga )

If the model is valid according the its definition, the last function will not do anything. If not an error is thrown.

Producing matlab/julia code using a custom recipe
-------------------------------------------------

Assuming a model and a recipe are respectively defined in files ``model.yaml`` and ``recipe.yaml``, one can generate
a compiled model for matlab ``model.m`` using the following command:

.. code-block:: sh

   dolo-matlab model.yaml --recipe=recipe.yaml


The same can be down for Julia using:

.. code-block:: sh

   dolo-matlab model.yaml --recipe=recipe.yaml


The resulting file will contain nested structures (or nested dictionaries depending on the target language) with
the following content: ::

    - symbols

        - val_1         ( cell array with names of symbols of type 1 )
        - val_2
        - parameters    ( names of parameters )
        - shocks        ( names of shocks )

    - calibration

        - steady_state
                - val_1     ( steady_state values for symbols of type 1 )
                - val_2
        - parameters        ( numeric values for parameters )
        - sigma             ( covariance matrix )

    - functions

        - fun1    ( function of type fun1 )
        - fun2


The generated function handlers (here ``fun1`` and ``fun2``) have the following signature : ``fun(arg1, arg2, ..., p)``
where ``arg_i`` is the i-th argument defined in the recipe as ``type_i``, and is expected to be a vertical array of size ``N x n_i``
where ``n_i`` is the number of variables of type ``type_i`` and ``N`` is the number of points at which the function
must be evaluated. ``p`` is a one dimensional vector of parameter values.

    .. seealso::

        For row-major languages, especially for Python, arrays are expected to be of the size ``n_i x N`` so that
        the function can still operate on contiguous chunks of memory when ``N`` is big.