Model API
=========

For numerical purposes, models are essentially represented as a set of symbols,
calibration and functions representing the various equation types of the model.
This data is held in a ``NumericalModel`` object whose API is described in this chapter. Models are usually created by writing a Yaml files as described in the the previous chapter, but as we will see below, they can also be written directly.

.. document model.residuals

Numerical Model Object
----------------------

As previously, let's consider, the Real Business Cycle example, from the introduction. The model object can be created using the yaml file:

.. code:: yaml

    model = yaml_import('models/rbc.yaml')

The object contains few meta-data:

.. code:: yaml

    display( model.name )  # -> Real Business Cycles
    display( model.model_type )   # -> `dtcc`
    display( model.model_specs )   # -> `(f,g,v)`

The ``model.name`` field contains a possibly long string identifying the model.
The ``'model.model_features'`` field summarizes which equations types are provided which determines the solution algorithms that can be used to solve the model. Here ``(f,g,v)`` means that ``arbitrage`` (short ``f``), ``transition`` (short ``g``) and ``value`` equations were provided meaning that time-iteration or value function iteration can both be used to solve the model. When using a yaml files, the ``model_type` and ``model_specs`` properties are automatically set.

..note::

    ``model_type`` field is now always ``dtcc``. Older model types (``'dtmscc'``, ``'dtcscc'``, ``'dynare'``) are not used anymore.

The various attributes of the model directly echo the sections from the Yaml file.

Symbols
+++++++

Symbols are held in the `model.symbols` dictionary, with each symbol type mapping to a list of symbol strings, that will be used in equations. Although these symbols are not needed stricto sensu for computations, they are very useful to calibrate the steady-state or to label the graphs and simulations

.. code:: yaml

    display(model.symbols)


.. note::

    Although dictionaries read from the yaml file are unordered, the structure representing them in Python is actually an `OrderedDict` rather than a `dict` object. This is to allow for more predictability and conistency in outputs. The order is conventional and the keys are ordered after the list 'variables, states, controls, auxiliaries, values, parameters' (missing types are omitted from the list).


Calibration
+++++++++++

Each models stores calibration information as `model.calibration`. It is a special dictionary-like object,  which contains
calibration information, that is values for parameters and initial values (or steady-state) for all other variables of the model.

It is possible to retrieve one or several variables calibrations:

.. code:: python

    display( model.calibration['k'] ) #  ->  2.9
    display( model.calibration['k', 'delta']  #  -> [2.9, 0.08]

When a key coresponds to one of the symbols group, one gets one or several vectors of variables instead:

.. code:: python

    model.calibration['states'] # - > np.array([2.9, 0]) (values of states [z, k])
    model.calibration['states', 'controls'] # -> [np.array([2.9, 0]), np.array([0.29, 1.0])] (values of states [z,k] and controls [i,n])


To get regular dictionary mapping states groups and vectors, one can use the attributed `.grouped`
The values are vectors (1d numpy arrays) of values for each symbol group. For instance the following code will print the calibrated values of the parameters:

.. code:: python

    for (variable_group, variables) in model.calibration.items():
        print(variables_group, variables)


In order to get a ``(key,values)`` of all the values of the model, one can call ``model.calibration.flat``.

.. code:: python

    for (variable_group, variables) in model.calibration.items():
        print(variables_group, variables)


.. note

    The calibration object can contain values that are not symbols of the model. These values can be used to calibrate model parameters
    and are also evaluated in the other yaml sections, using the supplied value.


One uses the ``model.set_calibration()`` routine to change the calibration of the model.  This one takes either a dict as an argument, or a set of keyword arguments. Both calls are valid:

.. code:: python

    model.set_calibration( {'delta':0.01} )
    model.set_calibration( {'i': 'delta*k'} )
    model.set_calibration( delta=0.08, k=2.8 )


This method also understands symbolic expressions (as string) which makes it possible to define symbols as a function of other symbols:

.. code:: python

    model.set_calibration(beta='1/(1+delta)')
    print(model.get_calibration('beta'))   # -> nan

    model.set_calibration(delta=0.04)
    print(model.get_calibration(['beta', 'delta'])) # -> [0.96, 0.04]

Under the hood, the method stores the symbolic relations between symbols. It is precisely equivalent to use the ``set_calibration`` method
or to change the values in the yaml files. In particular, the calibration order is irrelevant as long as all parameters can be deduced one from another.

Functions
+++++++++

A model of a specific type can feature various kinds of functions. For instance, a continuous-states-continuous-controls models, solved by iterating on the Euler equations may feature a transition equation :math:`g` and an arbitrage equation :math:`f`. Their signature is respectively :math:`s_t=g(s_{t-1},x_{t-1},e_t)` and :math:`E_t[f(s_t,x_t,s_{t+1},x_{t+1})]`, where :math:`s_t`, :math:`x_t` and :math:`e_t` respectively represent a vector of states, controls and shocks. Implicitly, all functions are also assumed to depend on the vector of parameters :math:`p`.

These functions can be accessed by their type in the model.functions dictionary:

.. code:: python

    g = model.functions['transition']
    f = model.functions['arbitrage']

Let's call the arbitrage function on the steady-state value, to see the residuals at the deterministic steady-state:

.. code:: python

    m = model.calibration['exogenous']
    s = model.calibration['states']
    x = model.calibration['controls']
    p = model.calibration['parameters']
    res = f(m,s,x,m,s,x,p)
    display(res)

The output (``res``) is two element vector, representing the residuals of the two arbitrage equations at the steady-state. It should be full of zero. Is it ? Great !

By inspecting the arbitrage function ( ``f?`` ), one can see that its call api is:

.. code:: python

    f(m,s,x,M,S,X,p,diff=False,out=None)

Since ``m``, ``s`` and ``x`` are the short names for exogenous shocks, states and controls, their values at date :math:`t+1` is denoted with ``S`` and ``X``. This simple convention prevails in most of dolo source code: when possible, vectors at date ``t`` are denoted with lowercase, while future vectors are with upper case. We have already commented the presence of the paramter vector ``p``.
Now, the generated functions also gives the option to perform in place computations, when an output vector is given:

.. code:: python

    out = numpy.ones(2)
    f(m,s,x,m,s,x,p,out)   # out now contains zeros

It is also possible to compute derivatives of the function by setting ``diff=True``. In that case, the residual and jacobians with respect to the various arguments are returned as a list:

.. code:: python

    r, r_m, r_s, r_x, r_M, r_S, r_X = f(m,s,x,m,s,x,p,diff=True)

Since there are two states and two controls, the variables ``r_s, r_x, r_S, r_X`` are all 2 by 2 matrices.

The generated functions also allow for efficient vectorized evaluation. In order to evaluate the residuals :math:`N` times, one needs to supply matrix arguments, instead of vectors, so that each line corresponds to one value to evaluate as in the following example:

.. code:: python

    N = 10000

    vec_m = m[None,:].repeat(N, axis=0) # we repeat each line N times
    vec_s = s[None,:].repeat(N, axis=0) # we repeat each line N times
    vec_x = x[None,:].repeat(N, axis=0)
    vec_X = X[None,:].repeat(N, axis=0)
    vec_p = p[None,:].repeat(N, axis=0)
    # actually, except for vec_s, the function repeat is not need since broadcast rules apply
    vec_s[:,0] = linspace(2,4,N) # we provide various guesses for the steady-state capital
    vec_S = vec_s

    out = f(vec_m, vec_s,vec_x,vec_M, vec_S,vec_X,vec_p)  # now a 10000 x 2 array

    out, out_m, out_s, out_x, out_M, out_S, out_X = f(vec_m, vec_s,vec_x, vec_m, vec_S,vec_X,vec_p)



The vectorized evaluation is optimized so that it is much faster to make a vectorized call rather than iterate on each point. By default, this is achieved by using the excellent `numexpr` library.

.. note::

    In the preceding example, the parameters are constant for all evaluations, yet they are repeated. This is not mandatory, and the call ``f(vec_m, vec_s, vec_x, vec_M, vec_S, vec_X, p)`` should work exactly as if `p` had been repeated along the first axis. We follow there numba's ``guvectorize`` conventions, even though they slightly differ from numpy's ones.


Exogenous shock
+++++++++++++++

TODO: expand

The `exogenous` field contains information about the driving process. To get its default, discretize version, one can call `model.exogenous.discretize()`.



Options structure
+++++++++++++++++

The ``model.options`` structure holds an information required by a particular solution method. For instance, for global methods, ``model.options['grid']`` is supposed to hold the boundaries and the number nodes at which to interpolate.

.. code::

    display( model.options['grid'] )

..
Source documentation
--------------------

Model
+++++++++++++++

.. autoclass:: dolo.compiler.model.Model
   :members:
