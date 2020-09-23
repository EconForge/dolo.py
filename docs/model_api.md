# Model API

For numerical purposes, models are essentially represented as a set of
symbols, calibration and functions representing the various equation
types of the model. This data is held in a `Model` object whose
API is described in this chapter. Models are usually created by writing
a Yaml files as described in the the previous chapter, but as we will
see below, they can also be written directly as long as they satisfy the requirements detailed below.

## Model Object


As previously, let\'s consider, the Real Business Cycle example, from
the introduction. The model object can be created using the yaml file:

``` {.python}
model = yaml_import('models/rbc.yaml')
```

The object contains few meta-data:

``` {.yaml}
display( model.name )  # -> Real Business Cycles
display( model.model_type )   # -> `dtcc`
```


!!! note

    ``model_type`` field is now always ``dtcc``. Older model types (``'dtmscc'``, ``'dtcscc'``, ``'dynare'``) are not used anymore.



## Calibration


Each models stores calibration information as `model.calibration`. It is a special dictionary-like object,  which contains
calibration information, that is values for parameters and initial values (or steady-state) for all other variables of the model.

It is possible to retrieve one or several variables calibrations:

``` python
display( model.calibration['k'] ) #  ->  2.9
display( model.calibration['k', 'delta']  #  -> [2.9, 0.08]
```

When a key coresponds to one of the symbols group, one gets one or several vectors of variables instead:

```python
model.calibration['states'] # - > np.array([2.9, 0]) (values of states [z, k])
model.calibration['states', 'controls'] # -> [np.array([2.9, 0]), np.array([0.29, 1.0])]
```


To get regular dictionary mapping states groups and vectors, one can use the attributed `.grouped`
The values are vectors (1d numpy arrays) of values for each symbol group. For instance the following code will print the calibrated values of the parameters:

```python
for (variable_group, variables) in model.calibration.items():
    print(variables_group, variables)
```

In order to get a ``(key,values)`` of all the values of the model, one can call ``model.calibration.flat``.

```python
for (variable_group, variables) in model.calibration.items():
    print(variables_group, variables)
```


!!! note

    The calibration object can contain values that are not symbols of the model. These values can be used to calibrate model parameters
    and are also evaluated in the other yaml sections, using the supplied value.


One uses the `model.set_calibration()` routine to change the calibration of the model.  This one takes either a dict as an argument, or a set of keyword arguments. Both calls are valid:

```python
model.set_calibration( {'delta':0.01} )
model.set_calibration( {'i': 'delta*k'} )
model.set_calibration( delta=0.08, k=2.8 )
```

This method also understands symbolic expressions (as string) which makes it possible to define symbols as a function of other symbols:

```python
model.set_calibration(beta='1/(1+delta)')
print(model.get_calibration('beta'))   # -> nan

model.set_calibration(delta=0.04)
print(model.get_calibration(['beta', 'delta'])) # -> [0.96, 0.04]
```

Under the hood, the method stores the symbolic relations between symbols. It is precisely equivalent to use the ``set_calibration`` method
or to change the values in the yaml files. In particular, the calibration order is irrelevant as long as all parameters can be deduced one from another.


## Functions

A model of a specific type can feature various kinds of functions. For instance, a continuous-states-continuous-controls models, solved by iterating on the Euler equations may feature a transition equation $g$ and an arbitrage equation $f$. Their signature is respectively $s_t=g(m_{t-1},s_{t-1},x_{t-1},m_t)$ and $E_t[f(m_t,s_t,x_t,s_{t+1},x_{t+1},m_{t+1})]$, where $s_t$, $x_t$ and $e_t$ respectively represent a vector of states, controls and exogenous shock. Implicitly, all functions are also assumed to depend on the vector of parameters :math:`p`.

These functions can be accessed by their type in the model.functions dictionary:

```python
g = model.functions['transition']
f = model.functions['arbitrage']
```

Let's call the arbitrage function on the steady-state value, to see the residuals at the deterministic steady-state:

```python
m = model.calibration['exogenous']
s = model.calibration['states']
x = model.calibration['controls']
p = model.calibration['parameters']
res = f(m,s,x,m,s,x,p)
display(res)
```

The output (`res`) is two element vector, representing the residuals of the two arbitrage equations at the steady-state. It should be full of zero. Is it ? Great !

By inspecting the arbitrage function ( `f?` ), one can see that its call api is:

```python
f(m,s,x,M,S,X,p,diff=False,out=None)
```

Since `m`, `s` and `x` are the short names for exogenous shocks, states and controls, their values at date $t+1$ is denoted with `S` and `X`. This simple convention prevails in most of dolo source code: when possible, vectors at date `t` are denoted with lowercase, while future vectors are with upper case. We have already commented the presence of the parameter vector `p`.
Now, the generated functions also gives the option to perform in place computations, when an output vector is given:

```python
out = numpy.ones(2)
f(m,s,x,m,s,x,p,out)   # out now contains zeros
```

It is also possible to compute derivatives of the function by setting ``diff=True``. In that case, the residual and jacobians with respect to the various arguments are returned as a list:

```python
r, r_m, r_s, r_x, r_M, r_S, r_X = f(m,s,x,m,s,x,p,diff=True)
```

Since there are two states and two controls, the variables ``r_s, r_x, r_S, r_X`` are all 2 by 2 matrices.

The generated functions also allow for efficient vectorized evaluation. In order to evaluate the residuals :math:`N` times, one needs to supply matrix arguments, instead of vectors, so that each line corresponds to one value to evaluate as in the following example:

```python
N = 10000

vec_m = m[None,:].repeat(N, axis=0) # we repeat each line N times
vec_s = s[None,:].repeat(N, axis=0) # we repeat each line N times
vec_x = x[None,:].repeat(N, axis=0)
vec_X = X[None,:].repeat(N, axis=0)
vec_p = p[None,:].repeat(N, axis=0)

# actually, except for vec_s, the function repeat is not need since broadcast rules apply
vec_s[:,0] = linspace(2,4,N)   # we provide various guesses for the steady-state capital
vec_S = vec_s

out = f(vec_m, vec_s,vec_x,vec_M, vec_S,vec_X,vec_p)  # now a 10000 x 2 array

out, out_m, out_s, out_x, out_M, out_S, out_X = f(vec_m, vec_s,vec_x, vec_m, vec_S,vec_X,vec_p)
```


The vectorized evaluation is optimized so that it is much faster to make a vectorized call rather than iterate on each point. 

!!! note
    In the preceding example, the parameters are constant for all evaluations, yet they are repeated. This is not mandatory, and the call ``f(vec_m, vec_s, vec_x, vec_M, vec_S, vec_X, p)`` should work exactly as if `p` had been repeated along the first axis. We follow there numba's ``guvectorize`` conventions, even though they slightly differ from numpy's ones.


## Exogenous shock

The `exogenous` field contains information about the driving process. To get its default, discretized version, one can call `model.exogenous.discretize()`.


## Options structure


The ``model.options`` structure holds an information required by a particular solution method. For instance, for global methods, ``model.options['grid']`` is supposed to hold the boundaries and the number nodes at which to interpolate.

```python
display( model.options['grid'] )
```
