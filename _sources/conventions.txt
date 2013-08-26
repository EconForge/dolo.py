Model conventions
=================

This page describes various conventions used in dolo.

+++++++++++
Model types
+++++++++++

There are several ways to represent a DSGE model. Here we list some of them:

state-free approach
-------------------

This approach is taken in Dynare and in most perturbation softwares. The model is specified by a vectorial function

:math:`f(y_{t+1},y_t,y_{t-1},\epsilon_t) = 0`

with the restriction that :math:`\epsilon_t` and :math:`y_{t+1}` cannot appear in the same equations.
For this kind of models, the solver finds a solution :math:`g` such that the law of motion of :math:`(y_t)` is given by: :math:`y_t = g \left( y_{t-1}, \epsilon_t \right)`

controlled process variants
---------------------------

We define several categories of models.

.. graphviz::

      digraph foo {
          "fgah" -> "fga";
          "fga" -> "fg";
          "fgh" -> "fg";
      }


*fg* model
//////////

With these versions, the state-space is chosen by the user. A law of motion for the state-space must be specified (depending on the controls and on the shocks). And optimality conditions must be given to pin down all the controls.
The model is specified by giving :math:`g` and :math:`f` such that:

:math:`s_t = g \left( s_{t-1}, x_{t-1}, \epsilon_t \right)`

:math:`E_t \left[ f \left( s_t, x_t, s_{t+1}, x_{t+1} \right) \right]=0`

The solution is a function :math:`\varphi` such that :math:`x_t=\varphi(s_t)`.

*fga* model
///////////

In some cases, some variables can be directly expressed as a function of other variables. We call them *auxiliary* variables. Auxiliary variables are restricted
to depend only on contemporaneous variables (controls or states).  The model can be rewritten:

:math:`a_t = a\left(s_t, x_t\right)`

:math:`s_t = g \left( s_{t-1}, x_{t-1}, a_{t-1}, \epsilon_t \right)`

:math:`E_t \left[ f \left( s_t, x_t, a_t, s_{t+1}, x_{t+1}, a_{t+1} \right) \right]`

Clearly, by substituting the variables *a* everywhere, this type of model can be seen as an *fg* model. Hence when some
algorithm is applicable to an *fg* model, it can be also be applied to an *fga* model.


not implemented ; fgh model
///////////////////////////

A sub-variant of this specification let the user choose equations to define expectations. This is useful for PEA approaches. The model is specified by giving :math:`g`, :math:`f` and :math:`h` such that:

:math:`s_t = g \left( s_{t-1}, x_{t-1}, \epsilon_t \right)`

:math:`f \left( s_t, x_t, z_t \right)`

:math:`z_t = E_t h \left( s_{t+1}, x_{t+1} \right)`

Same remark as for *fga* model: is needed, *fgh* models can behave exactly as an *fg* model.


not implemented ; fgah model
////////////////////////////

A sub-variant of this specification let the user choose equations to define expectations. This is useful for PEA approaches. The model is specified by giving :math:`g`, :math:`f` and :math:`h` such that:

:math:`a_t = a \left(s_t, x_t\right)`

:math:`s_t = g \left( s_{t-1}, x_{t-1}, a_{t-1}, \epsilon_t \right)`

:math:`f \left( s_t, x_t,a_t,  z_t \right)`

:math:`z_t = E_t h \left( s_{t+1}, x_{t+1} , a_{t+1} \right)`

*fgah* models can behave exactly as an *fg* model, or an *fga* model

+++++++++++++++++++++
Numerical Conventions
+++++++++++++++++++++

* A list of *N* points in a *d*-dimensional space is represented by a *d x N* matrix.
  In other words the last index always corresponds to the successive observations.
  This is consistent with default ordering in numpy: last index varies faster.


.. note::

   When dolo is used to compile a model for matlab the convention is reversed: first dimension
   denotes the successive points. This is adapted to matlab's storage order (Fortran order).



* In order to evaluate many times a function :math:`f: R^s \rightarrow R^x` for a list of *N* points we consider the vectorized version:

:math:`f: R^s x R^N \rightarrow R^x x R^N` . If *X* is the matrix of inputs and *F* the matrix of output, then for any i
 *F[i,:]* is the image of *X[i,:]* by *f*.

* In particular, the implementation of the model definition follow this convention, as well as the solution of the model.
