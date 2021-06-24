Model Specification
===================

Variables
--------------

### Variable types

The following types of variables can be used in models:

> -   `exogenous` (`m`) (can be autocorrelated)
> -   `states` (`s`)
> -   `controls` (`x`)
> -   `rewards` (`r`)
> -   `values` (`v`)
> -   `expectations` (`z`)
> -   `parameters` (`p`)

Symbol types that are present in a model are always listed in that
order.

### State-space

The unknown vector of controls $x$ is a function $\varphi$ of the
states, both exogenous ($m$) and endogenous ($s$) In
general we have:

$$x = \varphi(m, s)$$

In case the exogenous process is iid, dolo looks for a decision rule $x=\varphi(s)$.

!!! info

    This fact must be kept in mind when designing a model.
    TODO: explain how one can get the RBC wrong...

The function $\varphi$ is typically approximated by the solution
algorithm. It can be either a Taylor expansion, or an intepolating
object (splines, smolyak). In both cases, it behaves like a numpy gufunc
and can be called on a vector or a list of points:

``` {.python}
# for an iid model
dr = perturb(model)
m0, s0 = model.calibration['exogenous', 'states']
dr(m0, s0)                               # evaluates on a vector
dr(m0, s0[None,:].repeat(10, axis=0) )   # works on a list of points too
```


Equations
--------------


### Valid equations

The various equations that can be defined using these symbol types is
summarized in the following table. They are also reviewed below with
more details.


| Function                              | Standard name     | Short name | Definition                    |
| ------------------------------------- | ----------------- | ---------- | ----------------------------- |
| Transitions                           | `transition`      | `g`        | `s = g(m(-1), s(-1),x(-1),m)` |
| Lower bound                           | `controls_lb`     | `lb`       | `x_lb = lb(m, s)`             |
| Upper bound                           | `controls_ub`     | `ub`       | `x_ub = ub(m, s)`             |
| Utility                               | `utility`         | `u`        | `r = u(m,s,x)`                |
| Value updating                        | `alue_updating`   | `v`        | `w = v(s,x,v,s(1),x(1),w(1))` |
| Arbitrage                             | `arbitrage`       | `f`        | `0=f(m,s,x,m(1),s(1),x(1))`   |
| Expectations                          | `expectation`     | `h`        | `z=h(s(1),x(1))`              |
| Generalized  expectations             | `expectation_2`   | `h_2`      | `z=h_2(s,x,m(1),s(1),x(1))`   |
| Arbitrage  (explicit    expectations) | `arbitrage_2`     | `f_2`      | `0=f_2(s,x,z)`                |
| Direct response                       | `direct_response` | `d`        | `x=d(s,z)`                    |

When present these functions can be accessed from the `model.functions`
dictionary by using the standard name. For instance to compute the
auxiliary variables at the steady-state one can compute:

``` {.python}
# recover steady-state values
e = model.calibration['exogenous']
s = model.calibration['states']
x = model.calibration['controls']
p = model.calibration['parameters']

# compute the vector of auxiliary variables
a = model.functions['auxiliary']
y = a(e,s,x,p)

# it should correspond to the calibrated values:
calib_y = model.calibration['auxiliaries']
assert( abs(y - calib_y).max() < 0.0000001 )
```

### Equation Types

#### Transitions

    - name: `transition`
    - short name: `g`

Transitions are given by a function $g$ such that at all times:

$$s_t = g(m_{t-1}, s_{t-1}, x_{t-1}, m_t)$$

where $m_t$ is a vector of exogenous shocks

!!! example

    In the RBC model, the vector of states is $s_t=(a_t,k_t)$. The
    transitions are:

    $$\begin{eqnarray}a_t &= & \rho a_{t-1} + \epsilon_t\\
    k_t & = & (1-\delta) k_{t-1} + i_{t-1}\end{eqnarray}$$

    The yaml file is amended with:

    ``` {.yaml}
    symbols:
        states: [a,k]
        controls: [i]
        shocks: [ϵ]
        ...
    equations:
        transition:
            a[t] = rho*a[t-1] + ϵ[t]
            k = k[t-1]*(1-δ) + i[t-1]
    ```

Note that the transitions are given in the declaration order.

#### Auxiliary variables

    - name: `auxiliary`
    - short name: `a`

In order to reduce the number of variables, it is useful to define
auxiliary variables $y_t$ using a function $a$ such that:

$$y_t = a(m_t, s_t, x_t)$$

When they appear in an equation they are automatically substituted by
the corresponding expression in $m_t$, $s_t$ and $x_t$. Note that auxiliary
variables are not explicitely listed in the following definition.
Implicitly, wherever states and controls are allowed with the same date
in an equation type, then auxiliary variable are also allowed as long as the variables, they depend on are allowed. 

Auxiliary variables are defined in a special `definitions` block.

!!! example

    In the RBC model, three auxiliary variables are defined
    $y_t, c_t, r_{k,t}$ and $w_t$. They are a closed form function of
    $a_t, k_t, i_t, n_t$. Defining these variables speeds up computation
    since they are don't need to be solved for or interpolated.

#### Utility function and Bellman equation

    - name: `utility`
    - short name: `u`

The (separable) value equation defines the value $v_t$ of a given policy
as:

$$v_t = u(m_t, s_t,x_t) + \beta E_t \left[ v_{t+1} \right]$$

This gives rise to the Bellman equation:

> $$v_t = \max_{x_t} \left( u(m_t,s_t,x_t) + \beta E_t \left[ v_{t+1} \right] \right)$$

These two equations are characterized by the reward function $u$ and the
discount rate $\beta$. Function $u$ defines the vector of symbols
`rewards`. Since the definition of $u$ alone is not sufficient, the
parameter used for the discount factor must be given to routines that
compute the value. Several values can be computed at once, if $U$ is a
vector function and $\beta$ a vector of discount factors, but in that
case in cannot be used to solve for the Bellman equation.

!!! example

    Our RBC example defines the value as
    $v_t = \frac{(c_t)^{1-\gamma}}{1-\gamma} + \beta E_t v_{t+1}$. This
    information is coded using: \#\# TODO add labour to utility

    ``` {.yaml}
    symbols:
        ...
        rewards: [r]

    equations:
        ...
        utility:
            - r[t] = c[t]^(1-γ)/(1-γ)

    calibration:
        ...
        beta: 0.96   # beta is the default name of the discount
    ```


#### Value

    - name: `value`
    - short name: `w`

A more general updating equation can be useful to express non-separable
utilities or prices. the vector of (generalized) values $v^{*}$ are
defined by a function `w` such that:

$$v_t = w(m_t,s_t,x_t,v_t,m_{t+1},s_{t+1},x_{t+1},v_{t+1})$$

As in the separable case, this function can either be used to compute
the value of a given policy $x=\varphi()$ or in order solve the
generalized Bellman equation:

$$v_t = \max_{x_t} \left( w(m_t,s_t,x_t,v_t,m_{t+1},s_{t+1},x_{t+1},v_{t+1}) \right)$$

!!! example

    Instead of defining the rewards of the RBC example, one can instead
    define a value updating equation instead:

    ``` {.yaml}
    symbols:
        ...
        values: [v]

    equations:
        ...
        value:
            - v[t] = c[t]^(1-γ)/(1-γ)*(1-n[t]) + β*v[t+1]
    ```

#### Boundaries

    - name: `controls_lb` and `controls_ub`
    - short name: `lb` and `ub`

The optimal controls must also satisfy bounds that are function of
states. There are two functions $\underline{b}()$ and $\overline{b}()$
such that:

$$\underline{b}(m_t, s_t) \leq x_t \leq \overline{b}(m_t, /s_t)$$

!!! example

    In our formulation of the RBC model we have excluded negative
    investment, implying $i_t \geq 0$. On the other hand, labour cannot be
    negative so that we add lower bounds to the model:

    ``` {.yaml}
    equations:
        ...
        controls_lb:
            i = 0
            n = 0
    ```

    TODO: this makes no sense.

    Specifying the lower bound on labour actually has no effect since agents
    endogeneously choose to work a positive amount of time in order to
    produce some consumption goods. As for upper bounds, it is not necessary
    to impose some: the maximum amount of investment is limited by the Inada
    conditions on consumption. As for labour `n`, it can be arbitrarily
    large without creating any paradox. Thus the upper bounds are omitted
    (and internally treated as infinite values).

#### Euler equation

    - name: `arbitrage` (`equilibrium`)
    - short name: `f`

A general formulation of the Euler equation is:

$$0 = E_t \left[ f(m_t, s_t, x_t, m_{t+1}, s_{t+1}, x_{t+1}) \right]$$

Note that the Euler equation and the boundaries interact via
"complementarity conditions". Evaluated at one given state, with the
vector of controls $x=(x_1, ..., x_i, ..., x_{n_x})$, the Euler equation
gives us the residuals $r=(f_1, ..., f_i, ...,
f_{n_x})$. Suppose that the $i$-th control $x_i$ is supposed to lie in
the interval $[ \underline{b}_i, \overline{b}_i ]$. Then one of the
following conditions must be true:

-   $f_i$ = 0
-   $f_i<0$ and $x_i=\overline{b}_i$
-   $f_i>0$ and $x_i=\underline{b}_i$

By definition, this set of conditions is denoted by:

-   $f_i = 0 \perp \underline{b}_i \leq x_i \leq \overline{b}_i$

These notations extend to a vector setting so that the Euler equations
can also be written:

$$0 = E_t \left[ f(m_t, s_t, x_t, m_{t+1}, s_{t+1}, x_{t+1}) \right] \perp \underline{b}(m_t, s_t) \leq x_t \leq \overline{b}(m_t, s_t)$$

Specifying the boundaries together with Euler equation, or providing
them separately is exactly equivalent. In any case, when the boundaries
are finite and occasionally binding, some attention should be devoted to
write the Euler equations in a consistent manner. In particular, note
that the Euler equations are order-sensitive.

The Euler conditions, together with the complementarity conditions
typically often come from Kuhn-Tucker conditions associated with the
Bellman problem, but that is not true in general.

!!! example

    The RBC model has two Euler equations associated with investment and
    labour supply respectively. They are added to the model as:

    ``` {.yaml}
    arbitrage:
        - 1 - beta*(c[t]/c[t+1])^(sigma)*(1-delta+rk[t+1])  ⟂ 0 <= i[t] <= inf
        - w - chi*n[t]^eta*c[t]^sigma                       ⟂ 0 <= n[t] <= inf
    ```

    Putting the complementarity conditions close to the Euler equations,
    instead of entering them as separate equations, helps to check the sign
    of the Euler residuals when constraints are binding. Here, when
    investment is less desirable, the first expression becomes bigger. When
    the representative is prevented to invest less due to the constraint
    (i.e. $i_t=0$), the expression is then *positive* consistently with the
    complementarity conventions.


#### Expectations

    - name: `expectation`
    - short name: `h`

The vector of explicit expectations $z_t$ is defined by a function $h$
such that:

$$z_t = E_t \left[ h(m_{t+1}, s_{t+1},x_{t+1}) \right]$$

!!! example

    In the RBC example, one can define. the expected value tomorrow of one additional unit invested tomorrow:

    $$m_t=\beta c_{t+1}^{-\sigma}*(1-\delta+r_{k,t+1})$$

     It is a pure expectational variable in the sense that it is solely determined by future states and decisions. In the model file, it would be defined as:

    ```yaml

    symbols:
      ...
      expectations: [z]

    equations:
      expectations:
        - z = beta*(c[t+1])^(-sigma)*(1-delta+rk[t+1])
    ```

#### Generalized expectations

    - name: `expectation_2`
    - short name: `h_2`

The vector of generalized explicit expectations $z_t$ is defined by a
function $h^{\star}$ such that:

$$z_t = E_t \left[ h^{\star}(m_t, s_t,x_t,m_{t+1},s_{t+1},x_{t+1}) \right]$$

#### Euler equation with expectations

    - name: `arbitrage_2` (`equilibrium_2`)
    - short name: `f_2`

If expectations are defined using one of the two preceding definitions,
the Euler equation can be rewritten as:

$$0 = f(m_t, s_t, x_t, z_t) \perp \underline{b}(m_t, s_t) \leq x_t \leq \overline{b}(m_t, s_t)$$

!!! note

    Given the definition of the expectation variable $m_t$, today's
    consumption is given by: $c_t = z_t^{-\frac{1}{sigma}}$ so the Euler
    equations are rewritten as:

    ``` {.yaml}
    arbitrage_2:
        - 1 - beta*(c[t])^(sigma)/m[t]   | 0 <= i[t] <= inf
        - w[t] - chi*n[t]^eta*c[t]^sigma    | 0 <= n[t] <= inf
    ```

    Note the type of the arbitrage equation (`arbitrage_2` instead of
    `arbitrage`).

    However $c_t$ is not a control itself, but the controls $i_t, n_t$ can be easily deduced:

    $$\begin{eqnarray}
        n_t & =& ((1-\alpha) z_t k_t^\alpha \frac{m_t}{\chi})^{\frac{1}{\eta+\alpha}} \\
        i_t & = & z_t k_t^{\alpha} n_t^{1-\alpha} - (m_t)^{-\frac{1}{\sigma}}
    \end{eqnarray}$$

    This translates into the following YAML code:

    ``` {.yaml}
    arbitrage_2:
        - n[t] = ((1-alpha)*a[t]*k[t]^alpha*m[t]/chi)^(1/(eta+alpha))
        - i[t] = z[t]*k[t]^alpha*n[t]^(1-alpha) - m[t]^(-1/sigma)
    ```

#### Direct response function

    - name: `direct_response`
    - short name: `d`

In some simple cases, there a function $d()$ giving an explicit
definition of the controls:

$$x_t = d(m_t, s_t, z_t)$$

Compared to the preceding Euler equation, this formulation saves
computational time by removing the need to solve a nonlinear system to
recover the controls implicitly defined by the Euler equation.