The dolo language
=================

The easiest way to code a model in dolo consists in using specialized
Yaml files also referred to as dolo model files.

YAML format
-----------

YAML stands for Yet Another Markup Language. It is a serialization
language that allows complex data structures in a human-readable way.
Atomic elements are floats, integers and strings. An ordered list can be
defined by separating elements with commas and enclosing them with
squere brackets:

``` yaml
[1,2,3]
```

Equivalently, it can be done on several line, by prepending
[-]{.title-ref} to each line

``` {.yaml}
- 'element'
- element         # quotes are optional there is no ambiguity
- third element   # this is interpreted as ``'third element'``
```

Associative arrays map keys to (simple strings to arbitrary values) as
in the following example:

``` {.yaml}
{age: 18, name: peter}
```

Mappings can also be defined on severaly lines, and as in Python,
structures can be nested by using indentation (use spaces no tabs):

``` {.yaml}
age: 18
name: peter
occupations:
  - school
  - guitar
friends:
  paula: {age: 18}
```

The correspondance between the yaml definition and the resulting Python
object is very transparent. YAML mappings and lists are converted to
Python dictionaries and lists respectiveley.

!!! note
    TODO say something about YAML objects


Any model file must be syntactically correct in the Yaml sense, before
the content is analysed further. More information about the YAML syntax
can be found on the [YAML website](http://www.yaml.org/), especially
from the [language specification](http://www.yaml.org/).

Example
-------

Here is an example model contained in the file
`examples\models\rbc.yaml`

```
--8<-- "examples/models/rbc.yaml"
```


This model can be loaded using the command:

``` {.python}
model = yaml_import(`examples/models/rbc.yaml`)
```

The function `yaml_import` (cross) will raise errors until
the model satisfies basic compliance tests. . In the
following subsections, we describe the various syntactic rules prevailing
while writing yaml files.

Sections
--------

A dolo model consists in the following 4 or 5 parts:

-   a `symbols` section where all symbols used in the model
    must be defined
-   an `equations` section containing the list of equations
-   a `calibration` section providing numeric values for the
    symbols
-   a `domain` section, with the information about the
    solution domain
-   an `options` section containing additional informations
-   an `exogenous` section where exogenous shocks are defined.

These section have context dependent rules. We now review each of them in detail:

### Declaration section

This section is introduced by the `symbols`]{.title-ref}` keyword. All
symbols appearing in the model must be defined there.

Symbols must be valid Python identifiers (alphanumeric not beginning
with a number) and are case sensitive. Greek letters  are recognized. Subscripts and
superscripts can be denoted by `_` and `__`
respectively. For instance `beta_i_1__d` will be pretty
printed as $\beta_{i,1}^d$. Unicode characters are accepted too, as long as they are valid, when used within python identifiers.

!!! note

    In most modern text editor, greek characters can be typed, by entering their latex representation (like `beta`) and pressing Tab.

Symbols are sorted by type as in the following example:

``` {.yaml}
symbols:
  states: [a, b]
  controls: [u, v]
  exogenous: [e]
  parameters: [rho]
```

Note that each type of symbol is associated with a symbol list (like `[a,b]`).

!!! alert
    A common mistake consists in forgetting the commas, and use spaces only inside list. This doesn't work since the space will be ignored and the two symbols recognized as one.

The exact list of symbols to declare depends on which algorithm is meant to be used. In general, one needs to supply at least *states* (endogenous states), *exogenous* (for exogenous shocks), *controls* for 
decision variables, and *parameters* for scalar parameters, that the model can depend on.

### Declaration of equations

The `equations` section contains blocks of equations sorted by type.

Expressions follow (roughly) the Dynare conventions. Common arithmetic
operators ([+,-,\*,/,\^]{.title-ref}) are allowed with conventional
priorities as well as usual functions (`sqrt`, `log`, `exp`, `sin`, `cos`, `tan`,
`asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`).
The definitions of these functions match the definitions from the
`numpy` package.

All symbols appearing in an expression must
either be declared in the `symbols` section or be one of the predefined functions. Parameters (that are time invariant) are ntot subscripted, while all other symbol types, are variables, indexed by time. A variable `v` appear as `v[t-1]` (for $v_{t-1}$), `v[t]` (for $v_t$), 
or `v[t+1]` (for $v_t$).

<!-- Any symbol [s]{.title-ref} that is not a parameter
is assumed to be considered at date [t]{.title-ref}. Values at date
[t+1]{.title-ref} and [t-1]{.title-ref} are denoted by
[s(1)]{.title-ref} and [s(-1)]{.title-ref} respectively. -->

All equations are implicitly enclosed by the expectation operator
$E_t\left[\cdots \right]$. Consequently, the law of motion for the capital

$$k_{t+1} = (1-\delta) k_{t} +  i_{t} + \epsilon_t$$

is written (in a `transition` section) as:


``` {.yaml}
k[t] = (1-δ)*k[t-1] + i[t-1]
```

while the Euler 

$$E_t \left[ \beta \left( \frac{c_{t+1}}{c_t} + (1-\delta)+r_{t+1} \right) \right] - 1$$

would be written (in the `arbitrage` section) as:

``` {.yaml}
β*(c[t]/c[t+1])^(σ)*(1-δ+r[t+1]) - 1   # note that expectiation is operator
```

!!! note

    In Python, the exponent operator is denoted by [\*\*]{.title-ref} while
    the caret operator [\^]{.title-ref} represents bitwise XOR. In dolo
    models, we ignore this distinction and interpret both as an exponent.

!!! note

    The default evaluator in dolo preserves the evaluation order. Thus
    `(c[t+1]/c[t])^(-gamma)` is not evaluated in the same way (and is numerically
    more stable) than `c(1)^(-gamma)/c^(-gamma)`. Currently, this is not
    true for symbolically computed derivatives, as expressions are
    automatically simplified, implying that execution order is not
    guaranteed. This impacts only higher order perturbations.


An equation can consist of one expression, or two expressions separated
by [=]{.title-ref}. There are two types of equation blocks:

- __Condition blocks__: in these blocks, each equation `lhs = rhs` define the scalar value `(rhs)-(lhs)`. A list of of such equations, i.e a block, defines a multivariate function of the appearing symbols. Certain     condition blocks, can be associated with complementarity conditions separated by `⟂` (or `|`) as in  `rhs-lhs ⟂ 0 < x < 1`. In this case it is advised to omit the equal sign in order to make it easier to interpret the complementarity. Also, when complementarity conditions are used, the ordering of variables appearing in the complementarities must match the declaration order (more in section Y).
- __Definition blocks__: the differ from condition blocks in that they define a group of variables (`states` or `auxiliaries`) as a function of the right hand side.

The types of variables appearing on the right hand side depend on the
block type. The variables enumerated on the left hand-side must appear
in the declaration order.

!!! note

    In the RBC example, the `defintitions` block defines variables (`y,c,rk,w`)
    that can be directly deduced from the states and the controls:

    ``` {.yaml}
    definitions:
        - y[t] = z[t]*k[t]^alpha*n[t]^(1-alpha)
        - c[t] = y[t] - i[t]
        - rk[t] = alpha*y[t]/k[t]
        - w[t] = (1-alpha)*y[t]/w[t]
    ```

Note that the declaration order matches the order in which variables
appear on the left hand side. Also, these variables are defined
recursively: `c`, `rk` and `w` depend on the value for `y`. In contrast
to the calibration block, the definition order matters. Assuming that
variables where listed as (`c,y,rk,w`) the following block would provide
incorrect result since `y` is not known when `c` is evaluated.

``` {.yaml}
definitions:
    - c[t] = y[t] - i[t]
    - y[t] = z[t]*k[t]^alpha*n[t]^(1-alpha)
    - rk[t] = alpha*y[t]/k[t]
    - w[t] = (1-alpha)*y[t]/w[t]
```

### Calibration section

The role of the calibration section consists in providing values for the
parameters and the variables. The calibration of all parameters
appearing in the equation is of course strictly necessary while the the
calibration of other types of variables is useful to define the
steady-state or an initial guess to the steady-state.

The calibrated values are also substituted in other sections, including
`exgogenous` and `options` sections. This is
particularly useful to make the covariance matrix depend on model
parameters, or to adapt the state-space to the model's calibration.

The calibration is given by an associative dictionary mapping symbols to
define with values. The values can be either a scalar or an expression.
All symbols are treated in the same way, and values can depend upon each
other as long as there is a way to resolve them recursively.

In particular, it is possible to define a parameter in order to target a
special value of an endogenous variable at the steady-state. This is
done in the RBC example where steady-state labour is targeted with
`n: 0.33` and the parameter `phi` calibrated so that the optimal labour
supply equation holds at the steady-state (`chi: w/c^sigma/n^eta`).

All symbols that are defined in the [symbols]{.title-ref} section but do not appear in the calibration section are initialized with the value [nan]{.title-ref} without issuing any warning.

!!! note

    No clear long term policy has been established yet about how to deal with undeclared symbols in the calibration section. Avoid them. TODO: reevaluate

### Domain section

The domain section contains boundaries for each endogenous state as in
the following example:

``` {.yaml}
domain:
    k: [0.5*k, 2*k]
    z: [-σ_z*3, σ_z*3]
```

!!! note

    In the above example, values can refer to the calibration dictionary.   Hence, `0.5 k` means `50%` of steady-state  `k`. Keys, are not replaced.


### Exogenous shocks specification

!!! alert
    This section is out-of-date. Syntax has changed.
    Many more shocks options are allowed. See [processes](processes.md) for a more recent description of the shocks.

    TODO: redo


The type of exogenous shock associated to a model determines the kind of decision rule, whih will be obtained by the solvers. Shocks can pertain to one of the following categories: continuous i.i.d. shocks (Normal law), continous autocorrelated process (VAR1 process) or a discrete markov chain. The type of the shock is specified using yaml type annotations (starting with exclamation mark) The exogenous shock section can refer to parameters specified in the calibration section. Here are some examples for each type of shock:

#### Normal

For Dynare and continuous-states models, one has to specifiy a
multivariate distribution of the i.i.d. process for the vector of
exogenous  variaibles (otherwise they are assumed to be constantly equal to zero). This is done
in the `exogenous` section. A gaussian distrubution (only one supported
so far), is specified by supplying the covariance matrix as a list of
list as in the following example.

``` {.yaml}
exogenous: !Normal:
    Sigma: [ [sigma_1, 0.0],
            [0.0, sigma_2] ]
```
!!! alert

    The shocks syntax is currently rather unforgiving. Normal shocks expect
    a covariance matrix (i.e. a list of list) and the keyword is `Sigma` not `sigma`.

#### Markov chains

Markov chains are constructed by providing a list of nodes and a
transition matrix.

``` {.yaml}
exogenous: !MarkovChain
    values: [[-0.01, 0.1],[0.01, 0.1]]
    transitions: [[0.9, 0.1], [0.1, 0.9]]
```

It is also possible to combine markov chains together.

``` {.yaml}
exogenous: !Product
    - !MarkovChain
        values: [[-0.01, 0.1],[0.01, 0.1]]
        transitions: [[0.9, 0.1], [0.1, 0.9]]
    - !MarkovChain
        values: [[-0.01, 0.1],[0.01, 0.1]]
        transitions: [[0.9, 0.1], [0.1, 0.9]]
```

### Options

The [options]{.title-ref} section contains all informations necessary to
solve the model. It can also contain arbitrary additional informations.
The section follows the mini-language convention, with all calibrated
values replaced by scalars and all keywords allowed.

Global solutions require the definition of an approximation space. The
lower, upper bounds and approximation orders (number of nodes in each
dimension) are defined as in the following example:

``` {.yaml}
options:
    grid: !Cartesian
        n: [10, 50]
    arbitrary_information: 42
```
