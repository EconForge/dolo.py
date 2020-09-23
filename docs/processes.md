# Shocks

The type of exogenous shock associated to a model determines the kind of decision rule, which will be obtained by the solvers. Shocks can pertain to one of the following categories:

- continuous i.i.d. shocks aka distributions

- continuous auto-correlated process such as AR1

- discrete processes such as discrete markov chains


Exogenous shock processes are specified in the section `exogenous` of a yaml file.

Here are some examples for each type of shock:

## Distributions / IIDProcess

### Univariate distributions

#### IID Normal

The type of the shock is specified using yaml type annotations (starting with exclamation mark)

Normal distribution with mean mu and variance σ^2 has the probability density function

$$f(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}}
\exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right)$$

A normal shock in the yaml file with mean 0.2 and standard deviation 0.1 can be declared as follows

```
!Normal:
    σ: 0.1
    μ: 0.2
```

or

```
!Normal:
    sigma: 0.1
    mu: 0.2
```

!!! note
    Greek letter 'σ' or 'sigma' (similarly 'μ' or 'mu' ) are accepted.


!!! note

      When defining shocks in a dolo model, that is in an `exogenous` section, The exogenous shock section can refer to parameters specified in the calibration section:

      ```   
      symbols:
      ...
            parameters: [alpha, beta, mu, sigma]
      ...
      calibration:
            sigma: 0.01
            mu: 0.0

      exogenous: !Normal:
            σ: sigma
            μ: mu

      ```      

#### IID LogNormal

Parametrization of a lognormal random variable Y is in terms of he mean, μ, and standard deviation, σ, of the unique normally distributed random variable X is as follows

$$f(x; \mu, \sigma) = \frac{1}{x \sqrt{2 \pi \sigma^2}}
\exp \left( - \frac{(\log(x) - \mu)^2}{2 \sigma^2} \right),
\quad x > 0$$

such that exp(X) = Y

```
exogenous: !LogNormal:
      σ: sigma
      μ: mu

```    

#### Uniform

Uniform distribution over an interval [a,b]

$$f(x; a, b) = \frac{1}{b - a}, \quad a \le x \le b$$


```
symbols:
      states: [k]
      controls: [c, d]
      exogenous: [e]
      parameters: [alpha, beta, mu, sigma, e_min, e_max]

.
.
.

exogenous: !Uniform:
      a: e_min
      b: e_max

```    

#### Beta

If X∼Gamma(α) and Y∼Gamma(β) are distributed independently, then X/(X+Y)∼Beta(α,β).

Beta distribution with shape parameters α and β has the following PDF

$$f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)} x^{\alpha - 1} (1 x)^{\beta - 1}, \quad x \in [0, 1]$$

```
exogenous: !Beta
      α: 0.3
      β: 0.1

```    

#### Bernouilli

Binomial distribution parameterized by $p$ yields $1$ with probability $p$ and $0$ with probability $1-p$.

```
!Bernouilli
      π: 0.3
```   
### Multivariate distributions

#### Normal (multivariate)

Note the difference with `UNormal`. Parameters `Σ` (not `σ`) and `μ` take a matrix and a vector respectively as argument.
```
!Normal:
      Σ: [[0.0]]
      μ: [0.1]
```


### Mixtures

For now, mixtures are defined for i.i.d. processes only. They take an integer valued distribution (like the Bernouilli one) and a different distribution associated to each of the values.

```yaml
exogenous: !Mixture
    index: !Bernouilli
        p: 0.3
    distributions:
        0: UNormal(μ=0.0, σ=0.01)
        1: UNormal(μ=0.0, σ=0.02)
```

Mixtures are not restricted to 1d distributions, but all distributions of the mixture must have the same dimension.

!!! note

Right now, mixtures accept only distributions as values. To switch between constants, one can use a `Constant` distribution as in the following examples.

```yaml
exogenous:
    e,v: !Mixture:
        index: !Bernouilli
            p: 0.3
        distributions:
            0: Constant(μ=[0.1, 0.2])
            1: Constant(μ=[0.2, 0.3])
```

## Continuous Autoregressive Process

### AR1 / VAR1

For now, `AR1` is an alias for `VAR1`. Autocorrelation `ρ` must be a scalar (otherwise we don't know how to discretize).

```yaml
exogenous: !AR1
    rho: 0.9
    Sigma: [[σ^2]]
```


## Markov chains

Markov chains are constructed by providing a list of nodes and a
transition matrix.

```yaml
exogenous: !MarkovChain
    values: [[-0.01, 0.1],[0.01, 0.1]]
    transitions: [[0.9, 0.1], [0.1, 0.9]]
```


## Product

We can also specify more than one process. For instance if we want to combine a VAR1 and an Normal Process we use the tag Product and write:

```
exogenous: !Product

    - !VAR1
         rho: 0.75
         Sigma:  [[0.015^2, -0.05], [-0.05, 0.012]]

    -  !Normal:
          σ: sigma
          μ: mu
```

!!! note

      Note that another syntax is accepted, in the specific context of a dolo exogenous section. It keeps the Product operator implicit. Suppose a dolo model has $r,w,e$ as exogenous shocks. It is possible to list several shocks for each variable as in the following example:

      ```
      symbols:
            ...
            exogenous: [r,w,e]

      exogenous:
          r,w: !VAR1
               rho: 0.75
               Sigma: [[0.015^2, -0.05], [-0.05, 0.012]]

          e  !Normal:
                σ: sigma
                μ: mu
      ```

      In this case we define several shocks for several variables (or combinations thereof).

## Conditional processes

Support is very limited for now. It is possible to define markov chains, whose transitions (not the values) depend on the output of another process.

```
exogenous: !Conditional
    condition: !UNormal
        mu: 0.0
        sigma: 0.2
    type: Markov
    arguments: !Function
        arguments: [x]
        value:
          states: [0.1, 0.2]
          transitions: !Matrix
              [[1-0.1-x, 0.1+x],
               [0.5,       0.5]]

```

!!! note

      The plan is to replace the clean and explicit but somewhat tedious syntax above by the following (where dependence is detected automatically):

      ```
      exogenous:
          x: !UNormal
              mu: 0.0
              sigma: 0.2
          y: !Markov
                states: [0.1, 0.2]
                transitions: !Matrix
                    [[1-0.1-x, 0.1+x],
                     [0.5,       0.5]]

      ```


## Discretization methods for continous shocks

To solve a non-linear model with a given exogenous process, one can apply different types of procedures to discretize the continous process:

| Type | Distribution | Discretization procedure             |
|--------------|--------------|-----------------------------------|
|Univariate iid| UNormal(μ, σ)| Equiprobable, Gauss-Hermite Nodes |
|Univariate iid| LogNormal(μ, σ) |Equiprobable |
|Univariate iid| Uniform(a, b ) |Equiprobable|
|Univariate iid| Beta(α, β)   |Equiprobable |
|Univariate iid| Beta(α, β)   |Equiprobable |
| VAR1 |   |Generalized Discretization Method (GDP), Markov Chain |

!!! note
    Here we can define shortly each method. Then perhaps link to a jupyter notebook as discussed: Conditional on the discretization approach, present the results of the corresponding method solutions and simulations. Discuss further discretization methods and related dolo objects.
