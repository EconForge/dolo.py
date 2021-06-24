Parameterized expectations
==========================

We consider an fgh model, that is a model with the form:

> $s_t = g\left(s_{t-1}, x_{t-1}, \epsilon_t \right)$
>
> $0 = f\left(s_{t}, x_{t}, E_t[h(s_{t+1}, x_{t+1})] \right)$

where $g$ is the state transition function, $f$ is the arbitrage
equation, and $h$ is the expectations function (more accurately, it is
the function over which expectations are taken).

The parameterized expectations algorithm consists in approximating the
expectations function, $h$, and solving for the associated optimal
controls, $x_t = x(s_t)$.

At step $n$, with a current guess of the control, $x(s_t) = \varphi^n(s_t)$, and expectations function, $h(s_t,x_t) = \psi^n(s_t)$ :

:   -   Compute the conditional expectation $z_t = E_t[\varphi^n(s_t)]$
    -   Given the expectation, update the optimal control by solving
        $0 = \left( f\left(s_{t}, \varphi^{n+1}(s), z_t \right) \right)$
    -   Given the optimal control, update the expectations function
        $\psi^{n+1}(s_t) = h(s_t, \varphi^{n+1}(s_t))$

TODO: link to updated function.