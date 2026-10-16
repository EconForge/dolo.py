# dolo.py

Dolo is a Python package to describe and solve economic models. It provides a
simple classification scheme to describe many types of models, lets you write
models as simple text (YAML) files, and compiles these files into efficient
Python objects representing them. It provides many reference solution
algorithms to find the solution of these models under rational expectations:
local perturbations up to third order, perfect foresight solutions, policy
iteration, value iteration, and time iteration. Most algorithms are either
parallelized or vectorized, and are written in pure Python, so they can
easily be inspected or adapted.

Documentation: <https://econforge.github.io/dolo.py/>

## Install

Dolo is distributed as a conda package on the `econforge` channel on
prefix.dev. With [pixi](https://pixi.sh), in a new or existing project:

```console
pixi init my-project && cd my-project            # skip for an existing project
pixi workspace channel add --prepend https://prefix.dev/econforge
pixi add dolo
```

`--prepend` gives the `econforge` channel priority over `conda-forge`, so you
get the latest Dolo release. Dolo can of course also be installed from PyPI
with `pip install dolo`.

## Example

Write a model in `rbc.yaml`:

```yaml
name: Real Business Cycle

symbols:
  exogenous: [e_z]
  states: [z, k]
  controls: [n, i]
  parameters: [beta, sigma, eta, chi, delta, alpha, rho, zbar, sig_z]

definitions: |
  y[t] = exp(z[t])*k[t]^alpha*n[t]^(1-alpha)
  c[t] = y[t] - i[t]

equations:
  arbitrage: |
    chi*n[t]^eta*c[t]^sigma - w[t]                     ⟂ 0.0 <= n[t] <= inf
    1 - beta*(c[t]/c[t+1])^(sigma)*(1-delta+rk[t+1])   ⟂ -inf <= i[t] <= inf
```

Then import it, solve it, and simulate:

```python
from dolo import yaml_import, time_iteration, simulate

model = yaml_import("rbc.yaml")
dr = time_iteration(model)
sim = simulate(model, dr)

print(dr)
```

## Contributing

Contributors work from a clone of this repository. The sections below
describe the development setup.

### Setup

#### Using pixi (production and development environment)

`pixi` can be installed on a *nix system using the following command:

```console
curl -fsSL https://pixi.sh/install.sh | sh
```

The `dev` environment brings together the test, lint, docs, and release
features:

```console
pixi install -e dev
```

To run unit and coverage tests:

```console
pixi run -e dev test
pixi run -e dev cov
```

To format the code and check the types:

```console
pixi run -e dev black
pixi run -e dev typecheck
```

To build and serve the documentation locally with `zensical`:

```console
pixi run -e dev docs
```

To launch JupyterLab with the package installed in editable mode:

```console
pixi run -e dev jlab
```
